"""Ranking metrics on full candidate lists (the official JDsearch protocol).

Every model produces a `score` for each (user_id, wid) row of `candidates.parquet`.
Items are ranked per user by descending score; ties are broken by a seeded random key,
never by list position (the raw list order leaks the label, see step 0).

Reported metrics (averaged over users with at least one positive):
  * MRR@200   - reciprocal rank of the first relevant item (paper metric)
  * P@1       - is the top-ranked item relevant (paper metric)
  * NDCG@10   - binary gain (paper metric)
  * gNDCG@10  - graded gain 2^label - 1 (click=1, cart=2, purchase=3)
  * HR@10     - at least one relevant item in the top 10
  * R@k       - recall of relevant items within the top k
"""
import numpy as np
import polars as pl

LOG2 = 1.0 / np.log2(np.arange(2, 202))  # discount for ranks 1..200


def add_ranks(df: pl.DataFrame, score_col: str = "score", seed: int = 0) -> pl.DataFrame:
    """Attach a 1-based `rank` per user (higher score -> smaller rank)."""
    rng = np.random.default_rng(seed)
    return (df.with_columns(_tie=pl.Series(rng.random(df.height)))
              .sort(["user_id", score_col, "_tie"], descending=[False, True, False])
              .with_columns(rank=pl.int_range(1, pl.len() + 1).over("user_id"))
              .drop("_tie"))


def per_user_metrics(df: pl.DataFrame, score_col: str = "score", seed: int = 0,
                     recall_ks=(10, 50)) -> pl.DataFrame:
    """One row per user with every metric; users without positives are dropped."""
    r = add_ranks(df.select("user_id", "label", score_col), score_col, seed)
    disc = pl.lit(1.0) / (pl.col("rank") + 1).cast(pl.Float64).log(2)
    rel = (pl.col("label") > 0).cast(pl.Float64)
    gain = (2.0 ** pl.col("label").cast(pl.Float64)) - 1.0
    top10 = pl.col("rank") <= 10

    agg = r.group_by("user_id").agg(
        n_pos=rel.sum(),
        first_hit=pl.col("rank").filter(pl.col("label") > 0).min(),
        p1=rel.filter(pl.col("rank") == 1).sum(),
        dcg=(rel * disc).filter(top10).sum(),
        gdcg=(gain * disc).filter(top10).sum(),
        labels=pl.col("label").sort(descending=True).head(10),
        **{f"hits{k}": rel.filter(pl.col("rank") <= k).sum() for k in recall_ks},
    ).filter(pl.col("n_pos") > 0)

    def idcg(labels, graded):
        out = np.empty(len(labels))
        for i, l in enumerate(labels):
            l = np.asarray(l, dtype=np.float64)
            g = (2.0 ** l - 1.0) if graded else (l > 0).astype(np.float64)
            out[i] = (g * LOG2[: len(g)]).sum()
        return out

    labels = agg["labels"].to_list()
    agg = agg.with_columns(
        idcg=pl.Series(idcg(labels, False)), gidcg=pl.Series(idcg(labels, True)))
    return agg.select(
        "user_id",
        mrr=1.0 / pl.col("first_hit").cast(pl.Float64),
        p1=pl.col("p1"),
        ndcg10=pl.col("dcg") / pl.col("idcg"),
        gndcg10=pl.col("gdcg") / pl.col("gidcg"),
        hr10=(pl.col(f"hits{recall_ks[0]}") > 0).cast(pl.Float64),
        **{f"r{k}": pl.col(f"hits{k}") / pl.col("n_pos") for k in recall_ks},
    )


def evaluate(df: pl.DataFrame, score_col: str = "score", seed: int = 0, name: str | None = None) -> dict:
    """Mean of per-user metrics, formatted like the paper (MRR, P@1, NDCG@10) plus extras."""
    pu = per_user_metrics(df, score_col, seed)
    res = {k: float(pu[k].mean()) for k in pu.columns if k != "user_id"}
    res["n_users"] = pu.height
    if name:
        res = {"model": name, **res}
    return res


def results_table(rows: list[dict]) -> pl.DataFrame:
    order = ["model", "mrr", "p1", "ndcg10", "gndcg10", "hr10", "r10", "r50", "n_users"]
    df = pl.DataFrame(rows)
    return df.select([c for c in order if c in df.columns]).with_columns(
        pl.col(c).round(4) for c in ["mrr", "p1", "ndcg10", "gndcg10", "hr10", "r10", "r50"] if c in df.columns)
