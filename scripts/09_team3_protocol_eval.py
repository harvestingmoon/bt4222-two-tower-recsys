"""Protocol-matched comparison against myathetchai/bt4222-team3-jdsearch.

team3 (https://github.com/myathetchai/bt4222-team3-jdsearch) reports
``Hit Rate@10: 0.9782`` / ``NDCG@10: 0.5684`` for its Hybrid NCF + LSTM.  Those
numbers are **not** comparable to a full-slate number, because:

1. ``Data Preparation/1. Sampling.ipynb`` keeps all positives and randomly draws
   ``4 * n_pos`` negatives per query, so their slate shrinks from ~89 displayed
   items to mean 9.17 / median 5 (their own cell-93 ``describe()``).
2. ``Models/3.1_Hybrid_NCF_with_LSTM.ipynb`` uses a 70/15/15 user split
   (test = 25,760 of the 171,728 users that have a positive).  Our split is
   80/10/10 -> 17,174 test users.
3. The eval loop sets ``K = 5`` but prints the label ``NDCG@10`` / ``Hit Rate@10``,
   so 0.5684 is really **NDCG@5** on a 9-item list.

This script re-scores our saved per-candidate test/val predictions on team3's
1:4 protocol so the two numbers sit in the same table.

Outputs
-------
outputs/full/results/team3_protocol.csv      report-ready comparison table
outputs/full/results/team3_protocol.json     same numbers + val split + oracle
outputs_v7/full/results/team3_protocol.*     working copies

Run:  python scripts/09_team3_protocol_eval.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("outputs_v7/full")
PUB = Path("outputs/full/results")  # tracked copy used by the notebooks/report
RES = OUT / "results"

SEED = 42
# numbers copied verbatim from their notebook output (cell 94 of 3.1_..._LSTM)
THEIR = {
    "system": "team3 Hybrid NCF + LSTM (their notebook)",
    "protocol": "team3 1:4 sampled",
    "n_users": 25760,
    "mean_slate": 9.17,
    "MRR": None,
    "P@1": None,
    "NDCG@5": 0.5684,
    "NDCG@10": None,
    "HR@10": 0.9782,
    "source": "their notebook 3.1 (K=5 printed as @10)",
}


def ndcg_at_k(ranked_labels: np.ndarray, k: int) -> float:
    top = ranked_labels[:k]
    disc = np.log2(np.arange(2, len(top) + 2))
    dcg = float(np.sum((2.0 ** top - 1.0) / disc))
    ideal = np.sort(ranked_labels)[::-1][: len(top)]
    idcg = float(np.sum((2.0 ** ideal - 1.0) / disc))
    return dcg / idcg if idcg > 0 else 0.0


def restore_sort(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by user then candidate id so the 1:4 sampler sees a stable order."""
    return df.sort(["user_id", "wid"])


def sample_1_to_4(df: pl.DataFrame, seed: int = SEED) -> pl.DataFrame:
    """team3 'Sampling 2': keep every positive, sample 4 * n_pos negatives."""
    rng = np.random.default_rng(seed)
    users = df["user_id"].to_numpy()
    labels = df["label"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, users[1:] != users[:-1], True])
    keep_rows: list[np.ndarray] = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        lab = labels[lo:hi]
        pos = np.flatnonzero(lab > 0)
        neg = np.flatnonzero(lab == 0)
        k = min(4 * len(pos), len(neg))
        picked = rng.choice(neg, size=k, replace=False) if k > 0 else np.array([], dtype=np.int64)
        keep_rows.append(lo + np.sort(np.concatenate([pos, picked])))
    return df[np.concatenate(keep_rows)]


def evaluate(df: pl.DataFrame) -> dict:
    """df: user_id, label, score with one row per candidate."""
    labels = df["label"].to_numpy()
    scores = df["score"].to_numpy()
    users = df["user_id"].to_numpy()

    order = np.lexsort((scores, users))  # user asc, score asc -> reverse per user
    users, labels, scores = users[order], labels[order], scores[order]
    bounds = np.flatnonzero(np.r_[True, users[1:] != users[:-1], True])

    ndcg5, ndcg10, hr10, mrr, p1, slate = [], [], [], [], [], []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        lab = labels[lo:hi][::-1]  # descending score
        if (lab > 0).sum() == 0:
            continue
        slate.append(len(lab))
        ndcg5.append(ndcg_at_k(lab, 5))
        ndcg10.append(ndcg_at_k(lab, 10))
        hr10.append(float((lab[:10] > 0).any()))
        first = np.flatnonzero(lab > 0)
        mrr.append(1.0 / (first[0] + 1) if first.size and first[0] < 10 else 0.0)
        p1.append(float(lab[0] > 0))

    return {
        "n_users": len(ndcg10),
        "mean_slate": round(float(np.mean(slate)), 2),
        "MRR": round(float(np.mean(mrr)), 4),
        "P@1": round(float(np.mean(p1)), 4),
        "NDCG@5": round(float(np.mean(ndcg5)), 4),
        "NDCG@10": round(float(np.mean(ndcg10)), 4),
        "HR@10": round(float(np.mean(hr10)), 4),
    }


def rows_for_split(split: str) -> list[dict]:
    """One row per (system, protocol) for the given split."""
    rows: list[dict] = []

    # --- two-tower alone -------------------------------------------------
    tower_path = OUT / "tower" / "tower_scores.parquet"
    if tower_path.exists():
        tower = (
            pl.read_parquet(tower_path, columns=["user_id", "wid", "label", "tower_score", "tower_model", "split"])
            .filter(pl.col("split") == split)
            .filter(pl.col("tower_model") == "full")
            .select("user_id", "wid", "label", pl.col("tower_score").alias("score"))
        )
        tower = restore_sort(tower)
        rows.append({"system": "two-tower only", "protocol": "full slate", **evaluate(tower)})
        rows.append({"system": "two-tower only", "protocol": "team3 1:4 sampled", **evaluate(sample_1_to_4(tower))})

    # --- two-tower + LambdaRank -----------------------------------------
    ranked = restore_sort(pl.read_parquet(OUT / "predictions" / f"ranker_{split}.parquet"))
    rows.append({"system": "two-tower + LambdaRank", "protocol": "full slate", **evaluate(ranked)})
    sampled = sample_1_to_4(ranked)
    rows.append({"system": "two-tower + LambdaRank", "protocol": "team3 1:4 sampled", **evaluate(sampled)})

    # --- oracle ceiling on the sampled slates ---------------------------
    oracle = sampled.with_columns(score=pl.col("label").cast(pl.Float32))
    rows.append({"system": "oracle (labels as scores)", "protocol": "team3 1:4 sampled", **evaluate(oracle)})

    for r in rows:
        r["source"] = "this repo (saved predictions re-scored)"
        r["split"] = split
    return rows


def main() -> None:
    rows = rows_for_split("test")
    val_rows = rows_for_split("val")

    # team3's own reported number, for the side-by-side table
    compare = [dict(THEIR, split="test")] + rows

    order = ["system", "protocol", "split", "n_users", "mean_slate", "MRR", "P@1", "NDCG@5", "NDCG@10", "HR@10", "source"]
    table = pl.DataFrame([{k: r.get(k) for k in order} for r in compare], schema_overrides={"n_users": pl.Int64})
    table.write_csv(RES / "team3_protocol.csv")

    payload = {
        "note": (
            "team3 samples 1:4 positives:negatives per query (mean slate 9.17 vs our 89.2) "
            "and prints K=5 metrics under an '@10' label, so its 0.5684 is NDCG@5 on a 9-item list."
        ),
        "params": {"seed": SEED, "neg_ratio": 4, "k_ndcg": [5, 10]},
        "comparison_test": compare,
        "detail_val": val_rows,
        "detail_test": rows,
    }
    (RES / "team3_protocol.json").write_text(json.dumps(payload, indent=2))

    PUB.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RES / "team3_protocol.csv", PUB / "team3_protocol.csv")
    shutil.copy2(RES / "team3_protocol.json", PUB / "team3_protocol.json")

    print(table)
    print("\nwrote", RES / "team3_protocol.csv", "and", PUB / "team3_protocol.csv")


if __name__ == "__main__":
    main()
