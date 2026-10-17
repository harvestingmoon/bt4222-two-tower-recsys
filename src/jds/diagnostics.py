"""Stage 7: where the model helps and why -> <scope>/results/diag_*.csv|json and figures/diag_*.png

  * segments  - test metrics by user / query segment (activity in the click window, history length, query frequency)
  * ablations - rankers retrained without one feature group at a time (same user subsample for all)
  * shap      - LightGBM's exact TreeSHAP contributions (pred_contrib) on test rows, aggregated by feature and group
  * attention - how the full tower distributes query-guided attention over the three history segments
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from . import features  # noqa: E402
from .config import get_device, get_logger, load_json, paths, save_json  # noqa: E402
from .metrics import evaluate, per_user_metrics, results_table  # noqa: E402
from .prepare import load  # noqa: E402
from .ranker import dataset  # noqa: E402
from .ranker import infer as rinfer  # noqa: E402
from .ranker import train as rtrain  # noqa: E402

GROUPS = {   # feature-name prefixes per group (the distinct data-field roles in features.py)
    "two-tower score": ("tower_",),
    "personal history (per type window)": ("u_wid_", "u_brand_id_", "u_shop_id_", "u_cate_id_4_", "u_cart_not_ordered"),
    "similar past queries": ("u_q_", "u_simq_"),
    "query -> item history (all users)": ("qpop_",),
    "impression CTR (out-of-fold)": ("ctr_", "gain_", "disp_"),
    "popularity (per type window)": ("ipop_", "apop_"),
    "lexical": ("lex_",),
    "slate context": ("slate_",),
    "user activity": ("ua_",),
    "item content": ("item_",),
}


def group_of(name: str) -> str:
    base = name[:-4] if name.endswith("_pct") and not name.startswith("tower_") else name
    for g, pre in GROUPS.items():
        if base.startswith(pre):
            return g
    return "other"


def segments(cfg, p, log) -> pl.DataFrame:
    users = load(p.data, "users")
    test = users.filter((pl.col("split") == "test") & pl.col("has_pos"))
    tr_q = users.filter(pl.col("split") == "train").group_by("qid").len().rename({"len": "q_train_users"})
    hist = load(p.data, "history", ["user_id", "type", "age_days"])
    act = hist.group_by("user_id").agg(clicks_7=((pl.col("type") == 1) & (pl.col("age_days") < 7)).sum())
    seg = (test.select("user_id", "qid", "hist_len").join(tr_q, on="qid", how="left").join(act, on="user_id", how="left")
           .with_columns(pl.col("q_train_users", "clicks_7").fill_null(0))
           .select("user_id",
                   activity=pl.when(pl.col("clicks_7") > 0).then(pl.lit("clicked in last 7 days")).otherwise(pl.lit("no click in last 7 days")),
                   history=pl.when(pl.col("hist_len") < 50).then(pl.lit("history < 50")).when(pl.col("hist_len") < 200)
                   .then(pl.lit("history 50-199")).otherwise(pl.lit("history >= 200")),
                   query=pl.when(pl.col("q_train_users") == 0).then(pl.lit("query unseen in training"))
                   .when(pl.col("q_train_users") < 10).then(pl.lit("query seen 1-9x")).otherwise(pl.lit("query seen >= 10x"))))
    models = {"ranker": pl.read_parquet(p.results / "per_user_ranker_test.parquet")}
    ts = pl.read_parquet(p.tower / "tower_scores.parquet").filter(pl.col("split") == "test")
    models["two-tower"] = per_user_metrics(ts, "tower_score", seed=cfg["seed"])
    pop = features.read(p, columns=["user_id", "label", "ipop_users"], splits=["test"])
    models["popularity"] = per_user_metrics(pop, "ipop_users", seed=cfg["seed"])
    rows = []
    for dim in ("activity", "history", "query"):
        for name, pu in models.items():
            d = pu.join(seg, on="user_id").group_by(dim).agg(n=pl.len(), mrr=pl.col("mrr").mean(), gndcg10=pl.col("gndcg10").mean())
            rows.append(d.rename({dim: "segment"}).with_columns(dimension=pl.lit(dim), model=pl.lit(name)))
    out = pl.concat(rows).select("dimension", "segment", "model", "n", "mrr", "gndcg10").sort("dimension", "segment", "model")
    out.write_csv(p.results / "diag_segments.csv")
    log.info("segments:\n" + str(out))
    return out


def ablations(cfg, p, log) -> pl.DataFrame:
    frac = cfg["diagnostics"]["ablation_user_frac"]
    names = dataset.feature_names(p)
    rows = []
    for g in ["(none)", *GROUPS]:
        keep = names if g == "(none)" else [n for n in names if group_of(n) != g]
        b = rtrain.fit(cfg, p, keep, user_frac=frac, tag=f"ablation: {g}", log=log)
        pred = rinfer.predict(cfg, p, b, keep, "test")
        rows.append(evaluate(pred, "score", seed=cfg["seed"], name=f"without {g}" if g != "(none)" else "all features"))
    tbl = results_table(rows)
    tbl.write_csv(p.results / "diag_ablations.csv")
    log.info(f"ablations (rankers on {frac:.0%} of training users):\n{tbl}")
    return tbl


def shap(cfg, p, log) -> pl.DataFrame:
    booster = lgb.Booster(model_file=str(p.ranker / "ranker.txt"))
    names = load_json(p.ranker / "ranker_meta.json")["features"]
    df = dataset.load(p, ["test"], cfg)
    n = min(cfg["diagnostics"]["shap_rows"], df.height)
    X = df.sample(n, seed=cfg["seed"]).select(names).to_numpy()
    contrib = booster.predict(X, pred_contrib=True)[:, :-1]          # last column is the expected value
    imp = (pl.DataFrame({"feature": names, "mean_abs_shap": np.abs(contrib).mean(0)})
           .with_columns(group=pl.col("feature").map_elements(group_of, return_dtype=pl.String))
           .sort("mean_abs_shap", descending=True))
    imp.write_csv(p.results / "diag_shap_features.csv")
    grp = imp.group_by("group").agg(pl.col("mean_abs_shap").sum()).sort("mean_abs_shap", descending=True)
    grp.write_csv(p.results / "diag_shap_groups.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    top = imp.head(20).reverse()
    ax[0].barh(top["feature"].to_list(), top["mean_abs_shap"].to_numpy(), color="#4C72B0")
    ax[0].set(title="Top 20 features (mean |SHAP|, test rows)")
    g = grp.reverse()
    ax[1].barh(g["group"].to_list(), g["mean_abs_shap"].to_numpy(), color="#DD8452")
    ax[1].set(title="Feature groups (sum of mean |SHAP|)")
    fig.tight_layout()
    fig.savefig(p.figures / "diag_shap.png", dpi=150)
    plt.close(fig)
    log.info("SHAP by group:\n" + str(grp))
    return grp


@torch.no_grad()
def attention(cfg, p, log) -> dict:
    from .twotower.data import SEGMENTS
    from .twotower.train import ItemFeats, load_assets, load_model, make_stores, to_dev
    device = get_device(cfg)
    a = load_assets(p)
    st, users = make_stores(cfg, p, a)
    model = load_model(p.tower / "full.pt", a, device)
    feats = ItemFeats(a["X"], device, cfg["tower"]["item_matrix_on_gpu"] and device == "cuda")
    ids = users.filter(pl.col("split") == "test")["user_id"].to_numpy()[: cfg["diagnostics"]["attention_users"]]
    res = {s: {"null_mass": [], "nonempty": []} for s in SEGMENTS}
    for i in range(0, len(ids), cfg["tower"]["eval_batch_users"]):
        b = to_dev(st.user_batch(ids[i:i + cfg["tower"]["eval_batch_users"]]), device)
        model.user_vec(b["q"], b["ctx"], feats)
        for s in SEGMENTS:
            att = model.user.pools[s].last_attn.float().cpu().numpy()
            nonempty = b["ctx"][s]["mask"].any(-1).cpu().numpy()
            res[s]["null_mass"].append(att[:, 0])
            res[s]["nonempty"].append(nonempty)
    out = {}
    for s in SEGMENTS:
        nm, ne = np.concatenate(res[s]["null_mass"]), np.concatenate(res[s]["nonempty"])
        out[s] = {"share_users_with_events": float(ne.mean()),
                  "mean_attention_on_events_when_present": float(1 - nm[ne].mean()) if ne.any() else None}
    save_json(out, p.results / "diag_attention.json")
    log.info(f"tower attention by history segment: {out}")
    return out


def run(cfg: dict, parts: list[str]) -> None:
    p = paths(cfg)
    log = get_logger("diagnostics", p)
    if "attention" in parts:          # PyTorch before LightGBM: two OpenMP runtimes in one process can deadlock on macOS
        if sys.platform == "darwin":
            torch.set_num_threads(1)
        attention(cfg, p, log)
    if "segments" in parts:
        segments(cfg, p, log)
    if "shap" in parts:
        shap(cfg, p, log)
    if "ablations" in parts:
        ablations(cfg, p, log)
    for v in ("no_history", "no_query"):
        f = p.results / f"tower_{v}.json"
        if f.exists():
            log.info(f"tower ablation {v}: " + str([(r["model"], round(r["mrr"], 4)) for r in load_json(f)]))
    log.info("\n" + str(rinfer.summary(cfg)))
