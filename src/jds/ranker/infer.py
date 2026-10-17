"""Stage 6: final predictions of the re-ranker for validation and test users + the results table."""
from __future__ import annotations

import lightgbm as lgb
import polars as pl

from ..config import get_logger, load_json, paths, save_json
from ..metrics import evaluate, per_user_metrics, results_table
from . import dataset


def predict(cfg: dict, p, booster: lgb.Booster, names: list[str], split: str, use_tower: bool = True) -> pl.DataFrame:
    df = dataset.load(p, [split], cfg, use_tower)
    return df.select("user_id", "wid", "label").with_columns(
        score=pl.Series(booster.predict(df.select(names).to_numpy(), num_threads=0)).cast(pl.Float32))


def run(cfg: dict) -> pl.DataFrame:
    p = paths(cfg)
    log = get_logger("ranker", p)
    f = p.ranker / "ranker.txt"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/05_train_ranker.py first")
    booster = lgb.Booster(model_file=str(f))
    names = load_json(p.ranker / "ranker_meta.json")["features"]
    rows = []
    for split in ("val", "test"):
        pred = predict(cfg, p, booster, names, split)
        pred.write_parquet(p.preds / f"ranker_{split}.parquet")
        rows.append(evaluate(pred, "score", seed=cfg["seed"], name=f"two-tower + LambdaRank [{split}]"))
        if split == "test":
            per_user_metrics(pred, "score", seed=cfg["seed"]).write_parquet(p.results / "per_user_ranker_test.parquet")
    save_json(rows, p.results / "ranker.json")
    log.info("\n" + str(results_table(rows)))
    return results_table(rows)


def summary(cfg: dict) -> pl.DataFrame:
    """Every model's test metrics in one table (baselines, prior team's model, tower, ranker)."""
    p = paths(cfg)
    rows = []
    for name in ("baselines.json", "prior_ncf.json", "tower.json", "ranker.json"):
        f = p.results / name
        if f.exists():
            rows += [r for r in load_json(f) if "[test]" in r["model"]]
    if not rows:
        return pl.DataFrame()
    tbl = results_table(rows)
    tbl.write_csv(p.results / "summary_test.csv")
    return tbl
