"""Ranker input: hand-crafted features + cross-fitted two-tower scores, one row per displayed (user, item)."""
from __future__ import annotations

import numpy as np
import polars as pl

from .. import features
from ..config import load_json

TOWER_FEATURES = ["tower_score", "tower_score_pct", "tower_score_gap"]


def tower_features(scores: pl.DataFrame) -> pl.DataFrame:
    """Tower score, its within-slate percentile and its distance to the slate's best score."""
    n = pl.len().over("user_id")
    return scores.select(
        "user_id", "wid", "tower_score",
        tower_score_pct=((pl.col("tower_score").rank("average").over("user_id") - 1) / (n - 1).clip(1)).cast(pl.Float32),
        tower_score_gap=(pl.col("tower_score").max().over("user_id") - pl.col("tower_score")).cast(pl.Float32))


def feature_names(p, use_tower: bool = True, drop: tuple[str, ...] = ()) -> list[str]:
    names = load_json(p.features / "feature_list.json") + (TOWER_FEATURES if use_tower else [])
    return [n for n in names if not any(n == d or n.startswith(d) for d in drop)]


def load(p, splits, cfg: dict, use_tower: bool = True, user_frac: float = 1.0, tower_file: str = "tower_scores.parquet",
         only_with_pos: bool = False) -> pl.DataFrame:
    """Feature rows of the given splits, sorted by user (LightGBM groups must be contiguous)."""
    df = features.read(p, splits=splits, user_frac=user_frac, seed=cfg["seed"])
    if only_with_pos:
        df = df.filter((pl.col("label") > 0).any().over("user_id"))
    if use_tower:
        f = p.tower / tower_file
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: run scripts/04_score_tower.py first")
        ts = tower_features(pl.read_parquet(f, columns=["user_id", "wid", "tower_score"]))
        df = df.join(ts, on=["user_id", "wid"], how="left")
        miss = df["tower_score"].null_count()
        if miss:
            raise ValueError(f"{miss} rows without a tower score (tower scores and features come from different scopes?)")
    return df.sort("user_id", "wid")


def groups(df: pl.DataFrame) -> np.ndarray:
    """Group sizes in row order (rows sorted by user_id)."""
    return df.group_by("user_id", maintain_order=True).len()["len"].to_numpy()
