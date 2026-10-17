import numpy as np
import polars as pl

from jds.config import paths
from jds.prepare import load


def test_history_ages(tiny_cfg):
    h = load(paths(tiny_cfg).data, "history")
    u1 = h.filter(pl.col("user_id") == 1).sort("step")
    # README: time[i] = gap to the previous behaviour, last entry = gap to the test query
    np.testing.assert_allclose(u1["age_days"].to_numpy(), [200, 30, 2, 1 / 24], rtol=1e-5)
    assert u1["type"].to_list() == [4, 3, 1, 1]
    assert u1["qid"].null_count() == 2          # "-1" and the empty query string are query-less
    ages = h.sort("user_id", "step").with_columns(d=pl.col("age_days").diff().over("user_id"))
    assert (ages["d"].drop_nulls() <= 1e-6).all()


def test_candidates_shuffled_and_complete(tiny_cfg):
    p = paths(tiny_cfg)
    c = load(p.data, "candidates")
    assert c.group_by("user_id").len()["len"].unique().to_list() == [8]
    # raw order has positives first; after the shuffle some positive must sit behind a negative
    first_pos = c.filter(pl.col("label") > 0).group_by("user_id").agg(pl.col("pos").max())
    assert (first_pos["pos"] > 0).any()
    assert c.group_by("user_id").agg(pl.col("pos").sort()).with_columns(ok=pl.col("pos") == list(range(8)))["ok"].all()


def test_split_and_folds(tiny_cfg):
    u = load(paths(tiny_cfg).data, "users")
    assert u.filter(pl.col("user_id") == 0)["split"].item() == "train"     # no positive -> train only
    assert (u.filter(pl.col("split") != "train")["fold"] == -1).all()
    assert set(u.filter(pl.col("split") == "train")["fold"].unique()) == set(range(tiny_cfg["prepare"]["n_folds"]))
    assert u.filter(pl.col("split") != "train")["has_pos"].all()


def test_products_restricted_and_queries(tiny_cfg):
    p = paths(tiny_cfg)
    pr = load(p.data, "products")
    assert 999 not in pr["wid"].to_list()
    q = load(p.data, "queries")
    assert q["terms"].list.len().min() >= 1     # empty tokens removed ("a^X^Xb" -> [a, b])
