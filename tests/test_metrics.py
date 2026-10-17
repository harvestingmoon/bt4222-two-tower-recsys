import numpy as np
import polars as pl
import pytest

from jds.metrics import evaluate


def _df(labels, scores):
    return pl.DataFrame({"user_id": [0] * len(labels), "label": labels, "score": scores})


def test_perfect_ranking():
    r = evaluate(_df([0, 3, 0, 1], [0.1, 0.9, 0.2, 0.8]))
    assert r["mrr"] == 1 and r["p1"] == 1 and r["ndcg10"] == pytest.approx(1) and r["gndcg10"] == pytest.approx(1)


def test_known_values():
    r = evaluate(_df([1, 0, 0], [0.1, 0.9, 0.5]))       # the only positive is ranked 3rd
    assert r["mrr"] == pytest.approx(1 / 3)
    assert r["ndcg10"] == pytest.approx(1 / np.log2(4))
    assert r["p1"] == 0 and r["hr10"] == 1


def test_graded_gain_prefers_purchase():
    a = evaluate(_df([3, 1, 0], [0.9, 0.5, 0.1]))["gndcg10"]
    b = evaluate(_df([3, 1, 0], [0.5, 0.9, 0.1]))["gndcg10"]
    assert a == pytest.approx(1) and b < a


def test_users_without_positives_are_ignored():
    df = pl.concat([_df([1, 0], [1.0, 0.0]), _df([0, 0], [1.0, 0.0]).with_columns(user_id=pl.lit(1).cast(pl.Int64))])
    assert evaluate(df)["n_users"] == 1


def test_ties_are_not_broken_by_position():
    # all scores equal: the positive must not always win because it comes first
    df = pl.DataFrame({"user_id": np.repeat(np.arange(300), 10), "label": np.tile([1] + [0] * 9, 300),
                       "score": np.zeros(3000)})
    assert evaluate(df)["p1"] < 0.25
