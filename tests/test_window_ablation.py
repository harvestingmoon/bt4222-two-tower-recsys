import polars as pl

from jds import features, window_ablation
from jds.config import load_json, paths


def test_windowed_columns_are_history_counts_only(tiny_cfg):
    features.build(tiny_cfg)
    names = load_json(paths(tiny_cfg).features / "feature_list.json")
    win = window_ablation.windowed_history_columns(names, tiny_cfg)
    assert "u_wid_click_7" in win and "ipop_order_365" in win and "u_cart_not_ordered" in win and "ipop_click_7_pct" in win
    assert not any(n.startswith(("ctr_", "lex_", "slate_", "item_", "u_q_", "u_simq_")) for n in win)
    assert "ua_hist_len" not in win and "lex_bm25_pct" not in win


def test_naive_features_ignore_windows(tiny_cfg):
    features.build(tiny_cfg)
    names = window_ablation.build(tiny_cfg, force=True)
    p = paths(tiny_cfg)
    nv = pl.read_parquet(p.root / "features_naive" / "part_*.parquet")
    main = features.read(p, columns=["user_id", "wid", "u_wid_order_90", "u_wid_order_365"])
    assert nv.height == main.height and nv.select(names).null_count().sum_horizontal().item() == 0
    # every tiny user has exactly one order (200 days old) and two clicks: all-history counts see all of them
    r = nv.filter(pl.col("user_id") == 1).row(0, named=True)
    assert r["nv_ua_order_all"] == 1 and r["nv_ua_click_all"] == 2 and abs(r["nv_ua_click_ratio"] - 0.5) < 1e-6
    v = window_ablation.variants(load_json(p.features / "feature_list.json"), names, tiny_cfg)
    assert not set(v["naive (all-history counts, decay, ratios)"]) & set(window_ablation.windowed_history_columns(v["windowed (pipeline)"], tiny_cfg))
