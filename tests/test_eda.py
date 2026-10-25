from jds import eda
from jds.config import paths


def test_eda_funnel_and_slates(tiny_cfg):
    out = eda.run(tiny_cfg)
    p = paths(tiny_cfg)
    assert (p.figures / "funnel_by_age.png").exists() and (p.results / "eda_funnel_by_age.csv").exists()
    # tiny data: per user 2 clicks < 7 d, 1 cart at 30 d, 1 order at 200 d -> every cart and order predates the click window
    m = out["missing_clicks"]
    assert m["carts_older_than_click_window"] == 1.0 and m["orders_older_than_click_window"] == 1.0
    assert m["clicks_older_than_click_window"] == 0.0
    s = out["slates"]
    assert sum(s["positives_per_slate"].values()) == 40 and s["positives_per_slate"]["0"] == 1
    assert {r["label"] for r in s["by_label"]} == {0, 1, 2, 3}
