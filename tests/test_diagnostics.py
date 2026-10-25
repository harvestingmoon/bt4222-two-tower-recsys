import polars as pl

from jds import diagnostics, features
from jds.config import get_logger, paths


def test_segment_table_and_insights(tiny_cfg):
    features.build(tiny_cfg)
    p = paths(tiny_cfg)
    seg = diagnostics.segment_table(tiny_cfg, p)
    assert set(diagnostics.SEG_DIMS) <= set(seg.columns)
    assert seg.select(diagnostics.SEG_DIMS).null_count().sum_horizontal().item() == 0
    ins = diagnostics.insights(tiny_cfg, p, get_logger("test"))
    assert ins.height == len(diagnostics.INSIGHT_SIGNALS)
    assert ins.filter(pl.col("share_of_displayed") < 0).is_empty()
