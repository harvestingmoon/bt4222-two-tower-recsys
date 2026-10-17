"""Stage 6: re-rank validation and test slates with the trained ranker -> outputs/<scope>/predictions/ranker_{val,test}.parquet,
metrics -> outputs/<scope>/results/ranker.json, and the all-model test table -> results/summary_test.csv

    python scripts/06_predict.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

from jds.config import cli  # noqa: E402
from jds.ranker import infer  # noqa: E402

if __name__ == "__main__":
    cfg, args = cli(__doc__)
    infer.run(cfg)
    pl.Config.set_tbl_cols(12); pl.Config.set_tbl_width_chars(160); pl.Config.set_fmt_str_lengths(60)
    print("\nTest-set comparison (outputs/<scope>/results/summary_test.csv):")
    print(infer.summary(cfg))
