"""Stage 0: parse the raw files once into parquet (outputs/full/data), then cut the configured subsample if any.

    python scripts/00_prepare.py                              # full data
    python scripts/00_prepare.py --config configs/smoke.yaml  # full data (if missing) + sample_2000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jds.config import cli  # noqa: E402
from jds.prepare import make_subset, prepare_full  # noqa: E402

if __name__ == "__main__":
    cfg, args = cli(__doc__)
    prepare_full(cfg, force=args.force and cfg["scope"] == "full")
    make_subset(cfg, force=args.force)
