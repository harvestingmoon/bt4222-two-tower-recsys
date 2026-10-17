"""Stage 2: build the ranking feature table (outputs/<scope>/features/part_*.parquet).

    python scripts/02_features.py [--config configs/default.yaml] [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jds import features  # noqa: E402
from jds.config import cli  # noqa: E402

if __name__ == "__main__":
    cfg, args = cli(__doc__)
    names = features.build(cfg, force=args.force)
    print(f"{len(names)} features")
