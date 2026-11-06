"""Stage 3: train the cross-fitted two-tower models (fold0..fold{k-1} + full) -> outputs/<scope>/tower/*.pt

    python scripts/03_train_tower.py                         # all models
    python scripts/03_train_tower.py --models full           # only some
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jds.config import cli  # noqa: E402
from jds.twotower import train  # noqa: E402


def extra(p):
    p.add_argument("--models", default="", help="comma-separated subset of fold0,fold1,...,full")


if __name__ == "__main__":
    cfg, args = cli(__doc__, extra)
    train.train(cfg, models=[m for m in args.models.split(",") if m] or None, force=args.force)
