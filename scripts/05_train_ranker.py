"""Stage 5: train the LightGBM LambdaRank re-ranker -> outputs/<scope>/ranker/ranker.txt

    python scripts/05_train_ranker.py [--set ranker.train_user_frac=0.5]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jds.config import cli  # noqa: E402
from jds.ranker import train  # noqa: E402

if __name__ == "__main__":
    cfg, args = cli(__doc__)
    train.train(cfg, force=args.force)
