"""Stage 4: cross-fitted tower scores for every displayed item -> outputs/<scope>/tower/tower_scores.parquet
and tower-only metrics -> outputs/<scope>/results/tower.json

    python scripts/04_score_tower.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jds.config import cli  # noqa: E402
from jds.twotower import infer  # noqa: E402


if __name__ == "__main__":
    cfg, args = cli(__doc__)
    infer.score(cfg, force=args.force)
