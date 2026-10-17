"""Stage 02b: non-learned baselines on validation/test slates -> results/baselines.json

    python scripts/02b_baselines.py [--config configs/default.yaml]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jds import baselines  # noqa: E402
from jds.config import cli  # noqa: E402

if __name__ == "__main__":
    cfg, args = cli(__doc__)
    baselines.run(cfg)
