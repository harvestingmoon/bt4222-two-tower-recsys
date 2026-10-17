"""Stage 1: exploratory analysis (history retention windows, type mix by age, slates) -> figures/ and results/eda.json

    python scripts/01_eda.py [--config configs/default.yaml]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jds import eda  # noqa: E402
from jds.config import cli  # noqa: E402

if __name__ == "__main__":
    cfg, args = cli(__doc__)
    eda.run(cfg)
