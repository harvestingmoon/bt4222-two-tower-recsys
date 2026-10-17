"""Stage 02c: previous team's Hybrid NCF+LSTM (reference baseline) -> results/prior_ncf.json

    python scripts/02c_prior_ncf.py [--config configs/default.yaml]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jds import prior_ncf  # noqa: E402
from jds.config import cli  # noqa: E402

if __name__ == "__main__":
    cfg, args = cli(__doc__)
    prior_ncf.run(cfg)
