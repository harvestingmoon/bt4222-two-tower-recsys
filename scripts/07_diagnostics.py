"""Stage 7: diagnostics of the final model -> results/diag_* and figures/diag_*

    python scripts/07_diagnostics.py                                  # segments, insights, shap, attention
    python scripts/07_diagnostics.py --parts segments,insights,shap,attention,ablations
Tower ablations are trained separately (optional):
    python scripts/03_train_tower.py --variant no_history && python scripts/04_score_tower.py --variant no_history
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jds import diagnostics  # noqa: E402
from jds.config import cli  # noqa: E402


def extra(p):
    p.add_argument("--parts", default="segments,insights,shap,attention", help="comma-separated: segments,insights,shap,attention,ablations")


if __name__ == "__main__":
    cfg, args = cli(__doc__, extra)
    diagnostics.run(cfg, args.parts.split(","))
