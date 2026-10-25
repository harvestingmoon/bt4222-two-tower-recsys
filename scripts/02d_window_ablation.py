"""Stage 2d: windowed vs naive history features, same ranker (no tower score, runs on CPU after stage 2)
-> results/window_ablation.csv, results/window_ablation_segments.csv, figures/window_ablation.png

    python scripts/02d_window_ablation.py [--set diagnostics.ablation_user_frac=0.3]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jds import window_ablation  # noqa: E402
from jds.config import cli  # noqa: E402

if __name__ == "__main__":
    cfg, args = cli(__doc__)
    window_ablation.run(cfg, force=args.force)
