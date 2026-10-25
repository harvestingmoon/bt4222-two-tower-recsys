"""Helpers for the stage notebooks (notebooks/step*.ipynb): run a pipeline script with live output, find its
results and show them with the performance numbers highlighted.

Every stage runs as a subprocess (same command as the terminal), so a crash never leaves a half-initialised kernel,
and every stage skips work whose outputs already exist (pass "--force" to redo one).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .config import REPO, load_config, paths


class Runner:
    def __init__(self, config: str = "configs/default.yaml", drive_dir: str | None = None, extra: list[str] | None = None):
        """`drive_dir` (Colab): folder on Google Drive holding raw/ (the two JDsearch files); outputs/ is written next
        to it so results survive the Colab session and are shared by all notebooks."""
        self.config = config
        self.extra = list(extra or [])
        if drive_dir:
            self.extra += ["--set", f"paths.raw_dir={drive_dir}/raw", "--set", f"paths.work_dir={drive_dir}/outputs"]
        overrides = [self.extra[i + 1] for i in range(len(self.extra) - 1) if self.extra[i] == "--set"]
        self.cfg = load_config(REPO / config, overrides)
        self.p = paths(self.cfg)
        print(f"config {config} | scope {self.p.scope} | raw data {self.p.raw} | outputs {self.p.root}")

    def stage(self, script: str, *args: str) -> None:
        """Run scripts/<script> and stream its log into the cell output."""
        cmd = [sys.executable, "-u", f"scripts/{script}", "--config", self.config, *self.extra, *args]
        print("$ " + " ".join(cmd), flush=True)
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            print(line, end="", flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f"{script} failed (exit code {proc.returncode}); see the log above")

    # ------------------------------------------------------------ results
    def results(self, name: str) -> Path:
        return self.p.results / name

    def json(self, name: str):
        f = self.results(name)
        if not f.exists() and self.p.scope != "full":      # e.g. prepare_stats.json is written for the full data only
            f = self.p.work / "full" / "results" / name
            print(f"(from the full data: {f})")
        return json.loads(f.read_text())

    def figure(self, name: str):
        from IPython.display import Image, display
        f = self.p.figures / name
        if f.exists():
            display(Image(str(f)))
        else:
            print(f"(figure {name} not produced yet)")

    def table(self, name: str, highlight=("mrr", "p1", "ndcg10", "gndcg10", "hr10"), caption: str | None = None, lower_is_better=()):
        """Show a results CSV; the best value of each metric column is highlighted (green)."""
        import pandas as pd
        f = self.results(name)
        if not f.exists():
            print(f"({name} not produced yet)")
            return None
        df = pd.read_csv(f)
        cols = [c for c in highlight if c in df.columns]
        sty = df.style.format(precision=4)
        if cols:
            sty = sty.highlight_max(subset=[c for c in cols if c not in lower_is_better], color="#c6efce")
            if lower_is_better:
                sty = sty.highlight_min(subset=[c for c in cols if c in lower_is_better], color="#c6efce")
        if caption:
            sty = sty.set_caption(caption)
        return sty


def banner(text: str = "PERFORMANCE OUTPUT"):
    """A visible marker above every cell whose output is a reported metric (deliverable guideline)."""
    from IPython.display import HTML, display
    display(HTML(f'<div style="background:#fff2cc;border-left:6px solid #e69138;padding:6px 10px;font-weight:bold">'
                 f'&#9733; {text}</div>'))
