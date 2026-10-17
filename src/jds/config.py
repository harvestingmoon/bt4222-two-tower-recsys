"""Configuration, paths, seeding, logging and device selection shared by every stage."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
TYPE_CODE = {"CLICK": 1, "FLW": 2, "CART": 3, "ORD": 4}
TYPE_NAME = {v: k for k, v in TYPE_CODE.items()}
TERM_SEP = "\x18"  # ^X separates anonymised term ids inside text fields


# ---------------------------------------------------------------- config
def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base = _read((path.parent / cfg.pop("base")).resolve())
        cfg = _deep_merge(base, cfg)
    return cfg


def _set(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config (following `base:` chains) and apply `key.sub=value` overrides."""
    cfg = _read(Path(path).resolve())
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        _set(cfg, key.strip(), yaml.safe_load(raw))
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def cli(description: str, extra=None) -> tuple[dict, argparse.Namespace]:
    """Standard command line for every script: --config, --set k=v (repeatable), --force."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(REPO / "configs" / "default.yaml"))
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set tower.epochs=2 (repeatable)")
    p.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    if extra:
        extra(p)
    args = p.parse_args()
    return load_config(args.config, args.overrides), args


# ---------------------------------------------------------------- paths
def _abs(p: str) -> Path:
    p = Path(os.path.expanduser(p))
    return p if p.is_absolute() else (REPO / p).resolve()


def paths(cfg: dict, scope: str | None = None) -> SimpleNamespace:
    """All locations used by the pipeline for one scope (`full` or `sample_<N>`)."""
    scope = scope or cfg["scope"]
    work = _abs(cfg["paths"]["work_dir"])
    root = work / scope
    p = SimpleNamespace(
        raw=_abs(cfg["paths"]["raw_dir"]), work=work, scope=scope, root=root,
        data=root / "data", features=root / "features", tower=root / "tower", ranker=root / "ranker",
        preds=root / "predictions", results=root / "results", figures=root / "figures", logs=root / "logs",
        full_data=work / "full" / "data",
    )
    for d in (p.data, p.features, p.tower, p.ranker, p.preds, p.results, p.figures, p.logs):
        d.mkdir(parents=True, exist_ok=True)
    return p


def sample_size(scope: str) -> int | None:
    if scope == "full":
        return None
    if scope.startswith("sample_") and scope[7:].isdigit():
        return int(scope[7:])
    raise ValueError(f"scope must be 'full' or 'sample_<N>', got {scope!r}")


# ---------------------------------------------------------------- runtime
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(cfg: dict) -> str:
    import torch
    want = cfg.get("device", "auto")
    if want == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return want


def get_logger(name: str, p: SimpleNamespace | None = None) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(name)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if p is not None:
        fh = logging.FileHandler(p.logs / f"{name}.log")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    log.propagate = False
    return log


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)
