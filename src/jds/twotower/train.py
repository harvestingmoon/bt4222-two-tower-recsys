"""Two-tower training with cross-fitting.

Objective: listwise softmax over each training user's displayed slate (target ∝ 2^label - 1, i.e. the graded gain),
which matches the evaluation task (re-rank the slate for the test query); plus an auxiliary in-batch softmax on
historical (query -> item) events inside the click window, weighted by event type, which adds query -> item
supervision from the logs of every user.

Cross-fitting: the ranker is trained on training users, so their tower scores must come from towers that never saw
their slates. Model `fold<k>` trains on training users of the other folds and scores fold k; model `full` trains on
all training users and scores validation and test users. Every model selects its best epoch on validation users.
"""
from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from ..config import get_device, get_logger, load_json, paths, save_json, set_seed
from ..metrics import evaluate
from ..prepare import load
from .data import ID_COLS, Stores, TermVocab, build_id_vocabs, build_item_matrix, build_term_vocab, item_popularity, query_matrix
from .model import TwoTower


# ---------------------------------------------------------------- assets (vocabularies, item matrix, query matrix)
def build_assets(cfg: dict, force: bool = False) -> None:
    p = paths(cfg)
    log = get_logger("tower", p)
    f = p.tower / "assets.npz"
    if f.exists() and not force:
        return
    t0 = time.time()
    tc = cfg["tower"]
    prods = pl.scan_parquet(p.data / "products.parquet")
    queries = load(p.data, "queries")
    history = load(p.data, "history", ["wid", "type", "age_days"])
    cands = load(p.data, "candidates", ["wid"])
    tv = build_term_vocab(prods, queries, tc)
    idv = build_id_vocabs(prods)
    item_wids = np.sort(pl.concat([cands["wid"], history["wid"]]).unique().to_numpy()).astype(np.int64)
    X = build_item_matrix(item_wids, p.data / "products.parquet", tv, idv, item_popularity(history, cfg))
    Q = query_matrix(queries, tv, tc["query_len"])
    np.savez(f, X=X, Q=Q, item_wids=item_wids, terms=tv.terms, oov=tv.oov,
             id_sizes=np.array([idv[c].size for c in ID_COLS]))
    log.info(f"assets: {len(item_wids):,} items, {tv.size:,} term rows, {Q.shape[0]:,} queries ({time.time()-t0:.0f}s)")


def load_assets(p) -> dict:
    a = dict(np.load(p.tower / "assets.npz"))
    a["n_terms"] = TermVocab(a["terms"], int(a["oov"])).size
    return a


def make_stores(cfg: dict, p, a: dict) -> tuple[Stores, pl.DataFrame]:
    users = load(p.data, "users")
    st = Stores(load(p.data, "history", ["user_id", "step", "wid", "qid", "type", "age_days"]), users,
                load(p.data, "candidates", ["user_id", "pos", "wid", "label"]), a["Q"], a["item_wids"], cfg)
    return st, users


# ---------------------------------------------------------------- helpers
def to_dev(b, device):
    if isinstance(b, dict):
        return {k: to_dev(v, device) for k, v in b.items()}
    return torch.as_tensor(b, device=device)


class ItemFeats:
    """Gathers item feature rows by row index from the item matrix (kept on the GPU when it fits)."""

    def __init__(self, X: np.ndarray, device: str, on_device: bool):
        self.device = device
        self.X = torch.as_tensor(X, device=device if on_device else "cpu")

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.X[idx.to(self.X.device)].to(self.device, non_blocking=True)


def autocast(device: str, enabled: bool):
    if enabled and device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def fp32(t: torch.Tensor):
    """Disable autocast so similarity scores keep float32 precision (bfloat16 would create ties inside a slate)."""
    return torch.autocast("cuda", enabled=False) if t.is_cuda else nullcontext()


def slate_scores(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    with fp32(u):
        return torch.einsum("bd,bsd->bs", u.float(), v.float())


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Softmax cross-entropy between the slate scores and the graded-gain distribution (padding excluded)."""
    valid = labels >= 0
    gain = torch.where(valid, 2.0 ** labels.float().clamp(min=0) - 1.0, torch.zeros_like(scores))
    has_pos = gain.sum(-1) > 0
    target = gain / gain.sum(-1, keepdim=True).clamp(min=1e-9)
    logp = torch.log_softmax(scores.masked_fill(~valid, -1e4), -1)
    loss = -(target * logp).sum(-1)
    return loss[has_pos].mean() if has_pos.any() else scores.sum() * 0.0


def inbatch_loss(u: torch.Tensor, v: torch.Tensor, item_idx: torch.Tensor, w: torch.Tensor, scale) -> torch.Tensor:
    """In-batch softmax; other rows holding the same item are not treated as negatives."""
    with fp32(u):
        logits = scale * (u.float() @ v.float().T)
    same = (item_idx[:, None] == item_idx[None, :]) & ~torch.eye(len(item_idx), dtype=torch.bool, device=u.device)
    logits = logits.masked_fill(same, -1e4)
    ce = F.cross_entropy(logits, torch.arange(len(u), device=u.device), reduction="none")
    return (ce * w).sum() / w.sum().clamp(min=1e-9)


@torch.no_grad()
def score_users(model: TwoTower, st: Stores, feats: ItemFeats, users: np.ndarray, cfg: dict, device: str) -> pl.DataFrame:
    """Cosine score of every displayed item for each user's test query (rows: user_id, wid, label, tower_score)."""
    model.eval()
    bs = int(cfg["tower"]["eval_batch_users"])
    out_u, out_w, out_l, out_s = [], [], [], []
    for s in range(0, len(users), bs):
        ub = users[s:s + bs]
        b, items, labels = st.slate_batch(ub)
        with autocast(device, cfg["tower"]["amp"]):
            bd = to_dev(b, device)
            u = model.user_vec(bd["q"], bd["ctx"], feats)
            it = torch.as_tensor(items, device=device)
            v = model.item_vec(feats(it))
            sc = slate_scores(u, v).cpu().numpy()
        m = labels >= 0
        out_u.append(np.repeat(ub, m.sum(1))); out_w.append(st.s_wid[ub][m]); out_l.append(labels[m]); out_s.append(sc[m])
    return pl.DataFrame({"user_id": np.concatenate(out_u).astype(np.uint32), "wid": np.concatenate(out_w).astype(np.int32),
                         "label": np.concatenate(out_l).astype(np.int8), "tower_score": np.concatenate(out_s).astype(np.float32)})


# ---------------------------------------------------------------- training
def train_one(name: str, train_users: np.ndarray, val_users: np.ndarray, st: Stores, a: dict, cfg: dict,
              device: str, log, use_history: bool = True, use_query: bool = True) -> dict:
    tc = cfg["tower"]
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = TwoTower(a["n_terms"], [int(x) for x in a["id_sizes"]], tc, use_history, use_query).to(device)
    feats = ItemFeats(a["X"], device, tc["item_matrix_on_gpu"] and device == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    use_scaler = tc["amp"] and device == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    aux_ev = st.aux_events(tc["aux"]["max_age_days"]) if tc["aux"]["enabled"] else np.array([], np.int64)
    bs = int(tc["slate_batch"])
    steps_per_epoch = max(1, len(train_users) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=steps_per_epoch * tc["epochs"],
                                                pct_start=0.05, anneal_strategy="cos")
    log.info(f"[{name}] {len(train_users):,} training slates, {len(aux_ev):,} auxiliary anchors, "
             f"{sum(x.numel() for x in model.parameters()):,} parameters, device={device}")
    best, best_state, history = -1.0, None, []
    step = 0
    for ep in range(tc["epochs"]):
        model.train()
        order = rng.permutation(train_users)
        t0, run = time.time(), []
        for i in range(steps_per_epoch):
            ub = order[i * bs:(i + 1) * bs]
            b, items, labels = st.slate_batch(ub)
            with autocast(device, tc["amp"]):
                bd = to_dev(b, device)
                u = model.user_vec(bd["q"], bd["ctx"], feats)
                v = model.item_vec(feats(torch.as_tensor(items, device=device)))
                scores = model.scale() * slate_scores(u, v)
                loss = listwise_loss(scores.float(), torch.as_tensor(labels, device=device))
                aux = torch.zeros((), device=device)
                if len(aux_ev) and step % tc["aux"]["every"] == 0:
                    ev = aux_ev[rng.integers(0, len(aux_ev), size=min(int(tc["aux"]["batch"]), len(aux_ev)))]
                    pb, pit, pw = st.pair_batch(ev)
                    pbd = to_dev(pb, device)
                    pit_t = torch.as_tensor(pit, device=device)
                    aux = inbatch_loss(model.user_vec(pbd["q"], pbd["ctx"], feats), model.item_vec(feats(pit_t)),
                                       pit_t, torch.as_tensor(pw, device=device), model.scale())
                    loss = loss + tc["aux"]["weight"] * aux
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                opt.step()
            sched.step()
            step += 1
            run.append((loss.item(), aux.item()))
            if step % tc["log_every"] == 0:
                r = np.mean(run[-tc["log_every"]:], 0)
                log.info(f"[{name}] ep {ep} step {i+1}/{steps_per_epoch} loss {r[0]:.4f} aux {r[1]:.4f} "
                         f"scale {float(model.scale()):.1f} ({time.time()-t0:.0f}s)")
        res = evaluate(score_users(model, st, feats, val_users, cfg, device), "tower_score", seed=cfg["seed"])
        history.append({"epoch": ep, "train_loss": float(np.mean([r[0] for r in run])), **res})
        log.info(f"[{name}] epoch {ep}: val MRR {res['mrr']:.4f} NDCG@10 {res['ndcg10']:.4f} gNDCG@10 {res['gndcg10']:.4f} "
                 f"({time.time()-t0:.0f}s)")
        if res["gndcg10"] > best:
            best = res["gndcg10"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {"state": best_state, "history": history, "use_history": use_history, "use_query": use_query}


def model_plan(cfg: dict, users: pl.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(model name, training users, users it scores) for cross-fitting."""
    tr = users.filter((pl.col("split") == "train") & pl.col("has_pos"))
    plan = []
    for f in range(cfg["prepare"]["n_folds"]):
        plan.append((f"fold{f}", tr.filter(pl.col("fold") != f)["user_id"].to_numpy(),
                     users.filter((pl.col("split") == "train") & (pl.col("fold") == f))["user_id"].to_numpy()))
    plan.append(("full", tr["user_id"].to_numpy(), users.filter(pl.col("split").is_in(["val", "test"]))["user_id"].to_numpy()))
    return plan


def train(cfg: dict, models: list[str] | None = None, variant: str = "", force: bool = False) -> None:
    """Train the cross-fitted towers (or one ablation `variant` of the full model: no_history / no_query)."""
    p = paths(cfg)
    log = get_logger("tower", p)
    device = get_device(cfg)
    build_assets(cfg, force=force)
    a = load_assets(p)
    st, users = make_stores(cfg, p, a)
    val_users = users.filter((pl.col("split") == "val") & pl.col("has_pos"))["user_id"].to_numpy()
    plan = model_plan(cfg, users)
    if variant:
        plan = [x for x in plan if x[0] == "full"]
    for name, tr_users, _ in plan:
        if models and name not in models:
            continue
        tag = f"{name}_{variant}" if variant else name
        out = p.tower / f"{tag}.pt"
        if out.exists() and not force:
            log.info(f"[{tag}] exists, skipping (use --force to retrain)")
            continue
        r = train_one(tag, tr_users, val_users, st, a, cfg, device, log,
                      use_history=variant != "no_history", use_query=variant != "no_query")
        torch.save({"state": r["state"], "use_history": r["use_history"], "use_query": r["use_query"],
                    "tower_cfg": cfg["tower"]}, out)
        save_json(r["history"], p.tower / f"{tag}_history.json")


def load_model(path, a: dict, device: str) -> TwoTower:
    ck = torch.load(path, map_location="cpu")
    m = TwoTower(a["n_terms"], [int(x) for x in a["id_sizes"]], ck["tower_cfg"], ck["use_history"], ck["use_query"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


def history_of(p, tag: str):
    f = p.tower / f"{tag}_history.json"
    return load_json(f) if f.exists() else None
