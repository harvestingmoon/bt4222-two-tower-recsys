"""Stage 1: exploratory analysis that motivates the design -> <scope>/figures/*.png and results/eda.json.

Includes the history-timeline analysis requested in the interim feedback: event types by days before the test query
(retention windows), and how the share of recent events drives naive user ratios.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from .config import TYPE_NAME, get_logger, paths, save_json  # noqa: E402
from .prepare import load  # noqa: E402
from .windows import CLICK, mix_by_age, window_report  # noqa: E402

COLORS = {1: "#4C72B0", 2: "#8172B2", 3: "#DD8452", 4: "#55A868"}


def fig_timeline(history: pl.DataFrame, path) -> None:
    """Events by days before the test query, per type (log-spaced bins): the three retention windows."""
    bins = np.concatenate([[0], np.logspace(-3, np.log10(700), 60)])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for t in (1, 2, 3, 4):
        a = history.filter(pl.col("type") == t)["age_days"].to_numpy()
        ax[0].hist(a, bins=bins, histtype="step", lw=1.6, color=COLORS[t], label=f"{TYPE_NAME[t]} (n={len(a):,})")
    for d in (7, 90, 365):
        ax[0].axvline(d, color="grey", ls="--", lw=0.8)
        ax[0].text(d, ax[0].get_ylim()[1] * 0.9, f" {d} d", color="grey")
    ax[0].set(xscale="symlog", xlabel="days before the test query", ylabel="events", title="History events by age and type")
    ax[0].legend(fontsize=8)
    mix = mix_by_age(history)
    labels = mix["bucket"].cast(pl.String).to_list()
    bottom = np.zeros(mix.height)
    for t in (1, 2, 3, 4):
        v = mix[TYPE_NAME[t].lower()].to_numpy()
        ax[1].bar(labels, v, bottom=bottom, color=COLORS[t], label=TYPE_NAME[t])
        bottom += v
    ax[1].set(xlabel="days before the test query", ylabel="share of events", title="Event-type mix by age bucket")
    ax[1].tick_params(axis="x", rotation=45)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def recency_confound(history: pl.DataFrame, users: pl.DataFrame) -> dict:
    """How much of a naive 'click ratio' is just the share of the user's events inside the last 7 days."""
    u = (history.group_by("user_id").agg(click_ratio=(pl.col("type") == CLICK).mean(),
                                         recent_share=(pl.col("age_days") < 7).mean(),
                                         any_click=(pl.col("type") == CLICK).any()))
    corr = float(np.corrcoef(u["click_ratio"].to_numpy(), u["recent_share"].to_numpy())[0, 1])
    return {"corr_click_ratio_vs_recent_share": corr, "users_without_clicks": float(1 - u["any_click"].mean()),
            "n_users": users.height}


def signal_aucs(cands: pl.DataFrame, history: pl.DataFrame, users: pl.DataFrame, n: int, seed: int) -> dict:
    """Pooled AUC (positive vs not) of item popularity computed per type window, on a user sample."""
    ids = users.filter(pl.col("has_pos"))["user_id"].sample(min(n, users.height), seed=seed)
    c = cands.join(ids.to_frame(), on="user_id", how="semi")
    pops = history.group_by("wid").agg(all_events=pl.len(),
                                       click_7=((pl.col("type") == 1) & (pl.col("age_days") < 7)).sum(),
                                       cart_90=((pl.col("type") == 3) & (pl.col("age_days") < 90)).sum(),
                                       order_365=(pl.col("type") == 4).sum())
    c = c.join(pops, on="wid", how="left").fill_null(0)
    y = (c["label"] > 0).to_numpy()
    return {k: float(roc_auc_score(y, c[k].to_numpy())) for k in ("all_events", "click_7", "cart_90", "order_365")}


def run(cfg: dict) -> dict:
    p = paths(cfg)
    log = get_logger("eda", p)
    users = load(p.data, "users")
    cands = load(p.data, "candidates", ["user_id", "wid", "label"])
    history = load(p.data, "history", ["user_id", "wid", "qid", "type", "age_days"])
    out = {
        "labels": dict(cands.group_by("label").len().sort("label").iter_rows()),
        "slate_len": cands.group_by("user_id").len()["len"].describe().to_dicts(),
        "history_len": users["hist_len"].describe().to_dicts(),
        "retention": window_report(history).to_dicts(),
        "type_mix_by_age": mix_by_age(history).with_columns(pl.col("bucket").cast(pl.String)).to_dicts(),
        "recency_confound": recency_confound(history, users),
        "popularity_auc": signal_aucs(cands, history, users, cfg["eda"]["sample_users"], cfg["seed"]),
        "displayed_items_never_in_history": float(1 - cands.select("wid").unique().join(history.select("wid").unique(), on="wid",
                                                                                         how="semi").height
                                                  / cands["wid"].n_unique()),
    }
    fig_timeline(history, p.figures / "history_timeline.png")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].hist(cands.group_by("user_id").len()["len"].to_numpy(), bins=50, color="#4C72B0")
    ax[0].set(title="Displayed items per test query", xlabel="slate length")
    ax[1].hist(np.log10(1 + users["hist_len"].to_numpy()), bins=50, color="#DD8452")
    ax[1].set(title="History length", xlabel="log10(1 + events)")
    fig.tight_layout()
    fig.savefig(p.figures / "slates_and_history.png", dpi=150)
    plt.close(fig)
    save_json(out, p.results / "eda.json")
    log.info(f"retention windows:\n{window_report(history)}")
    log.info(f"recency confound: {out['recency_confound']}; popularity AUC by window: {out['popularity_auc']}")
    return out
