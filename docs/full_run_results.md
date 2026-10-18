# Full-run results — 2026-10-18

Reproduction of the complete pipeline (`configs/default.yaml`) on the full JDsearch dataset.

## Environment

| item | value |
|---|---|
| OS / hardware | Linux, 196 GB RAM, NVIDIA H100 80 GB |
| Python | 3.12 (`python3 -m venv .venv`) |
| PyTorch | 2.6.0+cu124 |
| LightGBM | 4.7.0 |
| Polars | 1.44.2 |
| Dataset | JDsearch raw files via <https://huggingface.co/datasets/lswhim/JDsearch> (official JD Cloud link needs QR login) |

Commands:

```bash
PYTHON=.venv/bin/python bash scripts/smoke_test.sh        # passed
PYTHON=.venv/bin/python bash scripts/run_all.sh configs/default.yaml 2>&1 | tee run_full.log
```

## Stage timings (H100)

| stage | wall time |
|---|---|
| 00 prepare | cached from smoke run |
| 01 EDA | 7 s |
| 02 features (105 features, 15.5 M rows) | 2 min |
| 02b baselines | 6 s |
| 02c prior NCF + LSTM | 2 min 22 s |
| 03 two-tower (fold0, fold1, full; 3 epochs each) | 11 min 50 s |
| 04 score tower (15.5 M rows) | 18 s |
| 05 LambdaRank (best iter 691) | 5 min 12 s |
| 06 predict | 20 s |
| 07 diagnostics (no ablations) | 1 min 34 s |

Total: ~24 minutes on the H100 after data preparation.

## Test metrics (`outputs/full/results/summary_test.csv`)

17,174 users.

| model | MRR | P@1 | NDCG@10 | gNDCG@10 | HR@10 | R@10 | R@50 |
|---|---|---|---|---|---|---|---|
| Random | 0.1005 | 0.0298 | 0.0850 | 0.0833 | 0.2266 | 0.1599 | 0.6081 |
| Popularity (all history) | 0.3435 | 0.2012 | 0.3525 | 0.3479 | 0.6526 | 0.5403 | 0.8660 |
| Popularity, clicks 7 d | 0.3311 | 0.1915 | 0.3386 | 0.3340 | 0.6349 | 0.5207 | 0.8403 |
| Popularity, carts 90 d | 0.3344 | 0.1952 | 0.3424 | 0.3382 | 0.6353 | 0.5237 | 0.8459 |
| Popularity, orders 365 d | 0.2929 | 0.1649 | 0.2954 | 0.2914 | 0.5646 | 0.4578 | 0.7942 |
| Impression CTR | 0.2753 | 0.1661 | 0.2676 | 0.2641 | 0.4997 | 0.3956 | 0.6331 |
| Query-item CTR | 0.1898 | 0.1069 | 0.1727 | 0.1698 | 0.3394 | 0.2577 | 0.5932 |
| Query-item engagement | 0.2441 | 0.1399 | 0.2343 | 0.2310 | 0.4461 | 0.3520 | 0.7084 |
| BM25 | 0.1096 | 0.0385 | 0.0959 | 0.0938 | 0.2352 | 0.1734 | 0.6327 |
| User affinity | 0.1106 | 0.0353 | 0.0969 | 0.0950 | 0.2508 | 0.1813 | 0.6500 |
| Prior team NCF+LSTM (fixed) | 0.1708 | 0.0734 | 0.1644 | 0.1619 | 0.3706 | 0.2851 | 0.7288 |
| Prior team NCF+LSTM (published) | 0.1350 | 0.0597 | 0.1220 | 0.1195 | 0.2812 | 0.2055 | 0.5286 |
| **Two-tower** | 0.3576 | 0.2160 | 0.3639 | 0.3592 | 0.6612 | 0.5473 | 0.8789 |
| **Two-tower + LambdaRank** | **0.4610** | **0.3165** | **0.4607** | **0.4551** | **0.7637** | **0.6433** | **0.9256** |

Gain over the best baseline (popularity): +34% MRR, +31% NDCG@10, +17% HR@10.

## Tower training

| run | epoch 0 gNDCG@10 | epoch 1 gNDCG@10 | epoch 2 gNDCG@10 |
|---|---|---|---|
| fold0 | 0.3441 | **0.3531** | 0.3423 |
| fold1 | 0.3483 | **0.3552** | 0.3467 |
| full | 0.3553 | **0.3616** | 0.3534 |

All three runs peak at epoch 1 and degrade at epoch 2 (overfitting).

## Ranker

- LightGBM `lambdarank`, 108 features, best iteration 691.
- Validation: NDCG@1 0.2958, NDCG@10 0.4530.
- Top features by gain: `tower_score_pct`, `tower_score_gap`, `ipop_click_7_pct`,
  `u_wid_last_days`, `qpop_wid_click_7_pct`, `ipop_order_365_pct`,
  `u_shop_id_last_days`, `lex_bm25_pct`, `ua_gap_days`, `ipop_users`.

## Diagnostics

- Tower attention mass by history segment: short (<7 d) **0.697**, mid (7–90 d) **0.032**,
  long (≥90 d) **0.0009** — the model effectively ignores mid/long history.
- SHAP by group: slate context 0.759, impression CTR 0.719, popularity 0.597,
  tower score 0.488, query→item history 0.416, personal history 0.266,
  lexical 0.230, user activity 0.097, item content 0.058, similar past queries 0.005.
- Ranker improves every segment over popularity (e.g. queries seen ≥10×: MRR 0.513 vs 0.405
  for the tower; unseen queries: 0.437 vs 0.338).
