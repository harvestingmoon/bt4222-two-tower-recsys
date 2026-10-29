# Overfitting: diagnosis, fix and latest full run

> **Note:** the tower and re-ranker results below were produced before the hard-negative leak fix
> (`Stores.neg_batch` drew negatives from validation/test and held-out-fold slates; see `report/overfit_fixes.md` §5).
> Stages 03–07 need re-running with `--force` before these numbers are quoted.

## What was overfitting

The latest code had already fixed the two-tower overfit (train out-of-fold was *below* validation), but the
**LightGBM re-ranker was overfitting**:

| model | train | val | test | train − test |
|---|---|---|---|---|
| two-tower NDCG@10 | 0.1828 (OOF) | 0.1851 | 0.1818 | −0.001 |
| re-ranker NDCG@10 | 0.4766 | 0.4592 | 0.4607 | **0.0159** |
| re-ranker MRR | 0.4825 | 0.4604 | 0.4613 | **0.0212** |

(The previous run had a train − test NDCG gap of **0.032** and MRR gap of **0.042**.)

## Why it happened

1. **Popularity moved into the least regularised part of the pipeline.** Removing popularity from the item tower
   took the tower SHAP contribution from **0.488 → 0.054**, while `ipop_click_7_pct` gain rose from **0.50 M →
   1.57 M**. The same signal re-entered as raw, high-cardinality item-level features that trees can memorise.
2. **Capacity vs label sparsity.** 108 features and 63 leaves over 12.3 M rows, but only ~1.86 positives per slate
   (~255 k positive labels in total). Rare item/CTR keys have very little support per split.
3. **Noisy implicit labels.** 97.7 % of impressions are label 0; clicks are position-biased. Graded gains let the
   ranker chase individual clicks.
4. **Cross-fitting mismatch.** With 2 folds, train users' tower/CTR features came from models trained on 50 % of
   users, while val/test used all training users.

## Fixes applied

- `prepare.n_folds: 2 → 5` — train out-of-fold features now use 80 % of users, closer to the val/test regime.
- `features.ctr_prior: 20 → 50` — stronger shrinkage of rare item-level CTR keys.
- Regularised LambdaRank: `lr 0.03`, `num_leaves 31`, `min_data_in_leaf 1000`, `max_depth 7`,
  `feature_fraction 0.5`, `bagging_fraction 0.7`, `lambda_l2 5`, `path_smooth 10`.
- `07_diagnostics.py` attention step updated for candidate-aware attention (it called the removed per-segment pools).

## Results

### Two-tower (5 folds + full; 2 epochs)

| split | MRR | NDCG@10 |
|---|---|---|
| train (out-of-fold) | 0.2049 | 0.1828 |
| val | 0.2062 | 0.1851 |
| test | 0.2045 | 0.1818 |

Train ≈ val ≈ test: the tower is **not overfitting**. Every fold's validation gNDCG@10 improved monotonically through
epoch 1 (e.g. fold3: 0.1638 → 0.1758 → 0.1829 → 0.1830).

### Re-ranker (train metrics on 25 % of training users)

| split | MRR | P@1 | NDCG@10 | gNDCG@10 | HR@10 | R@10 | R@50 |
|---|---|---|---|---|---|---|---|
| train | 0.4825 | 0.3422 | 0.4766 | — | 0.7731 | 0.6518 | 0.9260 |
| val | 0.4604 | 0.3170 | 0.4592 | — | 0.7633 | 0.6420 | 0.9233 |
| **test** | **0.4613** | **0.3174** | **0.4607** | **0.4550** | **0.7627** | **0.6430** | **0.9235** |

Against the previous run: test NDCG@10 **0.4601 → 0.4607**, MRR **0.4609 → 0.4613**, while the train–test gap roughly
halved. The test score is essentially flat, but the model is materially less overfit.

### Baselines (test)

| model | MRR | NDCG@10 |
|---|---|---|
| Popularity (all history) | 0.3435 | 0.3525 |
| Prior team NCF+LSTM (fixed) | 0.1708 | 0.1644 |
| Two-tower only | 0.2045 | 0.1818 |
| **Two-tower + LambdaRank** | **0.4613** | **0.4607** |

## Why 0.9 is not reachable

The test slates average 89.2 items with 1.86 positives; 97.7 % of labels are 0 and 23.6 % of users have no positive
in the model's top 10. Oracle bounds: perfect ordering of the current top-10 gives MRR 0.7736 / NDCG@10 0.6666;
top-50 gives 0.9705 / 0.9365; the whole slate gives 1.0 / 1.0. A 0.9 headline is therefore only available through
metrics that match the sparsity, e.g. **recall@50 = 0.9235** or **HR@50 = 0.970**.

## Reproduce

```bash
.venv/bin/python scripts/00_prepare.py --set paths.work_dir=outputs_v5
.venv/bin/python scripts/02_features.py --set paths.work_dir=outputs_v5
.venv/bin/python scripts/03_train_tower.py --set paths.work_dir=outputs_v5
.venv/bin/python scripts/04_score_tower.py --set paths.work_dir=outputs_v5
.venv/bin/python scripts/05_train_ranker.py --set paths.work_dir=outputs_v5
.venv/bin/python scripts/06_predict.py --set paths.work_dir=outputs_v5
```
