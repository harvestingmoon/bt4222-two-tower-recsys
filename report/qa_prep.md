# Presentation Q&A prep

The guideline says any member may be asked to explain data preprocessing, model logic, diagnostics or code. Each
question below has a short answer and the place in the code to look. Read the code behind each answer yourself; don't
memorise these lines.

## Data and preprocessing

**Why do orders and carts outnumber clicks?**
The history is three logs with different retention windows: clicks are kept for 7 days, carts and follows for 90 days,
orders for 365 days, all counted back from the test query. Beyond 7 days, clicks are simply not recorded.
→ `src/jds/windows.py`, `figures/history_timeline.png`, `figures/funnel_by_age.png`

**How do you get "days before the query" from the data?**
`history_time_list` holds gaps in seconds between consecutive events, and the last gap is the one to the test query.
Each event's age is the total time minus the cumulative sum up to that event.
→ `prepare.py`, the `t_before_query` column

**What leakage did you find?**
In the raw file every slate is sorted by label, with positives first in 100% of slates. We shuffle each slate with a
fixed seed and never use its position as a feature.
→ `prepare.py`, `prepare_stats.json → positives_first_in_raw_order`

**How are users split?**
80/10/10 train/val/test by user, seed 42, evaluating only users with at least one positive. Training users get 2 folds
for cross-fitting.
→ `prepare.py`

## Formulation

**Why re-rank the displayed slate instead of retrieving from the whole catalogue?**
It's the official JDsearch protocol. The labels only exist for displayed items, so anything outside the slate has no
ground truth.

**What does each data field do?**
See the table in the README ("The roles of the data fields").

**Which metrics, and why graded NDCG?**
MRR, P@1 and NDCG@10 follow the paper. Graded NDCG@10 (gain 2^label − 1) also rewards putting an order above a click.
→ `metrics.py`

## Features

**Why per-type windows instead of total counts or decay weights?**
Outside its window, a type's count is zero by construction, not by behaviour. Mixing types (e.g. a click ratio) mostly
measures how recent the user's history is: see `eda.json → recency_confound`.
→ `features.py`, and evidence in `window_ablation.csv`

**What is out-of-fold CTR?**
Impression CTR for a training user in fold k is computed only from the other fold's users. Validation and test users
use all training users. `tests/test_features.py::test_ctr_is_out_of_fold` checks this.

**What happens to cold items?**
They have no history, so they get content features: BM25, term coverage, brand/shop/category popularity. The item tower
is built from text and attributes, with no item-id embedding.

## Models

**Two-tower: what goes into each tower?**
- Item tower: title, brand and category term embeddings, plus id embeddings and popularity buckets.
- Query + user tower: the query terms, plus three history segments (short < 7 d, mid 7–90 d, long ≥ 90 d), each pooled
  by query-guided attention.

The score is cosine similarity between the two outputs.
→ `twotower/model.py`

**Why three segments?**
They match the retention windows: each segment holds a different mix of event types by construction.

**What is the loss?**
A listwise softmax over each training slate, with targets proportional to 2^label − 1, plus an auxiliary in-batch
softmax on historical query → item events from the last 7 days.
→ `twotower/train.py`

**What is cross-fitting, and why?**
The ranker trains on training users, so their tower scores must come from a tower that never saw their slates.
Otherwise the ranker learns to trust an overfitted score. `fold0` and `fold1` score each other's users; `full` scores
validation and test.

**Why LambdaRank on top?**
It optimises NDCG directly, combines the tower score with the hand-crafted features, and has label gain [0, 1, 3, 7].
→ `ranker/train.py`

## Baseline diagnosis

**Why does the previous team's model do worse?**
The numbers are in `prior_ncf_gaps.json`:
- the query is never an input;
- cold items have no embedding;
- an attribute-lookup bug in the published version;
- it trains on 1:4 sampled lists but is tested on full slates;
- its whole-history ratios mix retention windows.

**Why are their originally reported numbers higher?**
They were evaluated on sampled 1:4 lists, which are much easier than ~89-item slates. Compare the "[reference]" rows in
`prior_ncf.json`.

## Diagnostics

**Where does the model help most and least?**
→ `diag_segments.csv` / `.png`
*[Answer from the results]*

**Which features matter?**
→ `diag_shap_groups.csv`, `diag_ablations.csv`
*[Answer from the results]*

**What does the tower attend to?**
→ `diag_attention.json`
*[Answer from the results]*

**Is the insight table causal?**
No. It shows associations in logged data, under JD's own ranking.
→ `diag_insights.csv`
