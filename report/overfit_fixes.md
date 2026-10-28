# Overfitting fixes applied to the two-tower

Symptoms: training loss kept falling while validation gN DCG@10 peaked early and then declined, and the ranker
gained almost nothing from `tower_score`. Six changes, in the order they matter.

## 1. Popularity moved out of the tower, into an explicit training offset

The item tower used to embed three windowed popularity buckets, so a large part of its capacity went on
reproducing a score the ranker already has as a feature. `ItemTower` no longer sees popularity at all. Instead
`TwoTower.pop_logit` adds `sum_j w_j * log2(1 + count_j)` to the training score, with `w` learned.

The point is the export: `tower_score` is still pure cosine, so the ranker receives a signal that is *complementary*
to its own popularity feature rather than a noisier copy of it.

## 2. Candidate-aware history attention (DIN-style)

The user tower used to pool each history segment once per user, which gives every candidate in the slate the same
user vector — so the only thing separating two candidates was the item side. `CandidateAttention` now lets each
candidate attend over the user's short-window history, with the attention score conditioned on the candidate *and*
the query text. The user vector is therefore `[B, S, D]`, not `[B, D]`.

Exact-match flags ride along as raw features: for each of the three type windows, whether the candidate shares a
product or a top category with something in that history. This is what lets history say "this exact product was in
my history" instead of only shifting a pooled average.

## 3. Smaller vocabulary

- `min_term_count` 3 -> 20, `max_terms` 400k -> 50k, `d_term` 64 -> 32.
- Rare `brand_id` / `shop_id` no longer collapse to a single dead index; they hash into 2048 shared buckets, so
  the long tail keeps a little signal instead of one common bucket.

400k near-singleton terms meant most embedding rows were seen a handful of times — pure memorisation.

## 4. Regularisation

`weight_decay` 1e-6 -> 1e-4, `dropout` 0.15 in both tower MLPs, `word_dropout` 0.15 (a share of each item's title
terms is hidden during training so the tower cannot lean on one distinctive word), `lr` 2e-3 -> 1e-3, epochs 3 -> 2,
and validation now runs twice per epoch (`eval_per_epoch: 2`) so the best checkpoint inside an epoch is not missed.

## 5. The auxiliary loss

Two fixes to the in-batch softmax on historical `(query, item)` pairs:

- **Popularity correction.** In-batch negatives are effectively sampled proportional to frequency, so the loss
  rewards items for being common. `inbatch_loss` subtracts `model.logq(...)`, a learned multiple of log popularity
  (the sampled-softmax bias correction).
- **Slate hard negatives.** Each anchor now also competes against up to `aux.hard_negatives: 8` items from *its own*
  slate that the user was shown and passed over — much harder than another anchor's positive. `Stores.neg_batch`
  builds them vectorised; the positive is excluded and invalid slots are masked.

## 6. History segments simplified

Only the short window is attended. Mid and long keep no attention pool; their content is summarised by
`ctx[seg]["tcount"]` — four per-type counts plus the exact log size. Those 15 numbers replace what were two more
attention pools, which is both less capacity (good, given the overfitting) and cheaper.

They also feed the exact-match flags above, which is where most of the mid/long value actually was.

## What this does not fix

Nothing here is validated: there is no data in this repo, so the only checks run were shape/behaviour assertions on
synthetic batches plus the existing unit tests (17 passing). The changes are mechanistic, not tuned. Expect to
re-run `scripts/03_train_tower.py --force` (assets must be rebuilt: the vocabulary and id sizes changed), then
`04_score_tower.py --force` and the ranker.
