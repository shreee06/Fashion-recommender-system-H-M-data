# Test protocol (frozen before the test week is evaluated)

Written before any test-week number exists. The test week is **2020-09-16 to 2020-09-22**, cutoff **2020-09-16**.
Everything below is fixed: no model, feature, rule or hyperparameter will be changed after the test run, and the
test run happens exactly once (`scripts/run_test_final.py --i-am-done-tuning`; it writes `test_run_record.json` before
reading any test label and refuses to run again if that file exists).

## Frozen code and models

- **Frozen commit.** The last commit before this protocol is `b62cb35` ("Frozen reranker, cold-start decision and
  validation reference"). The code that will be run is the commit that adds this file ("Test protocol and dry run"),
  which differs from `b62cb35` only by this protocol, the `--dry-run-on-validation` option and its guard tests, the
  `load_D(truncate_before=...)` option, `memguard.py`, and a regenerated `validation_reference.json`. The run record stores
  the exact `git rev-parse HEAD` at run time, and the script refuses to run with an uncommitted working tree.
- **Reranker:** `models/lgbm_reranker.txt`, sha256 `afa0b276e07cfc115acecfc20085042548c2930d66f1029449e766a1012ce8f6`.
  LightGBM lambdarank trained on cutoff 2020-09-02 (labels 2020-09-02..08), 24,000 customers sampled with seed 0
  (30,000 sampled, 24,000/6,000 split), 7,728,754 candidate rows, 31 features, params num_leaves 31, learning rate 0.1,
  min_data_in_leaf 400, best iteration 45. It was reconstructed deterministically after Step 7 (the Step 7 run did not
  save it) and verified: inner-customer MAP@12 0.037150 and validation MAP@12 0.033653, both identical to Step 7. It is
  never retrained or fine-tuned. Its metadata is in `models/lgbm_reranker_meta.json`
  (sha256 `691ef7fdd356ba922e0d72e8ad980ee3aea8433c7e9d129fdb8cc3b6a6d28b7f`).
- **Two-tower:** `models/two_tower_best.pt`, sha256 `f0252e5c3c4ea1b8b5aeb3da35d2ad18af3e5fe410618ed2e16e61714a984edb`
  (Step 6 weights, trained on target weeks ending before 2020-09-02). At the test cutoff its customer vectors are
  recomputed from purchases before 2020-09-16 and the top 100 of the 26-week catalogue are retrieved (no training).
- **Step 5 candidate sources and combination**, re-fitted on data before 2020-09-16 with this exact configuration
  (from `evals/results/candidates_validation.json`):

```json
{
  "item_cf_new": {
    "window_weeks": 8,
    "max_items": 20000,
    "tau_weeks": 16,
    "k_seeds": 5,
    "modelled_items": 20000,
    "similarity_entries": 1926408,
    "training_customers": 383875
  },
  "variants": {
    "pop_weeks": 1,
    "k_seeds": 3
  },
  "segment_popularity": {
    "age_weeks": 1,
    "index_group_weeks": 1,
    "segment_used_in_combination": "segage",
    "sales_channel_segments": "not evaluated (channel not in the cached arrays)"
  },
  "combination": {
    "order": [
      "var",
      "cf",
      "segage"
    ],
    "caps": [
      2,
      4,
      100
    ],
    "n_repurchase": 4,
    "form": "repurchase top-R, then each source in order up to its cap (last uncapped), then global pop4w, deduplicated, exactly 12"
  }
}
```

  Candidate pool for the reranker = union of: own recent purchases, item-CF, variants, age-bucket popularity, global
  4-week popularity, two-tower top 100. All candidate sources and features use only transactions before 2020-09-16.

## Cold-start rule and the pre-declared secondary system

A cold-start fallback (customers with no purchase before the cutoff get the age-bucket popularity list instead of the
reranker) was pre-declared as a secondary system. The rule was decided on the 6,000 inner tuning customers of the
training week (cutoff 2020-09-02) only: the fallback is adopted iff its cold-customer MAP@12 exceeds the reranker's.
Result (`evals/results/cold_fallback_decision.json`, sha256 `b173673c4ecb4d0e6fa3fb62940cbe66ba1766f55081c3bb8b942a93385c8950`):
reranker 0.0158 vs fallback 0.0100 on 465 cold customers, so **the fallback is NOT adopted** and the secondary system
is identical to the primary. **Only one reranker system is reported.** Caveats: the cold subset behind this decision is
small (465 customers), and the validation-week numbers, which were deliberately not used for the decision, pointed the
other way (cold-customer MAP@12 0.0050 for the reranker vs 0.0087 for the age-bucket list). The rule is kept as decided.

## Systems reported on the test week

`popular_last_1w`, `repurchase + pop4w` (repurchase+last_4w), `Step 5 combination`, `two-tower alone`, `reranker`.
No other variants will be reported as results.

## Metrics and population

Population: all customers who bought in the test week. Metrics: MAP@12, Recall@12, NDCG@12 (95% bootstrap CIs),
coverage@12, share of recommended items already bought by the customer, MAP@12 for cold (no purchase before the
cutoff) vs warm customers, population size and cold share. Paired bootstrap differences: reranker vs Step 5 combination,
reranker vs repurchase + pop4w. A table compares every test number with its validation counterpart.
Bootstrap: customers resampled with replacement, **1000 resamples, seed 123**, percentile 95% intervals.

## Dry run on the validation week (done before this commit)

`run_test_final.py --dry-run-on-validation` runs the identical pipeline with cutoff 2020-09-09 and validation labels,
drops every transaction on or after 2020-09-16 at load time (asserted), and does not write the run record. Result on
72,019 validation-week buyers (7.49% cold); each MAP@12 matches the earlier steps to six decimals:

| system | MAP@12 | Recall@12 | NDCG@12 |
|---|---|---|---|
| popular_last_1w | 0.006604 | 0.0209 | 0.0126 |
| repurchase+last_4w | 0.024223 | 0.0443 | 0.0335 |
| combination_step5 | 0.026091 | 0.0551 | 0.0380 |
| two_tower_alone | 0.010597 | 0.0291 | 0.0184 |
| reranker | 0.033653 | 0.0724 | 0.0504 |

Dry run: peak RSS 2857 MB, 147.4 s.

## Commitment

No change to models, features, rules or hyperparameters will be made after the test run. Any result, including a
disappointing one, will be reported as measured.
