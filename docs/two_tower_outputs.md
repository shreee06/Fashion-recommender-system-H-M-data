# Two-tower candidate files (for the reranker step)

Produced by `scripts/run_two_tower.py` into `data/processed/` (git-ignored, never committed).
Model weights: `models/two_tower_best.pt` (git-ignored).

| file | cutoff | rows |
|---|---|---|
| `two_tower_candidates_validation.npz` | 2020-09-09 (validation start), `cutoff_day` = 18514 | 72,019 customers: all validation-week buyers |
| `two_tower_candidates_heldout.npz` | 2020-09-02, `cutoff_day` = 18507 | 75,822 customers: all buyers of the held-out training week 2020-09-02..08 |

Arrays in each file (days are days since 1970-01-01):

- `customers`: int32 (n,), customer codes = row positions in `data/processed/customers.parquet`, sorted ascending.
- `cutoff_day`: scalar, the first day NOT used as input (inputs are purchases strictly before it).
- `codes`: int32 (n, 100), article codes = row positions in `data/processed/articles.parquet`; best first, so the column index is the rank (0 = best).
- `scores`: float32 (n, 100), cosine similarity between customer and item vectors, same order as `codes`.

Notes:

- The top 100 are taken over the modelled catalogue (articles sold in the 26 weeks before the cutoff) and are NOT filtered for already-bought items; filter on the reranker side if wanted.
- The model was trained only on target weeks ending before 2020-09-02 (12 weekly cutoffs, 2020-06-10 up to 2020-08-26 as cutoff dates; the held-out week 2020-09-02..08 was used only for early stopping and tuning). The held-out file is therefore a clean, un-trained-on week suitable for fitting a reranker whose scores are then applied to the validation file.
- Relevant-item labels are not stored; rebuild them with `fashion_recsys.evaluation.build_eval_set`.
