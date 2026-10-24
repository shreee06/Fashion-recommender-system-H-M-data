# Item-CF and combinations on the validation week

Validation week 2020-09-09 to 2020-09-15; everything trained on data before 2020-09-09. **Evaluated population: all 72,019 customers who bought in the validation week** (7.49% cold-start; 10,557 of them are in the 200k sample). Item-CF similarity is trained on the seeded 200k customer sample only, using the last 26 weeks, the 20,000 most popular articles of that window, 100 neighbours per article, seeds = last K=3 distinct purchases (K chosen on validation, see tuning_log.csv). 95% CIs: customer bootstrap, 1000 resamples, seed 123.

| method | MAP@12 [95% CI] | Recall@12 [95% CI] | NDCG@12 [95% CI] | coverage@12 | MAP cold | MAP warm |
|---|---|---|---|---|---|---|
| popular_global | 0.0034 [0.0032, 0.0037] | 0.0090 [0.0085, 0.0095] | 0.0063 [0.0060, 0.0067] | 0.0001 | 0.0039 | 0.0034 |
| popular_last_1w | 0.0066 [0.0062, 0.0070] | 0.0209 [0.0202, 0.0217] | 0.0126 [0.0121, 0.0131] | 0.0001 | 0.0064 | 0.0066 |
| popular_last_4w | 0.0056 [0.0053, 0.0059] | 0.0210 [0.0202, 0.0219] | 0.0118 [0.0114, 0.0123] | 0.0001 | 0.0060 | 0.0056 |
| repurchase+last_1w | 0.0243 [0.0234, 0.0252] | 0.0444 [0.0432, 0.0457] | 0.0336 [0.0327, 0.0346] | 0.4875 | 0.0064 | 0.0257 |
| repurchase+last_4w | 0.0242 [0.0233, 0.0251] | 0.0443 [0.0431, 0.0456] | 0.0335 [0.0326, 0.0345] | 0.4875 | 0.0060 | 0.0257 |
| item_cf(K=3) | 0.0070 [0.0066, 0.0075] | 0.0167 [0.0160, 0.0174] | 0.0114 [0.0109, 0.0120] | 0.1837 | 0.0060 | 0.0071 |
| repurchase(top8)+item_cf+pop4w | 0.0248 [0.0240, 0.0257] | 0.0486 [0.0473, 0.0499] | 0.0351 [0.0341, 0.0361] | 0.4232 | 0.0060 | 0.0264 |
| repurchase(all)+item_cf+pop4w | 0.0244 [0.0235, 0.0253] | 0.0448 [0.0435, 0.0461] | 0.0338 [0.0328, 0.0347] | 0.4890 | 0.0060 | 0.0259 |

## Paired differences (same customers, bootstrap)

| A - B | metric | diff | 95% CI | excludes 0 |
|---|---|---|---|---|
| repurchase(top8)+item_cf+pop4w - repurchase+last_4w | map@12 | +0.0006 | [+0.0005, +0.0007] | yes |
| repurchase(top8)+item_cf+pop4w - repurchase+last_4w | recall@12 | +0.0043 | [+0.0037, +0.0050] | yes |
| repurchase(top8)+item_cf+pop4w - repurchase+last_4w | ndcg@12 | +0.0015 | [+0.0013, +0.0018] | yes |
| repurchase(all)+item_cf+pop4w - repurchase+last_4w | map@12 | +0.0002 | [+0.0001, +0.0002] | yes |
| repurchase(all)+item_cf+pop4w - repurchase+last_4w | recall@12 | +0.0005 | [+0.0001, +0.0008] | yes |
| repurchase(all)+item_cf+pop4w - repurchase+last_4w | ndcg@12 | +0.0002 | [+0.0001, +0.0004] | yes |
| item_cf(K=3) - repurchase+last_4w | map@12 | -0.0172 | [-0.0181, -0.0162] | yes |
| item_cf(K=3) - repurchase+last_4w | recall@12 | -0.0276 | [-0.0290, -0.0263] | yes |
| item_cf(K=3) - repurchase+last_4w | ndcg@12 | -0.0221 | [-0.0232, -0.0210] | yes |
| item_cf(K=3) - popular_last_4w | map@12 | +0.0014 | [+0.0010, +0.0019] | yes |
| item_cf(K=3) - popular_last_4w | recall@12 | -0.0044 | [-0.0053, -0.0035] | yes |
| item_cf(K=3) - popular_last_4w | ndcg@12 | -0.0004 | [-0.0010, +0.0002] | no |
| repurchase+last_4w - repurchase+last_1w | map@12 | -0.0001 | [-0.0002, +0.0000] | no |
| repurchase+last_4w - repurchase+last_1w | recall@12 | -0.0001 | [-0.0003, +0.0001] | no |
| repurchase+last_4w - repurchase+last_1w | ndcg@12 | -0.0001 | [-0.0002, +0.0000] | no |

## Candidate recall (mean share of each customer's relevant items found in the top N)

| method | top 12 | top 50 | top 100 |
|---|---|---|---|
| item_cf(K=3) | 0.0167 | 0.0312 | 0.0429 |
| repurchase+last_4w | 0.0443 | 0.0744 | 0.1086 |
| popular_last_4w | 0.0210 | 0.0590 | 0.0945 |
| repurchase(top8)+item_cf+pop4w | 0.0486 | 0.0671 | 0.0792 |
| repurchase(all)+item_cf+pop4w | 0.0448 | 0.0670 | 0.0831 |

Unranked union pool, union(repurchase<=100 + item_cf top100 + pop4w top100): recall 0.1568 with mean pool size 222.2 (upper bound for a reranker over this pool).
