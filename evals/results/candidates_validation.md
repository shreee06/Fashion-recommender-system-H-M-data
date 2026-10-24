# Candidate generators on the validation week

Validation week 2020-09-09 to 2020-09-15; all sources use only data before 2020-09-09 plus static metadata. **Evaluated population: all 72,019 validation-week buyers** (7.49% cold-start). 95% CIs: customer bootstrap, 1000 resamples, seed 123. Standalone sources are filled with global pop4w to 12 items.

Item-CF (new): trained on ALL customers, window 8 weeks, top 20,000 articles (20,000 modelled), 100 neighbours, recency tau = 16 weeks, K = 5 seed items.

Age buckets (customers in customers.csv / among evaluated): <25: 357,169/19,525, 25-34: 393,292/22,467, 35-44: 169,041/7,623, 45-54: 253,951/13,107, 55+: 182,666/8,976, unknown: 15,861/321.

Combination: repurchase top-4, then var > cf > segage with caps (2, 4, 100) (last uncapped), then pop4w.

| method | MAP@12 [95% CI] | Recall@12 [95% CI] | NDCG@12 [95% CI] | MAP cold | MAP warm |
|---|---|---|---|---|---|
| item_cf_step4 (200k sample, 26w, 20k items, binary, K=3) | 0.0070 [0.0066, 0.0075] | 0.0167 [0.0160, 0.0174] | 0.0114 [0.0109, 0.0120] | 0.0060 | 0.0071 |
| item_cf_all_customers_binary (26w, 20k, K=3) | 0.0091 [0.0087, 0.0096] | 0.0235 [0.0226, 0.0244] | 0.0153 [0.0147, 0.0159] | 0.0060 | 0.0094 |
| item_cf_new (best config) | 0.0098 [0.0094, 0.0103] | 0.0266 [0.0257, 0.0275] | 0.0169 [0.0164, 0.0175] | 0.0060 | 0.0101 |
| variants_alone (+pop4w fill) | 0.0101 [0.0096, 0.0106] | 0.0255 [0.0247, 0.0265] | 0.0167 [0.0161, 0.0173] | 0.0060 | 0.0104 |
| popular_by_age_bucket (1w) | 0.0075 [0.0071, 0.0079] | 0.0243 [0.0234, 0.0251] | 0.0144 [0.0139, 0.0149] | 0.0087 | 0.0074 |
| popular_by_index_group (1w, cold -> global pop4w) | 0.0071 [0.0067, 0.0075] | 0.0216 [0.0208, 0.0224] | 0.0132 [0.0127, 0.0137] | 0.0060 | 0.0072 |
| popular_last_4w (global) | 0.0056 [0.0053, 0.0059] | 0.0210 [0.0202, 0.0219] | 0.0118 [0.0114, 0.0123] | 0.0060 | 0.0056 |
| repurchase+last_4w | 0.0242 [0.0233, 0.0251] | 0.0443 [0.0431, 0.0456] | 0.0335 [0.0326, 0.0345] | 0.0060 | 0.0257 |
| combination_new | 0.0261 [0.0253, 0.0270] | 0.0551 [0.0537, 0.0566] | 0.0380 [0.0370, 0.0390] | 0.0087 | 0.0275 |
| combination_step4 (repurchase top8 + step-4 CF + pop4w) | 0.0248 [0.0240, 0.0257] | 0.0486 [0.0473, 0.0499] | 0.0351 [0.0341, 0.0361] | 0.0060 | 0.0264 |

## Paired differences (A - B, same customers)

| A - B | metric | diff | 95% CI | excludes 0 |
|---|---|---|---|---|
| item_cf_all_customers_binary (26w, 20k, K=3) - item_cf_step4 (200k sample, 26w, 20k items, binary, K=3) | map@12 | +0.0021 | [+0.0018, +0.0023] | yes |
| item_cf_all_customers_binary (26w, 20k, K=3) - item_cf_step4 (200k sample, 26w, 20k items, binary, K=3) | recall@12 | +0.0068 | [+0.0063, +0.0074] | yes |
| item_cf_all_customers_binary (26w, 20k, K=3) - item_cf_step4 (200k sample, 26w, 20k items, binary, K=3) | ndcg@12 | +0.0039 | [+0.0035, +0.0042] | yes |
| item_cf_new (best config) - item_cf_all_customers_binary (26w, 20k, K=3) | map@12 | +0.0007 | [+0.0004, +0.0010] | yes |
| item_cf_new (best config) - item_cf_all_customers_binary (26w, 20k, K=3) | recall@12 | +0.0031 | [+0.0024, +0.0039] | yes |
| item_cf_new (best config) - item_cf_all_customers_binary (26w, 20k, K=3) | ndcg@12 | +0.0017 | [+0.0012, +0.0021] | yes |
| item_cf_new (best config) - item_cf_step4 (200k sample, 26w, 20k items, binary, K=3) | map@12 | +0.0028 | [+0.0024, +0.0031] | yes |
| item_cf_new (best config) - item_cf_step4 (200k sample, 26w, 20k items, binary, K=3) | recall@12 | +0.0099 | [+0.0092, +0.0107] | yes |
| item_cf_new (best config) - item_cf_step4 (200k sample, 26w, 20k items, binary, K=3) | ndcg@12 | +0.0055 | [+0.0050, +0.0060] | yes |
| variants_alone (+pop4w fill) - popular_last_4w (global) | map@12 | +0.0045 | [+0.0040, +0.0050] | yes |
| variants_alone (+pop4w fill) - popular_last_4w (global) | recall@12 | +0.0045 | [+0.0036, +0.0054] | yes |
| variants_alone (+pop4w fill) - popular_last_4w (global) | ndcg@12 | +0.0049 | [+0.0042, +0.0055] | yes |
| popular_by_age_bucket (1w) - popular_last_4w (global) | map@12 | +0.0019 | [+0.0015, +0.0023] | yes |
| popular_by_age_bucket (1w) - popular_last_4w (global) | recall@12 | +0.0032 | [+0.0025, +0.0040] | yes |
| popular_by_age_bucket (1w) - popular_last_4w (global) | ndcg@12 | +0.0026 | [+0.0021, +0.0030] | yes |
| popular_by_index_group (1w, cold -> global pop4w) - popular_last_4w (global) | map@12 | +0.0015 | [+0.0012, +0.0018] | yes |
| popular_by_index_group (1w, cold -> global pop4w) - popular_last_4w (global) | recall@12 | +0.0006 | [-0.0001, +0.0013] | no |
| popular_by_index_group (1w, cold -> global pop4w) - popular_last_4w (global) | ndcg@12 | +0.0014 | [+0.0010, +0.0018] | yes |
| combination_new - repurchase+last_4w | map@12 | +0.0019 | [+0.0017, +0.0021] | yes |
| combination_new - repurchase+last_4w | recall@12 | +0.0108 | [+0.0098, +0.0119] | yes |
| combination_new - repurchase+last_4w | ndcg@12 | +0.0045 | [+0.0040, +0.0049] | yes |
| combination_new - combination_step4 (repurchase top8 + step-4 CF + pop4w) | map@12 | +0.0012 | [+0.0010, +0.0014] | yes |
| combination_new - combination_step4 (repurchase top8 + step-4 CF + pop4w) | recall@12 | +0.0065 | [+0.0055, +0.0074] | yes |
| combination_new - combination_step4 (repurchase top8 + step-4 CF + pop4w) | ndcg@12 | +0.0029 | [+0.0025, +0.0033] | yes |
| combination_step4 (repurchase top8 + step-4 CF + pop4w) - repurchase+last_4w | map@12 | +0.0006 | [+0.0005, +0.0007] | yes |
| combination_step4 (repurchase top8 + step-4 CF + pop4w) - repurchase+last_4w | recall@12 | +0.0043 | [+0.0037, +0.0050] | yes |
| combination_step4 (repurchase top8 + step-4 CF + pop4w) - repurchase+last_4w | ndcg@12 | +0.0015 | [+0.0013, +0.0018] | yes |

## Candidate recall (each source's own list, unfilled)

| source | recall@12 | recall@50 | recall@100 | mean list length |
|---|---|---|---|---|
| repurchase (recents) | 0.0406 | 0.0475 | 0.0490 | 41.1 |
| item_cf_step4 | 0.0127 | 0.0209 | 0.0264 | 83.0 |
| item_cf_new | 0.0182 | 0.0314 | 0.0399 | 62.0 |
| variants | 0.0161 | 0.0183 | 0.0183 | 8.6 |
| popular_by_age_bucket | 0.0243 | 0.0697 | 0.1111 | 100.0 |
| popular_by_index_group | 0.0216 | 0.0626 | 0.1034 | 100.0 |
| popular_last_4w | 0.0210 | 0.0590 | 0.0945 | 100.0 |
| UNION (round-robin merge of: repurchase (recents), item_cf_new, variants, popular_by_age_bucket, popular_by_index_group, popular_last_4w) | 0.0502 | 0.1036 | 0.1433 | 290.3 |

Union pool: recall 0.2338 at mean pool size 290.3 (this is the ceiling for a reranker over the full pool).
