# Candidate pool + LightGBM reranker on the validation week

Evaluated population: all 72,019 validation-week buyers (7.49% cold-start). 95% CIs: customer bootstrap, 1000 resamples, seed 123. Trained on cutoff 2020-09-02: 24,000 fit customers (7,728,754 candidate rows) + 6,000 inner-validation customers, sampled at random (seed 0) from the 75,822 buyers of that label week; every pool candidate kept (no negative subsampling). LightGBM lambdarank, params {'num_leaves': 31, 'learning_rate': 0.1, 'min_data_in_leaf': 400}, best iteration 45, 31 features, tuned on the inner customers of the training week only.

| method | MAP@12 [95% CI] | Recall@12 [95% CI] | NDCG@12 [95% CI] | coverage@12 | already-bought share | MAP cold | MAP warm | Recall cold | Recall warm |
|---|---|---|---|---|---|---|---|---|---|
| repurchase+last_4w | 0.0242 [0.0233, 0.0251] | 0.0443 [0.0431, 0.0456] | 0.0335 [0.0326, 0.0345] | 0.4875 | 0.835 | 0.0060 | 0.0257 | 0.0219 | 0.0461 |
| combination_step5 | 0.0261 [0.0253, 0.0270] | 0.0551 [0.0537, 0.0566] | 0.0380 [0.0370, 0.0390] | 0.3364 | 0.306 | 0.0087 | 0.0275 | 0.0278 | 0.0574 |
| two_tower_alone | 0.0106 [0.0101, 0.0111] | 0.0291 [0.0282, 0.0301] | 0.0184 [0.0178, 0.0191] | 0.0389 | 0.047 | 0.0046 | 0.0111 | 0.0147 | 0.0303 |
| reranker (full) | 0.0337 [0.0327, 0.0346] | 0.0724 [0.0709, 0.0740] | 0.0504 [0.0492, 0.0515] | 0.1273 | 0.223 | 0.0050 | 0.0360 | 0.0240 | 0.0764 |
| reranker without two-tower features | 0.0335 [0.0325, 0.0344] | 0.0715 [0.0700, 0.0731] | 0.0500 [0.0488, 0.0511] | 0.1252 | 0.226 | 0.0055 | 0.0357 | 0.0249 | 0.0753 |
| reranker without content features | 0.0342 [0.0332, 0.0352] | 0.0724 [0.0709, 0.0740] | 0.0509 [0.0498, 0.0520] | 0.1339 | 0.221 | 0.0049 | 0.0366 | 0.0181 | 0.0768 |

## Paired differences (A - B)

| A - B | metric | diff | 95% CI | excludes 0 |
|---|---|---|---|---|
| reranker (full) - combination_step5 | map@12 | +0.0076 | [+0.0069, +0.0082] | yes |
| reranker (full) - combination_step5 | recall@12 | +0.0173 | [+0.0162, +0.0185] | yes |
| reranker (full) - combination_step5 | ndcg@12 | +0.0124 | [+0.0116, +0.0131] | yes |
| reranker (full) - repurchase+last_4w | map@12 | +0.0094 | [+0.0088, +0.0101] | yes |
| reranker (full) - repurchase+last_4w | recall@12 | +0.0281 | [+0.0268, +0.0294] | yes |
| reranker (full) - repurchase+last_4w | ndcg@12 | +0.0168 | [+0.0160, +0.0176] | yes |
| reranker (full) - two_tower_alone | map@12 | +0.0231 | [+0.0221, +0.0240] | yes |
| reranker (full) - two_tower_alone | recall@12 | +0.0433 | [+0.0418, +0.0449] | yes |
| reranker (full) - two_tower_alone | ndcg@12 | +0.0319 | [+0.0308, +0.0330] | yes |
| reranker (full) - reranker without two-tower features | map@12 | +0.0002 | [-0.0002, +0.0006] | no |
| reranker (full) - reranker without two-tower features | recall@12 | +0.0009 | [+0.0001, +0.0017] | yes |
| reranker (full) - reranker without two-tower features | ndcg@12 | +0.0004 | [-0.0000, +0.0009] | no |
| reranker (full) - reranker without content features | map@12 | -0.0006 | [-0.0010, -0.0001] | yes |
| reranker (full) - reranker without content features | recall@12 | +0.0000 | [-0.0008, +0.0009] | no |
| reranker (full) - reranker without content features | ndcg@12 | -0.0005 | [-0.0011, +0.0001] | no |

## Pool ceiling vs achieved

Mean pool size 324.2 candidates per customer. Pool recall (share of relevant items anywhere in the pool): 0.2456 (cold 0.1633, warm 0.2523). Reranked top-12 recall: 0.0724, i.e. 29.5% of the ceiling.

## Feature importance (share of total gain, full model)

| feature | share |
|---|---|
| art_department | 0.3489 |
| pair_prod_days_since | 0.1298 |
| art_pop_1w | 0.0774 |
| cust_weeks_since_last | 0.0474 |
| pair_dept_count | 0.0469 |
| pair_art_days_since | 0.0429 |
| art_age_days | 0.0415 |
| art_colour_group | 0.0403 |
| rank_tt | 0.0323 |
| art_product_type | 0.0316 |
| rank_recent | 0.0311 |
| art_pop_4w | 0.0244 |
| rank_cf | 0.0171 |
| tt_score | 0.0143 |
| cust_n_purchases | 0.0132 |
| pair_prod_count | 0.0106 |
| rank_agepop | 0.0084 |
| rank_pop | 0.0061 |
| pair_art_count | 0.0052 |
| price_ratio | 0.0046 |
| pair_colour_count | 0.0035 |
| art_pop_12w | 0.0034 |
| cust_avg_price | 0.0030 |
| pair_ptype_count | 0.0030 |
| cust_share_ch1 | 0.0029 |
| cust_age | 0.0026 |
| rank_var | 0.0018 |
| art_price | 0.0018 |
| art_days_since_sale | 0.0016 |
| n_other_sources | 0.0015 |
| cust_share_ch2 | 0.0010 |
