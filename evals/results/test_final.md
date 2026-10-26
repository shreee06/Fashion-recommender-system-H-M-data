# Final test-week evaluation (2020-09-16 to 2020-09-22)

Population: all 68,984 customers who bought in the test week; 5,572 (8.08%) had no purchase before the cutoff 2020-09-16 (cold-start). 95% CIs: customer bootstrap, 1000 resamples, seed 123. Frozen systems only (see test_protocol.md); run record in test_run_record.json.

| system | MAP@12 [95% CI] | Recall@12 [95% CI] | NDCG@12 [95% CI] | coverage@12 | already-bought share | MAP cold | MAP warm |
|---|---|---|---|---|---|---|---|
| popular_last_1w | 0.0087 [0.0083, 0.0092] | 0.0255 [0.0246, 0.0264] | 0.0158 [0.0152, 0.0164] | 0.0001 | 0.009 | 0.0079 | 0.0088 |
| repurchase+last_4w | 0.0242 [0.0233, 0.0251] | 0.0441 [0.0428, 0.0454] | 0.0333 [0.0322, 0.0343] | 0.4842 | 0.829 | 0.0054 | 0.0258 |
| combination_step5 | 0.0263 [0.0254, 0.0272] | 0.0572 [0.0557, 0.0587] | 0.0385 [0.0374, 0.0396] | 0.3338 | 0.303 | 0.0089 | 0.0278 |
| two_tower_alone | 0.0091 [0.0086, 0.0095] | 0.0256 [0.0246, 0.0265] | 0.0159 [0.0153, 0.0165] | 0.0395 | 0.045 | 0.0035 | 0.0095 |
| reranker | 0.0339 [0.0329, 0.0349] | 0.0719 [0.0702, 0.0735] | 0.0501 [0.0489, 0.0512] | 0.1209 | 0.225 | 0.0057 | 0.0364 |

## Paired differences (A - B)

| A - B | metric | diff | 95% CI | excludes 0 |
|---|---|---|---|---|
| reranker - combination_step5 | map@12 | +0.0076 | [+0.0069, +0.0082] | yes |
| reranker - combination_step5 | recall@12 | +0.0147 | [+0.0135, +0.0160] | yes |
| reranker - combination_step5 | ndcg@12 | +0.0116 | [+0.0108, +0.0123] | yes |
| reranker - repurchase+last_4w | map@12 | +0.0097 | [+0.0091, +0.0104] | yes |
| reranker - repurchase+last_4w | recall@12 | +0.0278 | [+0.0265, +0.0290] | yes |
| reranker - repurchase+last_4w | ndcg@12 | +0.0168 | [+0.0160, +0.0176] | yes |

## Validation vs test (validation: 72,019 customers, 7.49% cold; same frozen code and models)

| system | metric | validation | test | change | test CI overlaps validation value? |
|---|---|---|---|---|---|
| popular_last_1w | map@12 | 0.0066 | 0.0087 | +0.0021 (+32.5%) | no |
| popular_last_1w | recall@12 | 0.0209 | 0.0255 | +0.0046 (+22.0%) | no |
| popular_last_1w | ndcg@12 | 0.0126 | 0.0158 | +0.0032 (+25.0%) | no |
| repurchase+last_4w | map@12 | 0.0242 | 0.0242 | -0.0000 (-0.2%) DROP | yes |
| repurchase+last_4w | recall@12 | 0.0443 | 0.0441 | -0.0002 (-0.5%) DROP | yes |
| repurchase+last_4w | ndcg@12 | 0.0335 | 0.0333 | -0.0003 (-0.8%) DROP | yes |
| combination_step5 | map@12 | 0.0261 | 0.0263 | +0.0002 (+0.9%) | yes |
| combination_step5 | recall@12 | 0.0551 | 0.0572 | +0.0021 (+3.7%) | no |
| combination_step5 | ndcg@12 | 0.0380 | 0.0385 | +0.0005 (+1.3%) | yes |
| two_tower_alone | map@12 | 0.0106 | 0.0091 | -0.0015 (-14.6%) DROP | no |
| two_tower_alone | recall@12 | 0.0291 | 0.0256 | -0.0035 (-12.2%) DROP | no |
| two_tower_alone | ndcg@12 | 0.0184 | 0.0159 | -0.0025 (-13.5%) DROP | no |
| reranker | map@12 | 0.0337 | 0.0339 | +0.0002 (+0.7%) | yes |
| reranker | recall@12 | 0.0724 | 0.0719 | -0.0006 (-0.8%) DROP | yes |
| reranker | ndcg@12 | 0.0504 | 0.0501 | -0.0003 (-0.6%) DROP | yes |
