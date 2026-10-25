# Two-tower retrieval on the validation week

Evaluated population: all 72,019 customers who bought in the validation week (7.49% cold-start). Retrieval is over the modelled catalogue of 50,641 articles (sold in the 26 weeks before the cutoff); 94.78% of relevant (customer, article) pairs are inside it. 95% CIs: customer bootstrap, 1000 resamples, seed 123.

Chosen config (tuned on a held-out TRAINING week, not on the validation week): `d=64;K=20;tau_in=8.0;temp=0.07;lr=0.004;bs=2048;logq=1;shared=1;content=1;train_weeks=12;lookback=26`. 6,804,136 parameters; 51 s/epoch on CPU; best epoch 11 of 12.

| method | MAP@12 [95% CI] | Recall@12 [95% CI] | NDCG@12 [95% CI] | MAP cold | MAP warm |
|---|---|---|---|---|---|
| item_cf_new (Step 5, filled) | 0.0098 [0.0094, 0.0103] | 0.0266 [0.0257, 0.0275] | 0.0169 [0.0164, 0.0175] | 0.0060 | 0.0101 |
| combination_step5 | 0.0261 [0.0253, 0.0270] | 0.0551 [0.0537, 0.0566] | 0.0380 [0.0370, 0.0390] | 0.0087 | 0.0275 |
| two_tower (chosen) | 0.0106 [0.0101, 0.0111] | 0.0291 [0.0282, 0.0301] | 0.0184 [0.0178, 0.0191] | 0.0046 | 0.0111 |
| two_tower (chosen) [exclude already-bought] | 0.0071 [0.0068, 0.0075] | 0.0213 [0.0204, 0.0221] | 0.0133 [0.0128, 0.0138] | 0.0046 | 0.0073 |
| two_tower logQ flipped | 0.0050 [0.0047, 0.0054] | 0.0135 [0.0128, 0.0142] | 0.0084 [0.0079, 0.0088] | 0.0006 | 0.0054 |
| two_tower logQ flipped [exclude already-bought] | 0.0023 [0.0021, 0.0025] | 0.0076 [0.0071, 0.0081] | 0.0044 [0.0042, 0.0047] | 0.0006 | 0.0024 |
| two_tower ID-only item tower | 0.0100 [0.0096, 0.0105] | 0.0277 [0.0268, 0.0286] | 0.0174 [0.0168, 0.0181] | 0.0052 | 0.0104 |
| two_tower ID-only item tower [exclude already-bought] | 0.0067 [0.0063, 0.0070] | 0.0204 [0.0196, 0.0212] | 0.0126 [0.0121, 0.0131] | 0.0052 | 0.0068 |

## Paired differences (A - B)

| A - B | metric | diff | 95% CI | excludes 0 |
|---|---|---|---|---|
| two_tower - item_cf_new | map@12 | +0.0008 | [+0.0001, +0.0014] | yes |
| two_tower - item_cf_new | recall@12 | +0.0025 | [+0.0013, +0.0038] | yes |
| two_tower - item_cf_new | ndcg@12 | +0.0015 | [+0.0007, +0.0023] | yes |
| two_tower - combination_step5 | map@12 | -0.0155 | [-0.0164, -0.0146] | yes |
| two_tower - combination_step5 | recall@12 | -0.0260 | [-0.0275, -0.0246] | yes |
| two_tower - combination_step5 | ndcg@12 | -0.0196 | [-0.0206, -0.0186] | yes |
| two_tower[exclude seen] - item_cf_new | map@12 | -0.0027 | [-0.0033, -0.0022] | yes |
| two_tower[exclude seen] - item_cf_new | recall@12 | -0.0054 | [-0.0065, -0.0043] | yes |
| two_tower[exclude seen] - item_cf_new | ndcg@12 | -0.0037 | [-0.0044, -0.0029] | yes |
| two_tower[exclude seen] - combination_step5 | map@12 | -0.0190 | [-0.0199, -0.0180] | yes |
| two_tower[exclude seen] - combination_step5 | recall@12 | -0.0339 | [-0.0354, -0.0324] | yes |
| two_tower[exclude seen] - combination_step5 | ndcg@12 | -0.0247 | [-0.0258, -0.0236] | yes |
| two_tower(ID+content) - two_tower(ID-only) | map@12 | +0.0006 | [+0.0002, +0.0009] | yes |
| two_tower(ID+content) - two_tower(ID-only) | recall@12 | +0.0014 | [+0.0007, +0.0022] | yes |
| two_tower(ID+content) - two_tower(ID-only) | ndcg@12 | +0.0010 | [+0.0005, +0.0014] | yes |
| two_tower(logQ as chosen) - two_tower(logQ flipped) | map@12 | +0.0056 | [+0.0051, +0.0061] | yes |
| two_tower(logQ as chosen) - two_tower(logQ flipped) | recall@12 | +0.0156 | [+0.0147, +0.0166] | yes |
| two_tower(logQ as chosen) - two_tower(logQ flipped) | ndcg@12 | +0.0101 | [+0.0095, +0.0107] | yes |

## Two-tower candidate recall

| variant | recall@50 | recall@100 | recall@100 cold | recall@100 warm |
|---|---|---|---|---|
| two_tower (chosen) | 0.0713 | 0.1060 | 0.0613 | 0.1096 |
| two_tower (chosen) [exclude already-bought] | 0.0574 | 0.0880 | nan | nan |
| two_tower logQ flipped | 0.0322 | 0.0491 | 0.0125 | 0.0520 |
| two_tower logQ flipped [exclude already-bought] | 0.0222 | 0.0366 | nan | nan |
| two_tower ID-only item tower | 0.0684 | 0.1014 | 0.0639 | 0.1045 |
| two_tower ID-only item tower [exclude already-bought] | 0.0554 | 0.0846 | nan | nan |

## Union pool (round-robin merge, deduplicated)

| pool | recall@12 | recall@50 | recall@100 | recall (full pool) | mean pool size |
|---|---|---|---|---|---|
| Step 5 sources | 0.0502 | 0.1036 | 0.1433 | 0.2338 | 290.3 |
| + two-tower top-100 | 0.0488 | 0.1051 | 0.1467 | 0.2634 | 354.8 |

Share of relevant items found only by the two-tower (not in the Step 5 pool): 0.0296.
