# Baselines on the validation week

Validation week 2020-09-09 to 2020-09-15; training data before 2020-09-09. Sample: 200,000 customers (seed 42), of which 10,557 bought in the validation week and are evaluated. Cold-start (no purchase before the cutoff): 7.19% of evaluated customers. Coverage = distinct recommended articles / 105,542 catalog articles.

| baseline | MAP@12 | Recall@12 | NDCG@12 | coverage@12 | MAP cold | MAP warm | NDCG cold | NDCG warm |
|---|---|---|---|---|---|---|---|---|
| popular_global | 0.0035 | 0.0095 | 0.0065 | 0.0001 | 0.0046 | 0.0034 | 0.0090 | 0.0063 |
| popular_last_1w | 0.0068 | 0.0214 | 0.0129 | 0.0001 | 0.0045 | 0.0070 | 0.0110 | 0.0130 |
| popular_last_4w | 0.0058 | 0.0218 | 0.0121 | 0.0001 | 0.0066 | 0.0057 | 0.0132 | 0.0120 |
| repurchase+last_1w | 0.0249 | 0.0463 | 0.0345 | 0.2580 | 0.0045 | 0.0265 | 0.0110 | 0.0363 |
| repurchase+last_4w | 0.0251 | 0.0467 | 0.0347 | 0.2580 | 0.0066 | 0.0265 | 0.0132 | 0.0364 |

Repurchase fallback chosen on validation MAP@12: **repurchase+last_4w**. Best baseline overall: **repurchase+last_4w**.
