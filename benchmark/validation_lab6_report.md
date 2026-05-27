# Lab6 Validation Report — CLUE-S Discrete

Grid: cs_moju | Cells: 5914 | Steps: 6 (1999–2004)

## Runtime

| ms/step |
|---|
| 70.3 |

## Demand check (step 5 = 2004)

| LU | Demand | Note |
|---|---|---|
| f  | 5468 | floresta |
| d  | 443 | desmatamento |
| o  | 3 | outros (estático) |

## Accuracy — `d` at step 5 (2004)

| Metric | Value |
|---|---|
| Overall Accuracy | 99.0869% |
| Cohen's κ        | 0.9301 |
| Precision (d=1)  | 1.0000 |
| Recall (d=1)     | 0.8778 |
| F1 Score         | 0.9349 |
| TP               | 388 |
| TN               | 5472 |
| FP               | 0 |
| FN               | 54 |
| N (aligned)      | 5914 |
