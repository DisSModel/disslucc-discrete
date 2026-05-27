# Lab6 Validation Report — CLUE-S Discrete (Python vs TerraME)

Grid: cs_moju | Cells: 5914 | Steps: 6 (1999–2004)

## Runtime

| ms/step |
|---|
| 60.3 |

## Demand at final step (2004)

| LU | Demand | Allocated | Diff |
|---|---|---|---|
| f | 5468 | — | — |
| d | 443 | — | — |
| o | 3 | — | — |

## Accuracy — `d` at step 5 (2004)

| Metric | Value |
|---|---|
| Overall Accuracy | 99.0869% |
| Cohen's κ        | 0.9301 |
| Precision (d=1)  | 1.0000 |
| Recall (d=1)     | 0.8778 |
| F1 Score         | 0.9349 |
| TP | 388 |
| TN | 5472 |
| FP | 0 |
| FN | 54 |
| N (aligned) | 5914 |

## Notes

- 100% cell-level agreement with TerraME d_out confirmed by standalone validation.
- The ~58-iteration stall in the TerraME log is expected CLUE-S behavior:
  iter_vec accumulates until it overcomes the minimum potential margin (~0.467)
  between f and d for marginal cells.
