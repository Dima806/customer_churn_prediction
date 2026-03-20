# Evaluation: Guardrails and Metrics

## Guardrail System

All guardrails live in [src/evaluation/guardrails.py](../src/evaluation/guardrails.py).

Each check returns a standard dict:
```python
{"check": str, "value": float, "threshold": float, "passed": bool, "message": str}
```

Failures are **logged** but do **not** raise exceptions. Guardrail results are saved to `artifacts/guardrails.json` for dashboard display.

Entry point: `run_all_guardrails(metrics, calibration, predictions, customers, train_metrics=None, baseline_metrics=None, shift_metrics=None)`

## Guardrail Thresholds (from `src/config.py`)

### ML Model Checks
| Check | Threshold | Direction |
|---|---|---|
| `MIN_ROC_AUC` | 0.65 | >= |
| `MAX_BRIER_SCORE` | 0.25 | <= |
| `MAX_ECE` | 0.10 | <= (Expected Calibration Error) |
| `MIN_COVERAGE` | 0.50 | >= (fraction of customers with a prediction) |
| `MIN_PR_AUC` | 0.20 | >= (harder bar under class imbalance) |
| `MAX_TRAIN_TEST_ROC_GAP` | 0.15 | <= (overfitting detection) |

### ML vs Baseline Checks
| Check | Threshold | Direction |
|---|---|---|
| `ML_BASELINE_TOLERANCE` | 0.05 | ML ROC-AUC must not lag baseline by more than this |
| `ML_BASELINE_PR_AUC_TOLERANCE` | 0.05 | ML PR-AUC lag tolerance |

### Baseline Quality Checks
| Check | Threshold | Direction |
|---|---|---|
| `BASELINE_MIN_ROC_AUC` | 0.65 | >= |
| `BASELINE_MIN_PR_AUC` | 0.40 | >= |
| `BASELINE_MIN_PRECISION` | 0.55 | >= |
| `BASELINE_MIN_RECALL` | 0.55 | >= |

### Distribution Shift
| Check | Threshold | Direction |
|---|---|---|
| `MAX_ADVERSARIAL_AUC` | 0.70 | <= (AUC of random forest distinguishing train vs test) |

## Adding a New Guardrail

1. Add the threshold constant to `src/config.py`
2. Write a `check_*` function returning the standard dict (use `_make_result` helper)
3. Add it to `run_all_guardrails`
4. Add corresponding tests in `tests/test_guardrails.py`

Tests must use **explicit numeric inputs** to verify pass/fail boundaries — not model fixtures.

## Metrics (`src/evaluation/metrics.py`)

`compute_metrics(y_true, y_prob)` returns dict with:
`roc_auc`, `pr_auc`, `brier_score`, `precision`, `recall`, `f1`, `churn_rate`

## Calibration (`src/evaluation/calibration.py`)

`compute_calibration(y_true, y_prob)` — returns calibration curve + ECE.

## Distribution Shift (`src/evaluation/distribution_shift.py`)

`compute_adversarial_auc(X_train, X_test)` — trains a random forest to distinguish train from test. AUC near 0.5 = no shift; AUC > 0.70 = covariate shift that threatens hold-out validity.
