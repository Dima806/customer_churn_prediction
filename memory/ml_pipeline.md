# ML Pipeline: Training, Tuning, and Prediction

## Full Training Flow (`make train` → `src/models/train.py`)

1. Load `data/orders.csv` and `data/customers.csv`
2. Clean with `clean_orders()` / `clean_customers()` from `src/preprocessing/clean.py`
3. `compute_churn_target(orders, customers)` → `(target_df, feature_end_date, t_max)`
4. `build_feature_matrix(orders, customers, feature_end_date)` → feature DataFrame
5. `time_based_split(features, target_df, customers)` → train/test split
6. `compute_adversarial_auc(X_train, X_test)` → distribution shift check
7. `build_augmented_training_set(...)` → extra training rows (train-only, no leakage)
8. Load best params: `load_best_params()` → reads `artifacts/best_params.json` if present
9. `train_xgboost(X_train, y_train, params)` → fitted XGBClassifier
10. Evaluate on test set + train set (overfitting check)
11. `evaluate_baseline(test_features, test_target)` — same test set for fair comparison
12. `run_all_guardrails(...)` → list of pass/fail checks
13. `save_model(model)` → `artifacts/xgb_churn_model.pkl`
14. `save_evaluation_artifacts(...)` → JSON files consumed by dashboard

## Artifact Files Written

All under `artifacts/`:
- `xgb_churn_model.pkl` — pickled XGBClassifier
- `eval_metrics.json` — ML + baseline metrics
- `calibration.json` — calibration curve data
- `roc_curve.json` — FPR/TPR arrays
- `feature_importance.json` — sorted feature importances
- `feature_cols.json` — ordered list of feature names
- `guardrails.json` — full guardrail results
- `distribution_shift.json` — adversarial AUC result
- `category_summary.json` — actual vs predicted churn by segment/channel/country
- `customer_stats.json` — customer counts by category
- `best_params.json` — tuned hyperparameters (written by `make tune`)

## XGBoost Default Params

From `src/config.py`:
```python
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,        # shallow = better generalisation
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5, # prevents tiny splits
    "gamma": 0.1,          # min gain for a split
    "reg_alpha": 0.1,      # L1 regularisation
    "eval_metric": "auc",
    "random_state": 42,
}
```

`scale_pos_weight` is deliberately left at 1.0 — **do not add automatic scale_pos_weight computation**.

## Tuning Search Space (`src/models/tune.py`)

```python
_PARAM_SPACE = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05, 0.10],
    "subsample": [0.70, 0.80, 0.90],
    "colsample_bytree": [0.70, 0.80, 0.90],
    "min_child_weight": [3, 5, 10],
    "gamma": [0.0, 0.1, 0.3],
    "reg_alpha": [0.0, 0.1, 0.5],
}
```

Random search, 30 trials, 3 walk-forward folds by default. `best_params.json` must include all three regularisation keys (`gamma`, `reg_alpha`, `min_child_weight`) or a warning is logged.

## Rule-Based Baseline (`src/baseline/rule_based.py`)

Rule: `if recency > 90 OR orders_last_60d == 0 → churned`

Soft score: normalised recency, boosted to min 0.70 when 60-day window is empty. Used for ROC-AUC / PR-AUC comparison with the ML model. Baseline is always evaluated on the **same test set** as the ML model.

## Batch Prediction (`make predict` → `src/prediction/batch_predict.py`)

Scores all customers using the saved model. Output: `predictions/churn_predictions.csv`.
The dashboard reads exclusively from `artifacts/` and `predictions/`.
