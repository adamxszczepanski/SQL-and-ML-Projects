# Insurance Charges Prediction (Machine Learning)

Predicting individual medical insurance charges from demographic and health
attributes, using a full regression workflow: exploratory data analysis, a
leakage-free preprocessing pipeline, model comparison with cross-validation and
hyperparameter search, target transformation, and a detailed error analysis of
the final model.

## Problem

Given a patient's `age`, `sex`, `bmi`, `children`, `smoker` status and `region`,
the goal is to estimate their annual insurance `charges`. This is a supervised
regression task on a right-skewed target.

## Dataset

The classic `insurance.csv` dataset (1,338 rows, 7 columns). One duplicate row is
removed, leaving 1,337 records. There are no missing values.

| Column | Type | Description |
|--------|------|-------------|
| age | int | Age of the primary beneficiary |
| sex | categorical | female / male |
| bmi | float | Body mass index |
| children | int | Number of dependents |
| smoker | categorical | yes / no |
| region | categorical | northeast / northwest / southeast / southwest |
| charges | float | **Target**: annual medical charges |

## Workflow

1. **Data cleaning:** duplicate removal, missing-value and type checks.
2. **EDA:** distributions, skewness, bivariate relationships and outlier
   inspection. Two dominant drivers emerge, smoking and age.
3. **Preprocessing:** a `ColumnTransformer` with `OneHotEncoder(drop='first')`
   for categoricals and `StandardScaler` for numerics, fit only on the training
   data to avoid leakage.
4. **Modelling:** a Linear Regression baseline, a Polynomial-degree elbow study,
   then a cross-validated comparison of Linear Regression, ElasticNet, SVR,
   Random Forest and XGBoost via `GridSearchCV` and `RandomizedSearchCV`.
5. **Target transformation:** because `charges` is right-skewed
   (skewness of about 1.52), every model is re-run inside a
   `TransformedTargetRegressor` using `log1p` and `expm1`.
6. **Final evaluation:** the selected model is scored once on the held-out test
   set, followed by an error analysis by segment and a feature-importance study.

## Data split

A 70 / 15 / 15 split with a fixed random seed (`random_state=88`):

| Split | Rows |
|-------|------|
| Train | 935 |
| Validation | 201 |
| Test | 201 |

The test set is touched only once, for the final model.

## Key EDA findings

- `charges` is strongly right-skewed, and a `log1p` transform makes it far more
  symmetric.
- Smokers pay dramatically more. Smoking is by far the strongest single
  predictor (Pearson correlation of about 0.79 with the target).
- Charges rise with age for both smokers and non-smokers.
- BMI matters mostly in combination with smoking; on its own it is a weak driver.
- The highest-charge records are all obese smokers, which is plausible rather
  than an error, so no rows were removed.

## Model comparison

Validation RMSE (lower is better), with the log-transformed target:

| Model | Validation RMSE |
|-------|-----------------|
| **Random Forest (log target)** | **about 5,275** |
| XGBoost (log target) | about 5,387 |
| SVR (log target) | about 5,821 |
| Linear Regression | about 7,702 |
| ElasticNet | about 7,707 |

For reference, the plain Linear Regression baseline scored a validation RMSE of
about 6,651 against a mean charge of about 13,468, which points to underfitting
with structured (non-random) residuals.

## Final model

A **Random Forest Regressor wrapped in a `TransformedTargetRegressor`
(`log1p` / `expm1`)**. It gave the lowest validation RMSE and the smallest
train-validation gap, indicating the best generalization.

Held-out test performance:

| Metric | Value |
|--------|-------|
| RMSE | 4,611 |
| MAE | 2,132 |

The MAE of about $2,132 is the average absolute error. RMSE is higher because a
small number of high-cost cases produce large errors.

## Error analysis

- **By smoker group:** the model is well calibrated for smokers (about 6.6% mean
  percentage error) and less precise in relative terms for non-smokers
  (about 20%), although the absolute errors are similar.
- **By charge band:** errors concentrate in the $20k to $30k range, which the
  model tends to under-predict. These are mostly mid-cost cases that sit between
  the low non-smoker cluster and the high smoker cluster.
- **Feature importance** (permutation) confirms the EDA. `smoker` dominates,
  followed by `bmi` and `age`, while `children`, `sex` and `region` contribute
  almost nothing.

## Repository structure

```
InsuranceChargesPrediction_ML/
├── InsuranceChargePrediction_ML.ipynb   # Full analysis notebook
├── insurance.csv                        # Dataset
├── models/
│   └── random_forest_log_target_final.pkl   # Serialized final model
└── README.md
```

## How to run

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn xgboost matplotlib seaborn joblib jupyter

# 2. Launch the notebook
jupyter notebook InsuranceChargePrediction_ML.ipynb
```

Make sure `insurance.csv` sits next to the notebook. Running all cells reproduces
the EDA, model search, final evaluation and the saved model in `models/`.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, joblib.

## Key takeaways

- Careful, leakage-free preprocessing and a strict train/validation/test protocol
  matter as much as model choice.
- Matching the model to the data, here by log-transforming a skewed target, gave
  a larger gain than swapping algorithms.
- Segment-level error analysis reveals where a model fails (the $20k to $30k
  band), which an aggregate RMSE alone hides.
