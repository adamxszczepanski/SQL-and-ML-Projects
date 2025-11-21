# Insurance Charges Prediction

## Project Overview

This project builds a regression model to predict **medical insurance charges** based on demographic and lifestyle features. The goal is to show an end-to-end machine learning workflow and extract practical insights that could support pricing and risk assessment in an insurance context.

## Dataset

Each row represents one insured person with:
- `age`, `bmi`, `children`
- `sex`, `smoker`, `region`
- `charges` – target variable (total medical insurance cost)

Basic cleaning, exploratory analysis and preprocessing (scaling numeric features, encoding categorical variables) are applied before modeling.

## Modeling

The workflow is implemented in Python (scikit-learn) using Pipelines and, where needed, `ColumnTransformer`. Models compared:

- **Linear Regression** – baseline
- **Polynomial Regression** – captures non-linear relationships
- **Support Vector Regression (SVR)** – non-linear model (e.g. RBF kernel)

Hyperparameters are tuned with **GridSearchCV** on a train/validation split. A separate test set is used for final evaluation.

**Metric:**  
- Root Mean Squared Error (**RMSE**) on validation and test sets.

## Key Insights

- **Smoker status** and **BMI** are among the strongest drivers of higher charges.
- **Age** also has a clear positive relationship with expected costs.
- Non-linear models (Polynomial Regression, SVR) can achieve lower RMSE than a simple linear model, at the cost of interpretability.
- The analysis illustrates how combining preprocessing, pipelines and grid search leads to a clean and reproducible modeling process.
