# SQL and Machine Learning Projects

A portfolio of hands on projects in data analysis and data science, built while
moving from applied mathematics into practical Data Science work.

## About me

I am an applied mathematics graduate who has always been drawn to the real world
uses of maths. I currently work as a data analyst, and I am steadily building my
Data Science skills through end to end projects. My focus is on doing things
properly: clean data handling, honest validation, and results I can explain
rather than just a number on a leaderboard.

This README brings those projects together in one place. Each project links to
its own code and write up.

## Projects

| Project | Type | Summary | Tech |
|---------|------|---------|------|
| [Insurance Charges Prediction](./InsuranceChargesPrediction_ML) | Machine Learning | Predicting medical insurance charges from demographic and health data. Full EDA, a leakage free preprocessing pipeline, 5 models compared with cross validation, a log transformed target, and a segment level error analysis. Final model: Random Forest on a log target, test MAE about 2,132. | Python, scikit-learn, XGBoost, pandas, matplotlib, seaborn |
| [Grocery Sales SQL Analysis](https://github.com/adamxszczepanski/Kaggle_SQLProject_GrocerySales) | SQL | Business analysis of grocery retail sales: top products and categories, monthly sales trends, and the most valuable customers. Uses joins across fact and dimension tables, CTEs and window functions to turn raw transactions into actionable insight. | SQL |

## Featured project: Insurance Charges Prediction

A complete regression workflow on the classic insurance dataset:

- Exploratory analysis that surfaces the two dominant cost drivers, smoking and
  age.
- A `ColumnTransformer` pipeline fit only on the training split, so no
  information leaks into validation or test.
- A cross validated comparison of Linear Regression, ElasticNet, SVR, Random
  Forest and XGBoost, then a rerun with a log transformed target to handle the
  skewed distribution.
- A held out test set scored once, plus an error analysis by smoker group and by
  charge band, and feature importance.

Full write up and a PDF report are in the
[project folder](./InsuranceChargesPrediction_ML).

## Featured project: Grocery Sales SQL Analysis

A SQL analysis of grocery retail transactions from a
[Kaggle dataset](https://www.kaggle.com/datasets/andrexibiza/grocery-sales-dataset/data),
framed around questions a store manager would actually ask:

- Which products and categories drive the most revenue, and how concentrated is
  that revenue.
- How sales evolve month over month.
- Who the highest value customers are and where promotions could pay off.

The queries join fact and dimension tables, aggregate with `GROUP BY` and
`HAVING`, structure logic with CTEs, and use window functions such as
`ROW_NUMBER`, `RANK` and running totals for ranking and trend analysis. 

Code and full write up in the
[project repository](https://github.com/adamxszczepanski/Kaggle_SQLProject_GrocerySales).

## Tech and tools

- **Languages:** Python, SQL
- **Data and ML:** pandas, NumPy, scikit-learn, XGBoost
- **Visualization:** matplotlib, seaborn, Power BI
- **Workflow:** Jupyter, Git, cross validation and proper train / validation /
  test protocols

## What is next

I am looking for a larger and messier dataset for the next project, one that
needs real data cleaning and feature work, possibly from a different domain, to
push beyond what a small clean dataset can teach.

## Contact

- Email: adamxszczepanski@gmail.com
- LinkedIn: _add your profile link here_

Feedback and suggestions are always welcome.
