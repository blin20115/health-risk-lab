# HealthRiskLab

HealthRiskLab is a diabetes risk model backtesting project using public health survey data from the CDC Diabetes Health Indicators dataset.

The goal is not just to train a classifier, but to compare models across:
- ROC-AUC
- precision and recall
- F1 score
- calibration
- threshold tradeoffs
- cross-validation stability

## Prediction Task

Given health, lifestyle, and demographic survey indicators, predict whether a respondent has diabetes or prediabetes.

Target:
- 0 = no diabetes
- 1 = prediabetes or diabetes

## Models

Planned models:
- Logistic Regression
- Random Forest
- XGBoost

## Project Structure

```text
src/
  data.py        # data loading and preprocessing
  train.py       # model training
  evaluate.py    # metrics and validation
  plots.py       # plotting utilities

notebooks/
  01_explore_data.ipynb

results/
  figures/

reports/
