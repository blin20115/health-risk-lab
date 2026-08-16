# HealthRiskLab

HealthRiskLab is a diabetes risk model evaluation project using the [CDC Diabetes Health Indicators dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).

The goal is not just to train a classifier, but to evaluate health-risk prediction models carefully under class imbalance.

## Project Goal

This project predicts whether a survey respondent has diabetes or prediabetes based on health, lifestyle, and demographic indicators.

Target:

- `0` = no diabetes
- `1` = diabetes or prediabetes

The project focuses on:

- baseline model comparison
- precision/recall tradeoffs
- threshold analysis
- confusion matrix analysis
- calibration analysis

## Why This Project Matters

The dataset is imbalanced:

| Class | Meaning | Percentage |
|---|---|---:|
| 0 | No diabetes | 86.07% |
| 1 | Diabetes or prediabetes | 13.93% |

Because only about 14% of examples are positive, accuracy can be misleading.

A model that predicts "no diabetes" for everyone gets about 86% accuracy, but catches 0% of diabetes/prediabetes cases.

So this project evaluates models using metrics beyond accuracy, including precision, recall, F1, ROC-AUC, average precision, calibration, and threshold behavior.

## Dataset

The dataset contains:

- 253,680 rows
- 21 features
- no missing values

Example features include:

- high blood pressure
- high cholesterol
- BMI
- smoking history
- physical activity
- general health
- difficulty walking
- age category
- education
- income

## Models Compared

The current project compares:

1. Dummy classifier
   Always predicts the most common class.

2. Logistic regression
   Simple linear baseline model.

3. Random forest
   Tree-based model that can capture more complex feature interactions.

## Baseline Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision | Positive Rate Predicted |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy Most Frequent | 0.8607 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1393 | 0.0000 |
| Logistic Regression | 0.7315 | 0.3107 | 0.7611 | 0.4413 | 0.8196 | 0.3926 | 0.3412 |
| Random Forest | 0.7171 | 0.3019 | 0.7850 | 0.4361 | 0.8198 | 0.4099 | 0.3623 |

## Key Findings

The dummy model has high accuracy because most rows are negative, but it is not useful because it catches no diabetes/prediabetes cases.

Logistic regression is a strong simple baseline. It catches about 76% of positive cases and has an ROC-AUC of 0.8196.

Random forest has slightly higher recall than logistic regression, but lower precision. It predicts more people as positive overall, which means it catches more positives but also creates more false positives.

The ROC-AUC values for logistic regression and random forest are almost identical, so random forest is not clearly better despite being more complex.

## Threshold Analysis

The model outputs probabilities, and a threshold turns those probabilities into final predictions.

For example:

- threshold = 0.5 means predict positive if probability >= 0.5
- threshold = 0.7 means the model must be more confident before predicting positive

For logistic regression, increasing the threshold caused:

- precision to increase
- recall to decrease
- fewer people to be predicted positive

The highest F1 score occurred around threshold 0.6.

However, in a health-risk setting, the best threshold may not always be the one with the highest F1 score. If missing diabetes/prediabetes cases is worse than false positives, a lower threshold may be preferred.

![Threshold Tradeoff](results/figures/threshold_tradeoff.png)

## Confusion Matrix Analysis

Confusion matrices show the actual counts behind the model predictions.

For this project:

- false positive = model predicts diabetes/prediabetes, but person is actually negative
- false negative = model predicts no diabetes, but person is actually positive

False negatives are especially important in health-risk prediction because they represent missed at-risk cases.

## Calibration Analysis

Calibration checks whether predicted probabilities are trustworthy.

For example, if a model gives a group of people around 70% predicted risk, then ideally about 70% of that group should actually be positive.

Brier scores:

| Model | Brier Score |
|---|---:|
| Logistic Regression | 0.1776 |
| Random Forest | 0.1784 |

Lower Brier score is better.

Logistic regression had a slightly lower Brier score than random forest, but the difference was small.

The calibration curves did not closely follow the perfect calibration diagonal, suggesting that the predicted probabilities should not be interpreted as exact risk estimates without further calibration.

![Calibration Curves](results/figures/calibration_curves.png)

## Calibrated Classifiers

I also tested sigmoid-calibrated versions of logistic regression and random forest.

Calibration improved the Brier scores substantially:

| Model | Calibration | Brier Score | Recall at 0.5 Threshold | Positive Rate Predicted |
|---|---|---:|---:|---:|
| Logistic Regression | Uncalibrated | 0.1776 | 0.7611 | 0.3412 |
| Logistic Regression | Sigmoid Calibrated | 0.1001 | 0.1565 | 0.0423 |
| Random Forest | Uncalibrated | 0.1784 | 0.7850 | 0.3623 |
| Random Forest | Sigmoid Calibrated | 0.0986 | 0.1317 | 0.0333 |

Lower Brier score means better probability calibration, so sigmoid calibration made the predicted probabilities more trustworthy.

However, at the default threshold of 0.5, the calibrated models predicted very few positive cases and recall dropped sharply. This shows that calibration and threshold selection are separate steps.

Because the dataset is imbalanced, a calibrated probability below 0.5 can still be meaningful. I tested lower thresholds for the calibrated models and found that thresholds around 0.15 to 0.25 restored a more useful precision-recall tradeoff.

For example:

| Model | Threshold | Precision | Recall | F1 | Positive Rate Predicted |
|---|---:|---:|---:|---:|---:|
| Calibrated Logistic Regression | 0.15 | 0.3185 | 0.7414 | 0.4456 | 0.3243 |
| Calibrated Logistic Regression | 0.20 | 0.3602 | 0.6306 | 0.4585 | 0.2439 |
| Calibrated Random Forest | 0.15 | 0.3127 | 0.7520 | 0.4418 | 0.3350 |
| Calibrated Random Forest | 0.25 | 0.3784 | 0.5891 | 0.4608 | 0.2169 |

This reinforced one of the main project lessons: calibrated probabilities are useful, but the final threshold still needs to be chosen based on the real-world goal.

![Calibrated Model Curves](results/figures/calibrated_model_curves.png)

![Calibrated Logistic Regression Thresholds](results/figures/calibrated_threshold_logistic_regression.png)

![Calibrated Random Forest Thresholds](results/figures/calibrated_threshold_random_forest.png)

## Feature Importance

Feature importance helps explain which variables the models used most when predicting diabetes/prediabetes risk.

Both logistic regression and random forest identified similar top predictors, including:

- general health
- BMI
- age
- high blood pressure
- high cholesterol

Logistic regression coefficients show both direction and strength. Random forest importance shows how useful each feature was for splitting the data, but does not directly show direction.

These results should be interpreted as predictive associations, not causal claims.

![Logistic Regression Feature Importance](results/figures/logistic_feature_importance.png)

![Random Forest Feature Importance](results/figures/random_forest_feature_importance.png)

## Cross-Validation Stability

To check whether the results were stable, I also ran 3-fold stratified cross-validation.

Cross-validation showed that both logistic regression and random forest had low standard deviation across folds, suggesting that performance was not just due to one lucky train/test split.

| Model | ROC-AUC Mean | ROC-AUC Std | Recall Mean | Recall Std | Average Precision Mean |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8224 | 0.0026 | 0.7653 | 0.0082 | 0.4030 |
| Random Forest | 0.8234 | 0.0016 | 0.7868 | 0.0026 | 0.4208 |

Random forest had slightly higher recall, ROC-AUC, and average precision, while logistic regression had slightly better precision and F1. The differences were small, so logistic regression remains a strong simple baseline.

## Fairness / Subgroup Analysis

I also evaluated the logistic regression model across subgroups for sex, income, and education.

This analysis checks whether the model makes different types of errors for different groups. In a health-risk setting, false negatives are especially important because they represent missed diabetes/prediabetes cases.

The model showed relatively similar recall across sex groups:

| Group | Recall | False Negative Rate |
|---|---:|---:|
| sex = 0 | 0.7522 | 0.2478 |
| sex = 1 | 0.7707 | 0.2293 |

The differences were larger across income and education groups. Lower income and lower education groups tended to have higher recall, but also higher false positive rates. Higher income and higher education groups tended to have lower recall and higher false negative rates.

This suggests that average model metrics hide subgroup-level differences, so fairness analysis would be important before any real-world use.

![Fairness by Income](results/figures/fairness_income_recall_fnr.png)

![Fairness by Education](results/figures/fairness_education_recall_fnr.png)

## Final Conclusion

This project showed that accuracy alone is not a reliable metric for diabetes/prediabetes risk prediction because the dataset is highly imbalanced.

The dummy classifier achieved high accuracy by predicting the majority class, but it had zero recall and failed to identify any positive cases. Logistic regression and random forest both performed much better by detecting a meaningful share of diabetes/prediabetes cases.

Random forest had slightly higher recall, ROC-AUC, and average precision, meaning it caught more positive cases and ranked examples slightly better. However, logistic regression had slightly better precision and F1, was easier to interpret, and had slightly better calibration.

Overall, logistic regression remained a strong baseline because it was simple, interpretable, and performed similarly to the more complex random forest model.

The most important lesson from this project is that model evaluation depends on the real-world goal. In a health-risk setting, threshold choice matters because lowering the threshold catches more at-risk people but also increases false positives.

## Limitations

This project uses survey data, not direct clinical measurements.

The model should not be used for real medical diagnosis, treatment decisions, or clinical decision-making.

The target label represents diabetes/prediabetes status in the dataset, but it may not perfectly reflect clinical diagnosis.

The model was trained and evaluated on one dataset source, so the results may not generalize to other populations.

Some features, such as income, education, and sex, may raise fairness concerns if used in real-world decision-making.

Feature importance should be interpreted as predictive association, not causation.

The calibration analysis suggests that predicted probabilities should not be treated as exact medical risk estimates without further calibration.

## Project Structure

```text
health-risk-lab/
  README.md
  requirements.txt

  src/
    __init__.py
    data.py
    evaluate.py
    train.py
    threshold_analysis.py
    confusion_analysis.py
    calibration_analysis.py
    calibrated_models.py
    calibrated_threshold_analysis.py
    feature_importance.py
    cross_validation.py
    fairness_analysis.py

  notebooks/
    01_explore_data.ipynb

  reports/
    model_findings.md
    model_card.md
    threshold_metrics.csv
    calibration_metrics.csv
    calibrated_model_metrics.csv
    calibrated_threshold_metrics.csv
    feature_importance.csv
    cross_validation_metrics.csv
    fairness_metrics.csv

  results/
    baseline_metrics.json
    figures/
      threshold_tradeoff.png
      calibration_curves.png
      calibrated_model_curves.png
      calibrated_threshold_logistic_regression.png
      calibrated_threshold_random_forest.png
      logistic_feature_importance.png
      random_forest_feature_importance.png
      confusion_matrix_threshold_0.5.png
      confusion_matrix_threshold_0.6.png
      fairness_income_recall_fnr.png
      fairness_education_recall_fnr.png
      fairness_sex_recall_fnr.png
