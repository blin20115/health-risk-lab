# Model Card: HealthRiskLab Diabetes Risk Models

## Model Overview

HealthRiskLab evaluates machine learning models for predicting diabetes/prediabetes risk using the CDC Diabetes Health Indicators dataset.

The project compares baseline classifiers and analyzes their behavior under class imbalance, including threshold tradeoffs, calibration, confusion matrices, feature importance, and cross-validation stability.

## Prediction Task

The model predicts a binary target:

- `0` = no diabetes
- `1` = diabetes or prediabetes

The input features are survey-based health, lifestyle, and demographic indicators such as BMI, high blood pressure, high cholesterol, physical activity, general health, age category, education, and income.

## Intended Use

This project is intended for:

- learning machine learning evaluation techniques
- studying model behavior on imbalanced health data
- comparing baseline models
- understanding precision-recall tradeoffs
- practicing responsible ML documentation

## Not Intended Use

This model should not be used for:

- real medical diagnosis
- clinical decision-making
- deciding treatment or care
- replacing advice from a healthcare professional
- making decisions about individuals

The dataset is based on survey responses, not direct clinical diagnosis data.

## Dataset

Dataset: [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

The dataset contains:

- 253,680 rows
- 21 features
- no missing values

Target distribution:

| Class | Meaning | Percentage |
|---|---|---:|
| 0 | No diabetes | 86.07% |
| 1 | Diabetes or prediabetes | 13.93% |

The dataset is imbalanced, so accuracy alone is not enough to evaluate model performance.

## Models Evaluated

This project currently evaluates:

1. Dummy classifier
   Always predicts the most common class.

2. Logistic regression
   A simple linear baseline model.

3. Random forest
   A tree-based model that can capture more complex feature interactions.

## Evaluation Metrics

The project evaluates models using:

- accuracy
- precision
- recall
- F1 score
- ROC-AUC
- average precision
- Brier score
- confusion matrices
- threshold analysis
- cross-validation mean and standard deviation

Because the positive class is relatively rare, precision, recall, F1, average precision, and threshold behavior are more informative than accuracy alone.

## Main Results

The dummy model achieved high accuracy because most examples are negative, but it had zero recall and did not identify any positive diabetes/prediabetes cases.

Logistic regression and random forest both performed much better than the dummy baseline.

Logistic regression had:

- ROC-AUC: 0.8196
- recall: 0.7611
- precision: 0.3107
- F1: 0.4413

Random forest had:

- ROC-AUC: 0.8198
- recall: 0.7850
- precision: 0.3019
- F1: 0.4361

Random forest caught slightly more positive cases, but also predicted more people as positive and had lower precision. Logistic regression remained a strong baseline because it was simpler, more interpretable, and performed similarly.

## Threshold Behavior

The logistic regression model outputs probabilities, and a threshold converts those probabilities into final predictions.

Lower thresholds increased recall but reduced precision. Higher thresholds increased precision but reduced recall.

The highest F1 score occurred around threshold `0.6`, but in a health-risk setting, the best threshold depends on the relative cost of false negatives versus false positives.

## Calibration

Calibration analysis showed that logistic regression and random forest had similar Brier scores:

| Model | Brier Score |
|---|---:|
| Logistic Regression | 0.1776 |
| Random Forest | 0.1784 |

The calibration curves did not closely follow the perfect calibration diagonal, so predicted probabilities should not be interpreted as exact risk estimates without further calibration.

## Feature Importance

Both logistic regression and random forest identified similar important predictors, including:

- general health
- BMI
- age
- high blood pressure
- high cholesterol

These features should be interpreted as predictive associations, not causal factors.

## Limitations

This project has several limitations:

1. The dataset is based on survey responses, not clinical measurements.

2. The target represents diabetes/prediabetes status from the dataset and may not perfectly reflect clinical diagnosis.

3. The model was evaluated on the same dataset source it was trained from, so performance may not generalize to other populations.

4. Some features, such as income, education, and sex, may raise fairness concerns.

5. The model does not explain causal relationships.

6. Calibration results suggest that predicted probabilities are not exact risk estimates.

7. The current project only evaluates baseline models.

## Fairness and Ethical Considerations

The dataset includes demographic and socioeconomic variables such as sex, education, and income.

These variables may improve prediction performance, but they can also introduce fairness concerns if used in real-world decision-making.

Before any real-world use, the model would need further analysis across demographic groups, including subgroup-level precision, recall, false positive rates, and false negative rates.

## Future Work

Potential next steps include:

- fairness analysis across demographic groups
- calibrated classifiers
- additional models such as gradient boosting
- bootstrapped confidence intervals
- subgroup-level threshold analysis
- external validation on another dataset

## Final Note

This project is for educational and model-evaluation purposes only. It should not be used as a medical device or diagnostic tool.
