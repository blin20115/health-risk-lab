# Model Findings

## Project Goal

This project predicts diabetes/prediabetes risk using the CDC Diabetes Health Indicators dataset.

The goal is not just to train a model, but to evaluate models carefully under class imbalance.

The positive class is:

- `1` = diabetes or prediabetes

The negative class is:

- `0` = no diabetes

## Dataset Imbalance

The dataset has 253,680 rows and 21 features.

The target is imbalanced:

| Class | Meaning | Percentage |
|---|---|---:|
| 0 | No diabetes | 86.07% |
| 1 | Diabetes or prediabetes | 13.93% |

Because only about 14% of examples are positive, accuracy can be misleading.

A model that predicts "no diabetes" for everyone would get about 86% accuracy, but it would catch 0% of diabetes/prediabetes cases.

## Baseline Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision | Positive Rate Predicted |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy Most Frequent | 0.8607 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1393 | 0.0000 |
| Logistic Regression | 0.7315 | 0.3107 | 0.7611 | 0.4413 | 0.8196 | 0.3926 | 0.3412 |
| Random Forest | 0.7171 | 0.3019 | 0.7850 | 0.4361 | 0.8198 | 0.4099 | 0.3623 |

## Baseline Interpretation

The dummy model has high accuracy because the dataset is mostly negative cases. However, it has zero recall, meaning it does not catch any diabetes/prediabetes cases.

Logistic regression is much more useful than the dummy model because it catches about 76% of positive cases.

Random forest has slightly higher recall than logistic regression, but lower precision. It also predicts more people as positive overall.

This suggests that random forest is slightly more aggressive about flagging diabetes/prediabetes risk.

## Logistic Regression vs Random Forest

Random forest had about the same ROC-AUC as logistic regression:

- Logistic regression ROC-AUC: 0.8196
- Random forest ROC-AUC: 0.8198

The difference is very small, so I would not claim that random forest is clearly better.

Random forest improved recall:

- Logistic regression recall: 0.7611
- Random forest recall: 0.7850

But random forest had worse precision:

- Logistic regression precision: 0.3107
- Random forest precision: 0.3019

This shows a common precision-recall tradeoff:

- Predicting more positives can increase recall.
- Predicting more positives can also create more false positives and lower precision.

## Threshold Analysis

The logistic regression model outputs probabilities. A threshold converts those probabilities into final 0/1 predictions.

For example:

- threshold = 0.5 means predict positive if probability >= 0.5
- threshold = 0.7 means the model has to be more confident before predicting positive

The threshold analysis showed:

| Threshold | Precision | Recall | F1 | Positive Rate Predicted |
|---:|---:|---:|---:|---:|
| 0.1 | 0.1669 | 0.9939 | 0.2858 | 0.8298 |
| 0.2 | 0.2031 | 0.9656 | 0.3356 | 0.6625 |
| 0.3 | 0.2384 | 0.9174 | 0.3785 | 0.5361 |
| 0.4 | 0.2740 | 0.8536 | 0.4148 | 0.4341 |
| 0.5 | 0.3107 | 0.7611 | 0.4413 | 0.3412 |
| 0.6 | 0.3545 | 0.6492 | 0.4586 | 0.2551 |
| 0.7 | 0.4004 | 0.4984 | 0.4440 | 0.1734 |
| 0.8 | 0.4557 | 0.3091 | 0.3683 | 0.0945 |
| 0.9 | 0.5283 | 0.1096 | 0.1816 | 0.0289 |

As the threshold increases:

- precision increases
- recall decreases
- the model predicts fewer people as positive

The highest F1 score occurred around threshold 0.6.

However, in a health-risk setting, the best threshold may not always be the one with the highest F1 score. If missing diabetes/prediabetes cases is considered worse than false positives, a lower threshold may be preferred.

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

The calibration curves did not closely follow the perfect calibration diagonal, which suggests that the predicted probabilities should not be interpreted as exact risk estimates without further calibration.

## Main Takeaways

1. Accuracy is misleading for this dataset because the target is imbalanced.

2. The dummy model gets high accuracy but is not useful because it catches no positive cases.

3. Logistic regression is a strong simple baseline with good recall and ROC-AUC.

4. Random forest catches slightly more positive cases but also creates more false positives.

5. Threshold choice strongly affects precision and recall.

6. Calibration analysis suggests that model probabilities should be treated carefully.

## Limitations

This project uses survey data, not clinical diagnosis data.

The model should not be used for real medical decision-making.

Some features, such as income, education, and sex, may raise fairness concerns and should be analyzed carefully before any real-world use.

The current analysis compares only baseline models. Future work could include:

- cross-validation stability analysis
- feature importance
- calibrated models
- fairness analysis across demographic groups
- additional models such as gradient boosting
