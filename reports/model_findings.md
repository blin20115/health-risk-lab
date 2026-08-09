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

## Feature Importance Analysis

Feature importance helps explain which variables the models used most when predicting diabetes/prediabetes risk.

This project compares feature importance from:

- logistic regression coefficients
- random forest feature importances

Logistic regression coefficients provide both strength and direction. A positive coefficient means higher feature values push the model toward predicting diabetes/prediabetes. A negative coefficient means higher feature values push the model toward predicting no diabetes.

Random forest feature importance measures how useful each feature was for splitting the data across the trees. It shows strength, but not direction.

### Top Logistic Regression Features

| Feature | Coefficient | Absolute Coefficient |
|---|---:|---:|
| genhlth | 0.6188 | 0.6188 |
| bmi | 0.4906 | 0.4906 |
| age | 0.4574 | 0.4574 |
| highbp | 0.3643 | 0.3643 |
| highchol | 0.2873 | 0.2873 |
| cholcheck | 0.2477 | 0.2477 |
| hvyalcoholconsump | -0.1730 | 0.1730 |
| sex | 0.1392 | 0.1392 |
| income | -0.1193 | 0.1193 |
| heartdiseaseorattack | 0.0730 | 0.0730 |

### Top Random Forest Features

| Feature | Random Forest Importance |
|---|---:|
| highbp | 0.2457 |
| genhlth | 0.2445 |
| bmi | 0.1317 |
| highchol | 0.1092 |
| age | 0.0952 |
| diffwalk | 0.0567 |
| heartdiseaseorattack | 0.0295 |
| income | 0.0238 |
| physhlth | 0.0225 |
| physactivity | 0.0086 |

### Interpretation

Both models identified general health, BMI, age, high blood pressure, and high cholesterol as important predictors.

This agreement is useful because logistic regression and random forest are different types of models. Logistic regression is simpler and more linear, while random forest can capture more complex patterns.

However, these results should not be interpreted causally. The model is identifying features that are useful for prediction, not proving that those features cause diabetes/prediabetes.

## Cross-Validation Stability Analysis

Cross-validation evaluates models across multiple train/validation splits.

This helps answer whether the model performance is stable or whether the earlier train/test result may have been lucky.

In this project, I used 3-fold stratified cross-validation. Stratification keeps the diabetes/prediabetes class ratio similar across folds, which matters because the dataset is imbalanced.

| Model | Metric | Mean | Std |
|---|---|---:|---:|
| Logistic Regression | Accuracy | 0.7314 | 0.0007 |
| Logistic Regression | Precision | 0.3113 | 0.0009 |
| Logistic Regression | Recall | 0.7653 | 0.0082 |
| Logistic Regression | F1 | 0.4426 | 0.0022 |
| Logistic Regression | ROC-AUC | 0.8224 | 0.0026 |
| Logistic Regression | Average Precision | 0.4030 | 0.0022 |
| Random Forest | Accuracy | 0.7191 | 0.0017 |
| Random Forest | Precision | 0.3038 | 0.0013 |
| Random Forest | Recall | 0.7868 | 0.0026 |
| Random Forest | F1 | 0.4384 | 0.0013 |
| Random Forest | ROC-AUC | 0.8234 | 0.0016 |
| Random Forest | Average Precision | 0.4208 | 0.0019 |

### Interpretation

Both models were stable across folds because the standard deviations were small.

Random forest had slightly higher recall, ROC-AUC, and average precision. This means it caught more positive diabetes/prediabetes cases and had slightly better ranking performance.

Logistic regression had slightly higher precision and F1. This means it had a slightly better balance between precision and recall at the default threshold.

Overall, random forest was not clearly better despite being more complex. Logistic regression remains a strong baseline because it is simpler, more interpretable, and performs similarly across validation folds.

## Main Takeaways

1. Accuracy is misleading for this dataset because the target is imbalanced.

2. The dummy model gets high accuracy but is not useful because it catches no positive cases.

3. Logistic regression is a strong simple baseline with good recall and ROC-AUC.

4. Random forest catches slightly more positive cases but also creates more false positives.

5. Threshold choice strongly affects precision and recall.

6. Calibration analysis suggests that model probabilities should be treated carefully.

## Final Conclusion

The main finding from this project is that diabetes/prediabetes risk prediction should be evaluated with more than accuracy.

Because only about 14% of the dataset belongs to the positive class, the dummy model achieved high accuracy by always predicting no diabetes. However, it had zero recall and was not useful for identifying diabetes/prediabetes cases.

Logistic regression and random forest both performed much better than the dummy baseline. Random forest had slightly better recall, ROC-AUC, and average precision, while logistic regression had slightly better precision, F1, interpretability, and calibration.

The differences between logistic regression and random forest were small, especially in cross-validation. This suggests that the simpler logistic regression model is a strong baseline for this dataset.

The threshold analysis showed that the model's behavior depends heavily on the chosen decision threshold. Lower thresholds catch more positive cases but create more false positives, while higher thresholds improve precision but miss more positive cases.

For a health-risk setting, the best threshold should depend on the relative cost of false negatives and false positives, not just the default threshold or highest accuracy.

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
