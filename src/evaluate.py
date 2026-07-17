from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

"""
Metric reference for this project

Positive class:
    1 = diabetes or prediabetes

Negative class:
    0 = no diabetes

Confusion matrix terms:
    True Positive:
        Model predicts diabetes/prediabetes, and the person actually has diabetes/prediabetes.

    False Positive:
        Model predicts diabetes/prediabetes, but the person does not have diabetes/prediabetes.

    True Negative:
        Model predicts no diabetes, and the person actually does not have diabetes/prediabetes.

    False Negative:
        Model predicts no diabetes, but the person actually has diabetes/prediabetes.

Metrics:
    Accuracy:
        How often the model is correct overall.
        This can be misleading for this dataset because only about 14% of rows are positive.

    Precision:
        Of the people the model predicted as positive, how many were actually positive?
        High precision means fewer false alarms.

    Recall:
        Of the people who were actually positive, how many did the model catch?
        High recall means fewer missed diabetes/prediabetes cases.

    F1 score:
        A balance between precision and recall.
        Useful when both false positives and false negatives matter.

    ROC-AUC:
        Measures how well the model ranks positive cases above negative cases.
        0.5 means random guessing, and 1.0 means perfect ranking.
        It uses predicted probabilities, not just final 0/1 predictions.

    Average precision:
        Measures precision-recall performance across thresholds.
        Especially useful for imbalanced datasets where the positive class is rare.

    Positive rate predicted:
        The percentage of rows the model predicts as positive.
        This tells us how aggressive the model is about flagging diabetes risk.
"""


def evaluate_binary_classifier(y_true, y_pred, y_proba) -> Dict[str, float]:
    """
    Evaluate a binary classifier using metrics that are useful for imbalanced data.

    Parameters
    ----------
    y_true:
        True binary labels.
    y_pred:
        Predicted binary labels.
    y_proba:
        Predicted probabilities for the positive class.

    Returns
    -------
    Dictionary of evaluation metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "positive_rate_predicted": float(np.mean(y_pred)),
    }
