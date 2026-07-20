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
        This can be misleading here because about 86% of rows are negative.

    Precision:
        Of the people the model predicted as positive, how many were actually positive?
        High precision means fewer false alarms.

    Recall:
        Of the people who were actually positive, how many did the model catch?
        High recall means fewer missed diabetes/prediabetes cases.

    F1:
        A balance between precision and recall.

    ROC-AUC:
        Measures how well the model ranks positive cases above negative cases.
        0.5 = random guessing.
        1.0 = perfect ranking.

    Average precision:
        Measures precision-recall performance across thresholds.
        This is useful for imbalanced datasets.

    Positive rate predicted:
        The fraction of examples the model predicts as positive.
        This tells us how aggressive the model is.
"""

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


def evaluate_binary_classifier(y_true, y_pred, y_proba) -> Dict[str, float]:
    """
    Evaluate a binary classifier.

    Parameters:
        y_true:
            The actual labels.

        y_pred:
            The model's final 0/1 predictions.

        y_proba:
            The model's predicted probability for the positive class.

    Why:
        For imbalanced health data, accuracy alone is not enough.
        We also need precision, recall, F1, ROC-AUC, and average precision.
    """
    return {
        # Overall fraction of correct predictions.
        "accuracy": accuracy_score(y_true, y_pred),

        # Of predicted positives, how many were truly positive?
        "precision": precision_score(y_true, y_pred, zero_division=0),

        # Of actual positives, how many did the model catch?
        "recall": recall_score(y_true, y_pred, zero_division=0),

        # Balance between precision and recall.
        "f1": f1_score(y_true, y_pred, zero_division=0),

        # Ranking quality across all possible thresholds.
        "roc_auc": roc_auc_score(y_true, y_proba),

        # Precision-recall quality across thresholds.
        "average_precision": average_precision_score(y_true, y_proba),

        # Fraction of rows predicted as positive.
        "positive_rate_predicted": float(np.mean(y_pred)),
    }
