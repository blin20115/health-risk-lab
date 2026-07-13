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
