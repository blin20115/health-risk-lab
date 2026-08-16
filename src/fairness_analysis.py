"""
Fairness / subgroup analysis

This script checks whether model performance differs across subgroups.

Why this matters:
    In healthcare-related ML, average performance is not enough.
    A model can perform well overall but worse for certain demographic
    or socioeconomic groups.

Subgroups analyzed:
    - sex
    - income
    - education

Important:
    This analysis does not prove discrimination or fairness by itself.
    It is a first diagnostic step to see whether error rates differ by group.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_diabetes_data


REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("results") / "figures"

REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def build_model():
    """
    Build the logistic regression model.

    Why logistic regression:
        It performed similarly to random forest, but is simpler and easier
        to interpret. For fairness diagnostics, starting with the simpler
        model makes the analysis easier to understand.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def calculate_group_metrics(y_true, y_pred):
    """
    Calculate classification metrics for one subgroup.

    Confusion matrix layout:
        [[true negatives, false positives],
         [false negatives, true positives]]
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # False positive rate:
    # Of the people who were actually negative, how many were incorrectly flagged?
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    # False negative rate:
    # Of the people who were actually positive, how many did the model miss?
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        "n": len(y_true),
        "positive_rate_actual": y_true.mean(),
        "positive_rate_predicted": y_pred.mean(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def run_subgroup_analysis():
    """
    Train logistic regression and evaluate metrics by subgroup.
    """
    X, y, target_name = load_diabetes_data(binary=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()

    print("Training logistic regression...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    subgroup_columns = ["sex", "income", "education"]

    rows = []

    for subgroup_column in subgroup_columns:
        print(f"Analyzing subgroup: {subgroup_column}")

        for group_value in sorted(X_test[subgroup_column].unique()):
            group_mask = X_test[subgroup_column] == group_value

            y_true_group = y_test[group_mask]
            y_pred_group = y_pred[group_mask]

            metrics = calculate_group_metrics(y_true_group, y_pred_group)

            rows.append({
                "subgroup": subgroup_column,
                "group_value": group_value,
                **metrics,
            })

    subgroup_df = pd.DataFrame(rows)

    output_path = REPORTS_DIR / "fairness_metrics.csv"
    subgroup_df.to_csv(output_path, index=False)

    print()
    print(subgroup_df.round(4))
    print()
    print(f"Saved subgroup metrics to {output_path}")

    plot_recall_and_fnr(subgroup_df)


def plot_recall_and_fnr(subgroup_df):
    """
    Plot recall and false negative rate by subgroup.

    Why:
        In a health-risk setting, false negatives are important because they
        represent missed diabetes/prediabetes cases.
    """
    for subgroup in subgroup_df["subgroup"].unique():
        subset = subgroup_df[subgroup_df["subgroup"] == subgroup].copy()

        labels = subset["group_value"].astype(str)

        plt.figure(figsize=(8, 5))
        plt.plot(labels, subset["recall"], marker="o", label="Recall")
        plt.plot(
            labels,
            subset["false_negative_rate"],
            marker="o",
            label="False Negative Rate",
        )

        plt.title(f"Recall and False Negative Rate by {subgroup}")
        plt.xlabel(subgroup)
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = FIGURES_DIR / f"fairness_{subgroup}_recall_fnr.png"
        plt.savefig(output_path, dpi=300)
        plt.show()

        print(f"Saved plot to {output_path}")


# Run this script with:
#     python -m src.fairness_analysis
if __name__ == "__main__":
    run_subgroup_analysis()
