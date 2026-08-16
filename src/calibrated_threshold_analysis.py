"""
Calibrated threshold analysis

Earlier, sigmoid calibration improved Brier score, which means the predicted
probabilities became more trustworthy.

However, at the default threshold of 0.5, calibrated models predicted very few
people as positive and recall dropped a lot.

This script asks:
    What happens if we use lower thresholds for calibrated models?

Why:
    For an imbalanced health-risk task, a calibrated probability below 0.5 can
    still be meaningful. The threshold should depend on the use case.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_diabetes_data


REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("results") / "figures"

REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def build_base_models():
    """
    Build the base models before calibration.

    These match the model settings we used earlier so the results are comparable.
    """
    return {
        "logistic_regression": Pipeline(
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
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_thresholds(model_name, y_true, y_proba, thresholds):
    """
    Evaluate precision, recall, F1, and prediction rate across thresholds.

    y_proba:
        Calibrated predicted probabilities for the positive class.

    threshold:
        The cutoff for turning probabilities into final predictions.

    Example:
        If threshold = 0.15, then:
            probability >= 0.15 -> predict 1
            probability < 0.15 -> predict 0
    """
    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        # False negative rate is 1 - recall.
        # It tells us the fraction of actual positive cases the model missed.
        recall = recall_score(y_true, y_pred, zero_division=0)

        rows.append({
            "model": model_name,
            "threshold": threshold,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall,
            "false_negative_rate": 1 - recall,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "positive_rate_predicted": y_pred.mean(),
        })

    return rows


def plot_threshold_tradeoff(results_df, model_name):
    """
    Plot threshold tradeoffs for one calibrated model.

    How to read:
        As threshold increases, precision usually increases and recall usually decreases.
    """
    model_df = results_df[results_df["model"] == model_name].copy()

    plt.figure(figsize=(8, 5))

    plt.plot(
        model_df["threshold"],
        model_df["precision"],
        marker="o",
        label="Precision",
    )

    plt.plot(
        model_df["threshold"],
        model_df["recall"],
        marker="o",
        label="Recall",
    )

    plt.plot(
        model_df["threshold"],
        model_df["f1"],
        marker="o",
        label="F1",
    )

    plt.title(f"Calibrated Threshold Tradeoff: {model_name}")
    plt.xlabel("Prediction Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURES_DIR / f"calibrated_threshold_{model_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved plot to {output_path}")


def run_calibrated_threshold_analysis():
    """
    Train calibrated models and evaluate them across lower thresholds.
    """
    X, y, target_name = load_diabetes_data(binary=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    base_models = build_base_models()

    # Lower thresholds are important here because the calibrated models
    # predicted very few positives at the default 0.5 threshold.
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    all_rows = []

    for model_name, base_model in base_models.items():
        print(f"Training calibrated {model_name}...")

        calibrated_model = CalibratedClassifierCV(
            clone(base_model),
            method="sigmoid",
            cv=3,
        )

        calibrated_model.fit(X_train, y_train)

        # Calibrated probability for the positive class.
        y_proba = calibrated_model.predict_proba(X_test)[:, 1]

        model_rows = evaluate_thresholds(
            model_name=model_name,
            y_true=y_test,
            y_proba=y_proba,
            thresholds=thresholds,
        )

        all_rows.extend(model_rows)

    results_df = pd.DataFrame(all_rows)

    print()
    print(results_df.round(4))

    output_path = REPORTS_DIR / "calibrated_threshold_metrics.csv"
    results_df.to_csv(output_path, index=False)

    print()
    print(f"Saved calibrated threshold metrics to {output_path}")

    for model_name in base_models.keys():
        plot_threshold_tradeoff(results_df, model_name)


# Run this script with:
#     python -m src.calibrated_threshold_analysis
if __name__ == "__main__":
    run_calibrated_threshold_analysis()
