"""
Calibrated classifier analysis

This script compares uncalibrated models against calibrated models.

Why calibration matters:
    A model can rank examples well but still output probabilities that are
    not trustworthy.

Example:
    If a model predicts 0.70 risk for a group of people, ideally about 70%
    of that group should actually be positive.

Calibration tries to make predicted probabilities better match real outcomes.

What to remember:
    ROC-AUC measures ranking quality.
    Brier score measures probability error.
    Lower Brier score is better.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
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
    Build the uncalibrated base models.

    These use the same model settings as the earlier analysis so the results
    are comparable.
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


def evaluate_predictions(model_name, calibration_type, y_true, y_pred, y_proba):
    """
    Calculate metrics for one model.

    calibration_type:
        "uncalibrated" or "sigmoid_calibrated"

    y_pred:
        Final 0/1 predictions.

    y_proba:
        Predicted probability for the positive class.
    """
    return {
        "model": model_name,
        "calibration_type": calibration_type,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "positive_rate_predicted": y_pred.mean(),
    }


def plot_calibration_curves(curve_results):
    """
    Plot calibration curves for uncalibrated and calibrated models.

    How to read:
        The dashed diagonal line is perfect calibration.

        A curve closer to the diagonal means the predicted probabilities are
        more trustworthy.
    """
    plt.figure(figsize=(8, 6))

    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    for label, result in curve_results.items():
        plt.plot(
            result["mean_predicted_probability"],
            result["fraction_of_positives"],
            marker="o",
            label=label,
        )

    plt.title("Calibration Curves: Uncalibrated vs Calibrated Models")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Actual Fraction of Positives")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURES_DIR / "calibrated_model_curves.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved calibration curve plot to {output_path}")


def run_calibrated_model_analysis():
    """
    Train uncalibrated and calibrated models, then compare their metrics.
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

    rows = []
    curve_results = {}

    for model_name, base_model in base_models.items():
        print(f"Training uncalibrated {model_name}...")

        uncalibrated_model = clone(base_model)
        uncalibrated_model.fit(X_train, y_train)

        y_proba = uncalibrated_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        rows.append(
            evaluate_predictions(
                model_name=model_name,
                calibration_type="uncalibrated",
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
            )
        )

        fraction_of_positives, mean_predicted_probability = calibration_curve(
            y_test,
            y_proba,
            n_bins=10,
            strategy="uniform",
        )

        curve_results[f"{model_name}_uncalibrated"] = {
            "fraction_of_positives": fraction_of_positives,
            "mean_predicted_probability": mean_predicted_probability,
        }

        print(f"Training calibrated {model_name}...")

        # Sigmoid calibration is also called Platt scaling.
        #
        # cv=3 means the training data is internally split into folds for
        # calibration. The test set is still only used for final evaluation.
        calibrated_model = CalibratedClassifierCV(
            clone(base_model),
            method="sigmoid",
            cv=3,
        )

        calibrated_model.fit(X_train, y_train)

        calibrated_y_proba = calibrated_model.predict_proba(X_test)[:, 1]
        calibrated_y_pred = (calibrated_y_proba >= 0.5).astype(int)

        rows.append(
            evaluate_predictions(
                model_name=model_name,
                calibration_type="sigmoid_calibrated",
                y_true=y_test,
                y_pred=calibrated_y_pred,
                y_proba=calibrated_y_proba,
            )
        )

        (
            calibrated_fraction_of_positives,
            calibrated_mean_predicted_probability,
        ) = calibration_curve(
            y_test,
            calibrated_y_proba,
            n_bins=10,
            strategy="uniform",
        )

        curve_results[f"{model_name}_sigmoid_calibrated"] = {
            "fraction_of_positives": calibrated_fraction_of_positives,
            "mean_predicted_probability": calibrated_mean_predicted_probability,
        }

    results_df = pd.DataFrame(rows)

    print()
    print(results_df.round(4))

    output_path = REPORTS_DIR / "calibrated_model_metrics.csv"
    results_df.to_csv(output_path, index=False)

    print()
    print(f"Saved calibrated model metrics to {output_path}")

    plot_calibration_curves(curve_results)


# Run this script with:
#     python -m src.calibrated_models
if __name__ == "__main__":
    run_calibrated_model_analysis()
