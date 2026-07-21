"""
Calibration analysis

Calibration checks whether predicted probabilities are trustworthy.

Example:
    If the model gives a group of people around 0.70 predicted risk,
    then ideally about 70% of that group should actually be positive.

Why this matters:
    ROC-AUC tells us whether the model ranks positives above negatives.
    Calibration tells us whether the predicted probabilities are meaningful.

For health-risk modeling:
    Good calibration matters if we want to interpret model outputs as risk scores.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_diabetes_data


REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("results") / "figures"

REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def build_models():
    """
    Create the models we want to compare.

    Why this function exists:
        It keeps model definitions in one place and makes the script easier to read.
    """
    return {
        "logistic_regression": Pipeline(
            steps=[
                # Logistic regression benefits from scaling.
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


def plot_calibration_curve(calibration_results):
    """
    Plot calibration curves for all models.

    How to read the plot:
        x-axis = average predicted probability
        y-axis = actual fraction of positives

    A perfectly calibrated model would follow the diagonal line.
    """
    plt.figure(figsize=(7, 6))

    # Diagonal reference line for perfect calibration.
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    for model_name, result in calibration_results.items():
        plt.plot(
            result["mean_predicted_probability"],
            result["fraction_of_positives"],
            marker="o",
            label=model_name,
        )

    plt.title("Calibration Curves")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Actual Fraction of Positives")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURES_DIR / "calibration_curves.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved calibration plot to {output_path}")


def run_calibration_analysis():
    """
    Train models and evaluate how well-calibrated their probabilities are.
    """
    X, y, target_name = load_diabetes_data(binary=True)

    # Use the same train/test split as the other scripts.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = build_models()

    calibration_results = {}
    brier_scores = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        model.fit(X_train, y_train)

        # Predicted probability for the positive class.
        y_proba = model.predict_proba(X_test)[:, 1]

        # Brier score measures probability error.
        # Lower Brier score is better.
        brier_scores[model_name] = brier_score_loss(y_test, y_proba)

        # calibration_curve groups predictions into bins.
        #
        # For each bin, it compares:
        #     average predicted probability
        #     actual fraction of positives
        fraction_of_positives, mean_predicted_probability = calibration_curve(
            y_test,
            y_proba,
            n_bins=10,
            strategy="uniform",
        )

        calibration_results[model_name] = {
            "fraction_of_positives": fraction_of_positives,
            "mean_predicted_probability": mean_predicted_probability,
        }

    brier_df = pd.DataFrame(
        [
            {"model": model_name, "brier_score": score}
            for model_name, score in brier_scores.items()
        ]
    ).sort_values("brier_score")

    print()
    print("Brier scores:")
    print(brier_df.round(4))

    output_path = REPORTS_DIR / "calibration_metrics.csv"
    brier_df.to_csv(output_path, index=False)

    print()
    print(f"Saved calibration metrics to {output_path}")

    plot_calibration_curve(calibration_results)


# Run this script with:
#     python -m src.calibration_analysis
if __name__ == "__main__":
    run_calibration_analysis()
