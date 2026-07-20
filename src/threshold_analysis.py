from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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


def evaluate_thresholds(y_true, y_proba, thresholds):
    """
    Evaluate precision, recall, F1, and predicted positive rate
    across different probability thresholds.
    """
    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "positive_rate_predicted": y_pred.mean(),
        })

    return pd.DataFrame(rows)


def plot_threshold_tradeoff(threshold_df):
    """
    Plot how precision, recall, and F1 change as the threshold changes.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")

    plt.title("Precision, Recall, and F1 Across Thresholds")
    plt.xlabel("Prediction Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURES_DIR / "threshold_tradeoff.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved threshold plot to {output_path}")


def run_threshold_analysis():
    """
    Train logistic regression and evaluate it across multiple thresholds.
    """
    X, y, target_name = load_diabetes_data(binary=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
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

    print("Training logistic regression...")
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = [round(x, 2) for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

    threshold_df = evaluate_thresholds(y_test, y_proba, thresholds)

    print()
    print(threshold_df.round(4))

    output_path = REPORTS_DIR / "threshold_metrics.csv"
    threshold_df.to_csv(output_path, index=False)

    print()
    print(f"Saved threshold metrics to {output_path}")

    plot_threshold_tradeoff(threshold_df)


if __name__ == "__main__":
    run_threshold_analysis()
