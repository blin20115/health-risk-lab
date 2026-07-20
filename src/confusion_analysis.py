from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_diabetes_data


FIGURES_DIR = Path("results") / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm, threshold):
    """
    Plot and save a confusion matrix for a given threshold.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    image = ax.imshow(cm)

    ax.set_title(f"Confusion Matrix at Threshold {threshold}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["No Diabetes", "Diabetes"])
    ax.set_yticklabels(["No Diabetes", "Diabetes"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)
    plt.tight_layout()

    output_path = FIGURES_DIR / f"confusion_matrix_threshold_{threshold}.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved confusion matrix to {output_path}")


def run_confusion_analysis():
    """
    Train logistic regression and create confusion matrices
    for selected probability thresholds.
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

    thresholds = [0.5, 0.6]

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        print()
        print(f"Threshold: {threshold}")
        print(cm)

        tn, fp, fn, tp = cm.ravel()
        print(f"True negatives: {tn:,}")
        print(f"False positives: {fp:,}")
        print(f"False negatives: {fn:,}")
        print(f"True positives: {tp:,}")

        plot_confusion_matrix(cm, threshold)


if __name__ == "__main__":
    run_confusion_analysis()
