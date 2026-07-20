"""
Confusion matrix analysis

A confusion matrix shows the actual counts behind the model's predictions.

For this project:

    True Negative:
        Model predicts no diabetes, and the person actually has no diabetes.

    False Positive:
        Model predicts diabetes/prediabetes, but the person actually has no diabetes.

    False Negative:
        Model predicts no diabetes, but the person actually has diabetes/prediabetes.

    True Positive:
        Model predicts diabetes/prediabetes, and the person actually has diabetes/prediabetes.

Why this matters:
    Metrics like recall and precision are useful, but counts are often easier
    to understand. The confusion matrix shows exactly how many people were
    correctly or incorrectly classified.
"""

from pathlib import Path

import matplotlib.pyplot as plt
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
    Plot and save a confusion matrix for one threshold.

    Parameters:
        cm:
            2x2 confusion matrix.

        threshold:
            Probability threshold used to create the predictions.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    # imshow displays the confusion matrix as a heatmap-like image.
    image = ax.imshow(cm)

    ax.set_title(f"Confusion Matrix at Threshold {threshold}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["No Diabetes", "Diabetes"])
    ax.set_yticklabels(["No Diabetes", "Diabetes"])

    # Write the actual count inside each matrix cell.
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

    # Save the plot so we can include it in the README/report.
    plt.savefig(output_path, dpi=300)

    # Show the plot while developing.
    # Close the plot window to let the script continue.
    plt.show()

    print(f"Saved confusion matrix to {output_path}")


def run_confusion_analysis():
    """
    Train logistic regression and create confusion matrices.

    We currently look at:
        threshold = 0.5
            Default threshold.

        threshold = 0.6
            Threshold with the best F1 score from our threshold analysis.
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

    # Predicted probabilities for the positive class.
    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = [0.5, 0.6]

    for threshold in thresholds:
        # Convert probabilities to final 0/1 predictions.
        y_pred = (y_proba >= threshold).astype(int)

        # Confusion matrix layout from sklearn:
        #
        # [[true negatives, false positives],
        #  [false negatives, true positives]]
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


# Run this script with:
#     python -m src.confusion_analysis
if __name__ == "__main__":
    run_confusion_analysis()
