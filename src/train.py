import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_diabetes_data
from src.evaluate import evaluate_binary_classifier


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def train_and_evaluate_baselines():
    """
    Train a dummy baseline and logistic regression model on the diabetes dataset.
    """
    X, y, target_name = load_diabetes_data(binary=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
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
    }

    results = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        metrics = evaluate_binary_classifier(y_test, y_pred, y_proba)
        results[model_name] = metrics

    results_df = pd.DataFrame(results).T
    print()
    print(results_df.round(4))

    output_path = RESULTS_DIR / "baseline_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    train_and_evaluate_baselines()
