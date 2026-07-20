import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.data import load_diabetes_data
from src.evaluate import evaluate_binary_classifier


# Folder where we save model evaluation results.
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def train_and_evaluate_baselines():
    """
    Train and evaluate baseline models.

    Models:
        1. Dummy classifier
            Always predicts the most common class.

        2. Logistic regression
            Simple linear ML model that predicts diabetes risk.

    Why:
        The dummy model tells us what a useless baseline looks like.
        Logistic regression gives us a simple real model to compare against.
    """
    X, y, target_name = load_diabetes_data(binary=True)

    # Split the dataset into training and testing data.
    #
    # Training data:
    #     Used to fit the model.
    #
    # Testing data:
    #     Used to evaluate the model on examples it has not seen before.
    #
    # stratify=y:
    #     Keeps the same class balance in train and test sets.
    #     This matters because the dataset is imbalanced.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        # Dummy baseline:
        # Since most rows are 0, this model always predicts 0.
        # It will have high accuracy but zero recall.
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),

        # Logistic regression baseline:
        # This is a simple, interpretable model.
        "logistic_regression": Pipeline(
            steps=[
                # StandardScaler puts features on a similar scale.
                # This is helpful for logistic regression.
                ("scaler", StandardScaler()),

                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,

                        # This tells the model to care more about the minority class.
                        # Without this, it might mostly predict the majority class.
                        class_weight="balanced",
                    ),
                ),
            ]
        ),

        # Random forest baseline:
        # A random forest is a collection of decision trees.
        #
        # Each tree learns rules like:
        #     if bmi is high and age is high and highbp == 1,
        #     then diabetes risk may be higher.
        #
        # Why we use it:
        #     Logistic regression is simpler and more linear.
        #     Random forest can capture more complex/nonlinear patterns.
        #
        # Important:
        #     More complex does not automatically mean better.
        #     We still need to compare metrics on the test set.
        "random_forest": RandomForestClassifier(
            # Number of trees in the forest.
            # More trees usually gives more stable predictions,
            # but also takes longer to train.
            n_estimators=100,

            # Limits how deep each tree can grow.
            # This helps prevent the model from memorizing the training data.
            max_depth=8,

            # Each final leaf must contain at least 50 samples.
            # This also helps prevent overfitting.
            min_samples_leaf=50,

            # Helps the model pay more attention to the minority positive class.
            class_weight="balanced",

            # Makes the random parts reproducible.
            random_state=42,

            # Uses all available CPU cores to train faster.
            n_jobs=-1,
        ),
    }

    results = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        # Fit the model using the training set.
        model.fit(X_train, y_train)

        # Final 0/1 predictions on the test set.
        y_pred = model.predict(X_test)

        # Predicted probabilities for the positive class.
        #
        # Example:
        #     0.73 means the model gives this row a 73% risk score.
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        # Compute metrics like accuracy, precision, recall, etc.
        metrics = evaluate_binary_classifier(y_test, y_pred, y_proba)
        results[model_name] = metrics

    # Convert results dictionary into a table.
    results_df = pd.DataFrame(results).T

    print()
    print(results_df.round(4))

    # Save the metrics so we can reference them later.
    output_path = RESULTS_DIR / "baseline_metrics.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Saved metrics to {output_path}")


# Run this script with:
#     python -m src.train
if __name__ == "__main__":
    train_and_evaluate_baselines()
