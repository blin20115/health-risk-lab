"""
Cross-validation stability analysis

So far, we evaluated models on one train/test split.

Cross-validation helps answer:
    Are the model results stable across different splits of the data?

Why this matters:
    A model can look good on one split by chance.
    Cross-validation gives a more reliable estimate of performance.

What to remember:
    Mean metric = average performance across folds.
    Standard deviation = how much performance changes across folds.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_diabetes_data


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def build_models():
    """
    Create the models we want to evaluate.

    We use the same model settings as the earlier baseline script so the results
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


def summarize_cv_results(model_name, cv_results):
    """
    Convert raw cross-validation results into a cleaner summary table.

    cross_validate returns one score per fold.
    This function computes the mean and standard deviation for each metric.
    """
    rows = []

    for key, values in cv_results.items():
        # We only want the actual metric scores, not timing columns.
        if not key.startswith("test_"):
            continue

        metric_name = key.replace("test_", "")

        rows.append({
            "model": model_name,
            "metric": metric_name,
            "mean": values.mean(),
            "std": values.std(),
        })

    return rows


def run_cross_validation():
    """
    Run cross-validation for logistic regression and random forest.
    """
    X, y, target_name = load_diabetes_data(binary=True)

    models = build_models()

    # StratifiedKFold keeps the class ratio similar in each fold.
    # This matters because only about 14% of rows are positive.
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    all_rows = []

    for model_name, model in models.items():
        print(f"Running cross-validation for {model_name}...")

        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )

        all_rows.extend(summarize_cv_results(model_name, cv_results))

    summary_df = pd.DataFrame(all_rows)

    print()
    print(summary_df.round(4))

    output_path = REPORTS_DIR / "cross_validation_metrics.csv"
    summary_df.to_csv(output_path, index=False)

    print()
    print(f"Saved cross-validation metrics to {output_path}")


# Run this script with:
#     python -m src.cross_validation
if __name__ == "__main__":
    run_cross_validation()
