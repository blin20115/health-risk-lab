"""
Feature importance analysis

This script helps answer:

    Which features seem most important for predicting diabetes/prediabetes risk?

We compare:
    1. Logistic regression coefficients
    2. Random forest feature importances

Important:
    Feature importance does not prove causation.
    It only tells us which variables the model used most strongly for prediction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_diabetes_data


REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("results") / "figures"

REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def train_models(X_train, y_train):
    """
    Train logistic regression and random forest models.

    Why:
        We want to compare how two different model types think about feature importance.

        Logistic regression:
            Uses coefficients.
            Bigger absolute coefficient = stronger effect on prediction.

        Random forest:
            Uses tree-based importance.
            Bigger importance = feature was used more to split the data.
    """
    logistic_model = Pipeline(
        steps=[
            # Scaling helps make logistic regression coefficients more comparable.
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

    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("Training logistic regression...")
    logistic_model.fit(X_train, y_train)

    print("Training random forest...")
    random_forest_model.fit(X_train, y_train)

    return logistic_model, random_forest_model


def create_feature_importance_table(X, logistic_model, random_forest_model):
    """
    Create one table with feature importance values from both models.
    """
    feature_names = X.columns

    # Logistic regression coefficient:
    #
    # Positive coefficient:
    #     Higher feature value pushes prediction toward diabetes/prediabetes.
    #
    # Negative coefficient:
    #     Higher feature value pushes prediction toward no diabetes.
    #
    # Absolute coefficient:
    #     Strength of the feature, ignoring direction.
    logistic_coefficients = logistic_model.named_steps["model"].coef_[0]

    # Random forest importance:
    #
    # This tells us how much each feature helped the trees split the data.
    random_forest_importances = random_forest_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "logistic_coefficient": logistic_coefficients,
        "logistic_abs_coefficient": abs(logistic_coefficients),
        "random_forest_importance": random_forest_importances,
    })

    return importance_df


def plot_top_features(importance_df, column, title, output_filename):
    """
    Plot the top 10 features for a selected importance column.
    """
    top_features = (
        importance_df
        .sort_values(column, ascending=False)
        .head(10)
        .sort_values(column)
    )

    plt.figure(figsize=(8, 5))
    plt.barh(top_features["feature"], top_features[column])

    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Feature")
    plt.tight_layout()

    output_path = FIGURES_DIR / output_filename
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved plot to {output_path}")


def run_feature_importance_analysis():
    """
    Train models, calculate feature importance, save results, and create plots.
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

    logistic_model, random_forest_model = train_models(X_train, y_train)

    importance_df = create_feature_importance_table(
        X,
        logistic_model,
        random_forest_model,
    )

    output_path = REPORTS_DIR / "feature_importance.csv"
    importance_df.to_csv(output_path, index=False)

    print()
    print("Top logistic regression features:")
    print(
        importance_df
        .sort_values("logistic_abs_coefficient", ascending=False)
        .head(10)
        [["feature", "logistic_coefficient", "logistic_abs_coefficient"]]
        .round(4)
    )

    print()
    print("Top random forest features:")
    print(
        importance_df
        .sort_values("random_forest_importance", ascending=False)
        .head(10)
        [["feature", "random_forest_importance"]]
        .round(4)
    )

    print()
    print(f"Saved feature importance table to {output_path}")

    plot_top_features(
        importance_df,
        column="logistic_abs_coefficient",
        title="Top Logistic Regression Features",
        output_filename="logistic_feature_importance.png",
    )

    plot_top_features(
        importance_df,
        column="random_forest_importance",
        title="Top Random Forest Features",
        output_filename="random_forest_feature_importance.png",
    )


# Run this script with:
#     python -m src.feature_importance
if __name__ == "__main__":
    run_feature_importance_analysis()
