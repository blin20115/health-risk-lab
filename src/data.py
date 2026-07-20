from ucimlrepo import fetch_ucirepo
import pandas as pd


# UCI dataset ID for the CDC Diabetes Health Indicators dataset.
# Using this ID lets us fetch the dataset directly from UCI.
DATASET_ID = 891


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names so they are easier to use in code.

    Example:
        'HighBP' -> 'highbp'

    Why:
        Lowercase column names are easier to type and less error-prone.
    """
    df = df.copy()

    df.columns = [
        col.strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    return df


def load_diabetes_data(binary: bool = True):
    """
    Load the CDC Diabetes Health Indicators dataset.

    Returns:
        X:
            Feature dataframe.
            These are the inputs the model uses to make predictions.

        y:
            Target labels.
            This is what the model is trying to predict.

        target_name:
            Name of the prediction target.

    Parameters:
        binary:
            If True, convert the target into:

                0 = no diabetes
                1 = prediabetes or diabetes

    Why:
        We want this project to be a binary classification problem.
        Binary classification means the model predicts one of two classes.
    """
    dataset = fetch_ucirepo(id=DATASET_ID)

    # X contains all input features, such as high blood pressure, BMI, age, etc.
    X = dataset.data.features.copy()

    # y_df contains the target column, which is the diabetes label.
    y_df = dataset.data.targets.copy()

    # This dataset should only have one target column.
    # This check helps catch unexpected format changes.
    if y_df.shape[1] != 1:
        raise ValueError(f"Expected one target column, found {y_df.shape[1]}.")

    target_name = y_df.columns[0]

    # Convert target values to integers so sklearn can use them.
    y = y_df.iloc[:, 0].astype(int)

    if binary:
        # Convert the original target into:
        #   0 = no diabetes
        #   1 = prediabetes or diabetes
        #
        # This makes the project easier to frame as a risk prediction task.
        y = (y > 0).astype(int)
        target_name = "diabetes_binary"

    X = clean_column_names(X)

    return X, y, target_name


def print_dataset_summary(X: pd.DataFrame, y: pd.Series, target_name: str) -> None:
    """
    Print basic dataset information.

    Why:
        Before training any model, we should check:
            - number of rows
            - number of features
            - missing values
            - target distribution

        This helps us catch obvious data issues early.
    """
    print("Dataset loaded successfully.")
    print()

    print(f"Number of rows: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target: {target_name}")
    print()

    print("Feature columns:")
    print(list(X.columns))
    print()

    print("Missing values:")

    # Count missing values in each feature column.
    missing = X.isna().sum()

    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found.")

    print()
    print("Target distribution:")

    # Count how many examples are in each class.
    target_counts = y.value_counts().sort_index()

    # Convert class counts into percentages.
    target_percentages = y.value_counts(normalize=True).sort_index() * 100

    summary = pd.DataFrame({
        "count": target_counts,
        "percentage": target_percentages.round(2),
    })

    print(summary)


# This block only runs when we directly run:
#     python src/data.py
#
# It does not run when another file imports load_diabetes_data().
if __name__ == "__main__":
    X, y, target_name = load_diabetes_data(binary=True)
    print_dataset_summary(X, y, target_name)
