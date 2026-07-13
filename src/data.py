from ucimlrepo import fetch_ucirepo
import pandas as pd


DATASET_ID = 891


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names so they are easier to use in code.
    Example: 'HighBP' -> 'highbp'
    """
    df = df.copy()
    df.columns = [
        col.strip().lower().replace(" ", "_")
        for col in df.columns
    ]
    return df


def load_diabetes_data(binary: bool = True):
    """
    Load the CDC Diabetes Health Indicators dataset from the UCI ML Repository.

    Parameters
    ----------
    binary:
        If True, convert the target into:
        0 = no diabetes
        1 = prediabetes or diabetes

    Returns
    -------
    X:
        Feature dataframe.
    y:
        Target series.
    target_name:
        Name of the prediction target.
    """
    dataset = fetch_ucirepo(id=DATASET_ID)

    X = dataset.data.features.copy()
    y_df = dataset.data.targets.copy()

    if y_df.shape[1] != 1:
        raise ValueError(f"Expected one target column, found {y_df.shape[1]}.")

    target_name = y_df.columns[0]
    y = y_df.iloc[:, 0].astype(int)

    if binary:
        y = (y > 0).astype(int)
        target_name = "diabetes_binary"

    X = clean_column_names(X)

    return X, y, target_name


def print_dataset_summary(X: pd.DataFrame, y: pd.Series, target_name: str) -> None:
    """
    Print basic dataset information for quick sanity checking.
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
    missing = X.isna().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")
    print()

    print("Target distribution:")
    target_counts = y.value_counts().sort_index()
    target_percentages = y.value_counts(normalize=True).sort_index() * 100

    summary = pd.DataFrame({
        "count": target_counts,
        "percentage": target_percentages.round(2)
    })

    print(summary)


if __name__ == "__main__":
    X, y, target_name = load_diabetes_data(binary=True)
    print_dataset_summary(X, y, target_name)
