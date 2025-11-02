from pathlib import Path
from argparse import ArgumentParser

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def create_data_set(dataset_path: Path) -> Path:
    """
    Download, clean, and write the SMS spam dataset to disk.

    Steps:
    1. Download the Kaggle dataset "uciml/sms-spam-collection-dataset"
       into the local Kaggle cache.
    2. Load the raw `spam.csv`.
    3. Clean it down to just the columns we care about: ["label", "text"].
    4. Save the cleaned dataset to `dataset_path` as CSV.

    If `dataset_path`'s parent directory does not exist, it will be created.

    Parameters
    ----------
    dataset_path : Path
        Output location for the cleaned dataset CSV (e.g. Path("dataset/spam.csv")).

    Returns
    -------
    Path
        The path where the cleaned dataset was written (`dataset_path`).

    Raises
    ------
    RuntimeError
        If the expected `spam.csv` file cannot be found in the Kaggle download.
    """
    # Download the dataset to a cache directory.
    raw_dataset_dir = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
    raw_dataset_path = Path(raw_dataset_dir) / "spam.csv"

    if not raw_dataset_path.exists():
        raise RuntimeError("Could not find spam data set in cache directory.")

    # Ensure the output directory (e.g. "dataset/") exists.
    dataset_path.parent.mkdir(exist_ok=True, parents=True)

    # Load and normalize to just the two columns we care about.
    cleaned_data_set = clean_data(pd.read_csv(raw_dataset_path, encoding="latin-1"))

    # Write cleaned dataset to disk. `index=False` prevents pandas from adding
    # an "Unnamed: 0" index column.
    cleaned_data_set.to_csv(dataset_path, index=False)

    return dataset_path


def clean_data(raw_data_set: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the raw Kaggle dataset columns to a minimal schema.

    The upstream CSV includes multiple unnamed or irrelevant columns like
    'v1', 'v2', etc. We:
      - rename 'v1' -> 'label'  ("ham" / "spam")
      - rename 'v2' -> 'text'   (the message content)
      - drop everything else

    Parameters
    ----------
    raw_data_set : pd.DataFrame
        DataFrame loaded directly from the Kaggle `spam.csv` file.

    Returns
    -------
    pd.DataFrame
        A DataFrame with exactly two columns: ["label", "text"].

    Raises
    ------
    AssertionError
        If the cleaning process somehow is not a pandas DataFrame.
    """
    cleaned_dataframe = raw_data_set.rename(columns={"v1": "label", "v2": "text"})[
        ["label", "text"]
    ]
    # Sanity check for downstream code: ensure we really did produce a DataFrame.
    assert isinstance(cleaned_dataframe, pd.DataFrame)
    return cleaned_dataframe


def load_test_train_data(
    dataset_path: Path,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load (or create) the cleaned dataset and return stratified train/test splits.

    If the cleaned dataset file does not exist at `dataset_path`, this function
    will create it by downloading the Kaggle dataset via `create_data_set()`.

    We do a stratified split so that the spam/ham ratio is preserved in both
    train and test sets, which is important for evaluation quality.

    Parameters
    ----------
    dataset_path : Path
        Path to the cleaned CSV written by `create_data_set`.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        A tuple `(X_train, X_test, y_train, y_test)` where:
        - X_* are message texts (pd.Series)
        - y_* are labels ("spam"/"ham") (pd.Series)

    Raises
    ------
    AssertionError
        If any of the returned split components are not pandas Series, which
        would break training code that assumes Series access patterns.
    """

    # If we don't have a cleaned dataset yet, create it now.
    if not dataset_path.exists():
        create_data_set(dataset_path)

    # Load the cleaned canonical dataset.
    dataset = pd.read_csv(dataset_path)

    # Split into train/test, preserving label distribution.
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["text"],
        dataset["label"],
        test_size=0.2,
        stratify=dataset["label"],
        random_state=RANDOM_STATE,
    )

    # Sanity checks: downstream training assumes each of these is a Series.
    assert isinstance(X_train, pd.Series), "X_train is not a dataframe."
    assert isinstance(X_test, pd.Series), "X_test is not a dataframe."
    assert isinstance(y_train, pd.Series), "x_train is not a dataframe."
    assert isinstance(y_test, pd.Series), "y_train is not a dataframe."

    return X_train, X_test, y_train, y_test


def main() -> None:
    """
    CLI entry point for creating the spam dataset.

    Example:
        create-spam-dataset --output dataset/spam.csv
    """
    parser = ArgumentParser("create-spam-dataset")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Where to write the cleaned dataset CSV (e.g. dataset/spam.csv)",
    )
    args = parser.parse_args()
    create_data_set(args.output)
