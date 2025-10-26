from pathlib import Path
from argparse import ArgumentParser

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def download_data(dataset_path: Path) -> Path:
    file = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
    raw_data_set_path = Path(file) / "spam.csv"

    if not raw_data_set_path.exists():
        raise RuntimeError("Could not find spam data set in cache directory.")

    dataset_path.parent.mkdir(exist_ok=True, parents=True)

    cleaned_data_set = clean_data(pd.read_csv(raw_data_set_path, encoding="latin-1"))
    cleaned_data_set.to_csv(dataset_path)

    return raw_data_set_path


def clean_data(raw_data_set: pd.DataFrame) -> pd.DataFrame:
    df = raw_data_set.rename(columns={"v1": "label", "v2": "text"})[["label", "text"]]
    assert isinstance(df, pd.DataFrame)
    return df


def load_test_train_data(
    dataset_path: Path,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if not dataset_path.exists():
        download_data(dataset_path)

    dataset = pd.read_csv(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset["text"],
        dataset["label"],
        test_size=0.2,
        stratify=dataset["label"],
        random_state=RANDOM_STATE,
    )

    # Ensure expected data types
    assert isinstance(X_train, pd.Series), "X_train was not the expected type"
    assert isinstance(X_test, pd.Series), "X_test was not the expected type"
    assert isinstance(y_train, pd.Series), "x_train was not the expected type"
    assert isinstance(y_test, pd.Series), "y_train was not the expected type"

    return X_train, X_test, y_train, y_test


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    download_data(args.output)
