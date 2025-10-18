from pathlib import Path
import shutil
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def download_data() -> Path:
    file = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
    file_path = Path(file) / "spam.csv"

    if not file_path.exists():
        raise RuntimeError("Could not find spam data set in cache directory.")

    return file_path


def clean_data(raw_data_file: Path, cleaned_data_output: Path) -> Path:
    df = pd.read_csv(raw_data_file, encoding="latin-1").rename(
        columns={"v1": "label", "v2": "text"}
    )[["label", "text"]]
    df.to_csv(cleaned_data_output)
    print(f"Wrote cleaned data file to '{cleaned_data_output}'")

    return cleaned_data_output


def load_test_train_data(data_dir: Path) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    cleaned_data_file = data_dir / "cleaned-spam.csv"

    if not cleaned_data_file.exists():
        raw_data_file = download_data()
        clean_data(raw_data_file, cleaned_data_file)

    else:
        print(f"Found cached data '{cleaned_data_file}'")

    df = pd.read_csv(cleaned_data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    # Ensure expected data types
    assert isinstance(X_train, pd.Series), "X_train was not the expected type"
    assert isinstance(X_test, pd.Series), "X_test was not the expected type"
    assert isinstance(y_train, pd.Series), "x_train was not the expected type"
    assert isinstance(y_test, pd.Series), "y_train was not the expected type"

    return X_train, X_test, y_train, y_test
