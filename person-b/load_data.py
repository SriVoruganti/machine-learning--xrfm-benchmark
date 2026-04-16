import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "preprocessed_data"

DATASET_FILES = {
    "superconductivity": "superconductivity.npz",
    "seoul_bike": "seoul_bike.npz",
    "online_shoppers": "online_shoppers.npz",
    "bankruptcy": "bankruptcy.npz",
    "cdc_diabetes": "cdc_diabetes.npz",
}


def load_dataset(name, return_feature_names=False):
    """Load train/val/test splits for a given dataset name."""

    print(f"\nLoading dataset: '{name}'")

    if name not in DATASET_FILES:
        available = sorted(DATASET_FILES.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    path = OUTPUT_DIR / DATASET_FILES[name]

    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed file not found: {path}\n"
            f"Run preprocess.py first!"
        )

    data = np.load(path, allow_pickle=True)
    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]

    if return_feature_names:
        feature_names = data["feature_names"].tolist() if "feature_names" in data else None
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Example usage
    dataset_name = "superconductivity"
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(dataset_name)
    print(f"Loaded '{dataset_name}' dataset:")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_val shape:   {X_val.shape},   y_val shape:   {y_val.shape}")
    print(f"  X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}")