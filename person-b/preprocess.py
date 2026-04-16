from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "preprocessed_data"

OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def split_data(X, y, test_size=0.15, val_size=0.15, stratify=None):
    """70/15/15 train/val/test split."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )
    inner_strat = y_trainval if stratify is not None else None
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio,
        random_state=RANDOM_SEED,
        stratify=inner_strat,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Fit StandardScaler on train only, transform all splits."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test


def encode_mixed_features(X_train_df, X_val_df, X_test_df, num_cols, cat_cols):
    """Fit OHE on train only; transform val/test with handle_unknown='ignore'."""
    X_train_num = X_train_df[num_cols].to_numpy(dtype=float)
    X_val_num = X_val_df[num_cols].to_numpy(dtype=float)
    X_test_num = X_test_df[num_cols].to_numpy(dtype=float)

    if not cat_cols:
        return X_train_num, X_val_num, X_test_num

    ohe = make_ohe()
    X_train_cat = ohe.fit_transform(X_train_df[cat_cols].astype(str))
    X_val_cat = ohe.transform(X_val_df[cat_cols].astype(str))
    X_test_cat = ohe.transform(X_test_df[cat_cols].astype(str))

    X_train = np.hstack([X_train_num, X_train_cat])
    X_val = np.hstack([X_val_num, X_val_cat])
    X_test = np.hstack([X_test_num, X_test_cat])
    return X_train, X_val, X_test


def make_ohe():
    """Create OneHotEncoder compatible with old/new sklearn versions."""
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def save_splits(name, X_train, X_val, X_test, y_train, y_val, y_test, feature_names=None):
    path = OUTPUT_DIR / f"{name}.npz"
    save_dict = dict(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=np.array(y_train),
        y_val=np.array(y_val),
        y_test=np.array(y_test),
    )
    if feature_names is not None:
        save_dict["feature_names"] = np.array(feature_names, dtype=str)

    np.savez(path, **save_dict)
    print(f"  saved -> {path}")
    print(
        f"  shapes: train={X_train.shape[0]}, "
        f"val={X_val.shape[0]}, test={X_test.shape[0]}, d={X_train.shape[1]}"
    )

def check_file(filepath, dataset_name):
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"  SKIPPED - {dataset_name} file not found: '{filepath}'")
        return False
    return True


def preprocess_superconductivity(filepath=DATA_DIR / "superconductivity.csv"):
    print("\n[1/5] Superconductivity ...")
    if not check_file(filepath, "Superconductivity"):
        return False

    df = pd.read_csv(filepath)

    target_col = "critical_temp"
    if target_col not in df.columns:
        target_col = df.columns[-1]
        print(f"  using last column as target: '{target_col}'")

    y = df[target_col].values.astype(float)
    X = df.drop(columns=[target_col]).values.astype(float)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    save_splits("superconductivity", X_train, X_val, X_test, y_train, y_val, y_test, feature_names=feature_names)
    return True


def preprocess_seoul_bike(filepath=DATA_DIR / "SeoulBikeData.csv"):
    print("\n[2/5] Seoul Bike Sharing Demand ...")
    if not check_file(filepath, "Seoul Bike"):
        return False

    df = pd.read_csv(filepath, encoding="cp1252")

    target_col = "Rented Bike Count"
    if target_col not in df.columns:
        candidates = [
            c
            for c in df.columns
            if "bike" in c.lower() or "count" in c.lower() or "rent" in c.lower()
        ]
        target_col = candidates[0] if candidates else df.columns[1]
        print(f"  using '{target_col}' as target")

    y = df[target_col].to_numpy(dtype=float)
    X_df = df.drop(columns=[target_col]).copy()

    date_cols = [c for c in X_df.columns if "date" in c.lower()]
    if date_cols:
        X_df = X_df.drop(columns=date_cols)

    cat_cols = X_df.select_dtypes(include="object").columns.tolist()
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()

    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = split_data(X_df, y)
    X_train, X_val, X_test = encode_mixed_features(
        X_train_df, X_val_df, X_test_df, num_cols=num_cols, cat_cols=cat_cols
    )
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)
    save_splits("seoul_bike", X_train, X_val, X_test, y_train, y_val, y_test)
    return True


def preprocess_online_shoppers(filepath=DATA_DIR / "online_shoppers_intention.csv"):
    print("\n[3/5] Online Shoppers Intention ...")
    if not check_file(filepath, "Online Shoppers"):
        return False

    df = pd.read_csv(filepath)

    target_col = "Revenue"
    y = df[target_col].astype(int).to_numpy()
    X_df = df.drop(columns=[target_col]).copy()

    bool_cols = X_df.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        X_df[col] = X_df[col].astype(int)

    cat_cols = X_df.select_dtypes(include="object").columns.tolist()
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()

    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = split_data(
        X_df, y, stratify=y
    )
    X_train, X_val, X_test = encode_mixed_features(
        X_train_df, X_val_df, X_test_df, num_cols=num_cols, cat_cols=cat_cols
    )
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)
    save_splits("online_shoppers", X_train, X_val, X_test, y_train, y_val, y_test)
    print(f"  positive class rate: {y.mean() * 100:.1f}%")
    return True


def preprocess_cdc_diabetes(filepath=DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"):
    print("\n[4/5] CDC Diabetes Health Indicators ...")
    if not check_file(filepath, "CDC Diabetes"):
        return False

    df = pd.read_csv(filepath)

    target_col = "Diabetes_binary"
    y = df[target_col].values.astype(int)
    X = df.drop(columns=[target_col]).values.astype(float)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, stratify=y)
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)
    save_splits("cdc_diabetes", X_train, X_val, X_test, y_train, y_val, y_test)
    print(f"  diabetic rate: {y.mean() * 100:.1f}%")
    return True


def preprocess_bankruptcy(filepath=DATA_DIR / "bankruptcy.csv"):
    print("\n[5/5] Company Bankruptcy Prediction ...")
    if not check_file(filepath, "Bankruptcy"):
        return False

    df = pd.read_csv(filepath)

    target_col = "Bankrupt?"
    y = df[target_col].values.astype(int)
    X = df.drop(columns=[target_col]).values.astype(float)
    X = np.nan_to_num(X, nan=0.0)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, stratify=y)
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)
    save_splits("bankruptcy", X_train, X_val, X_test, y_train, y_val, y_test)
    print(f"  bankrupt rate: {y.mean() * 100:.1f}%")
    return True


if __name__ == "__main__":

    print("\nPerforming Dataset Preprocessing")
    
    preprocess_superconductivity()
    preprocess_seoul_bike()
    preprocess_online_shoppers()
    preprocess_cdc_diabetes()
    preprocess_bankruptcy()
    
    print("\nPreprocessed splits saved to: preprocessed_data/")