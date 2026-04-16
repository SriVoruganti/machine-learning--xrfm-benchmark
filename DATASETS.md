# Dataset Guide
## COMP9417 xRFM Project - Dataset Selection and Preprocessing

This file explains the final dataset setup and preprocessing steps for our project.

---

## Quick Summary

We use 5 datasets in total: 2 regression and 3 classification.

| Dataset | Task | n | d | Why we picked it |
|---|---|---|---|---|
| Superconductivity | Regression | ~21k | 81 | Covers n > 10k and d > 50 |
| Seoul Bike Sharing | Regression | ~8.7k | Varies after one-hot | Mixed feature types (categorical + numerical) |
| Online Shoppers Intention | Classification | ~12.3k | Varies after one-hot | Mixed types and class imbalance (~15% positive) |
| CDC Diabetes (BRFSS) | Classification | ~253k | 21 | Very large n for scalability experiments |
| Company Bankruptcy | Classification | ~6.8k | 95 | High dimensional classification (d > 50) |

These 5 datasets together cover the project constraints.

---

## Setup

Install dependencies first:

```bash
pip install scikit-learn pandas numpy
```

---

## How To Run Preprocessing

### Step 1 - Download all 5 datasets

1. Superconductivity
    - Link: https://archive.ics.uci.edu/dataset/464/superconductivty+data
    - Download: `train.csv`
    - Rename to: `superconductivity.csv`

2. Seoul Bike Sharing
    - Link: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand
    - Download: `SeoulBikeData.csv`

3. Online Shoppers Intention
    - Link: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
    - Download: `online_shoppers_intention.csv`

4. CDC Diabetes BRFSS
    - Link: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv
    - Download: `diabetes_binary_health_indicators_BRFSS2015.csv`

5. Company Bankruptcy
    - Link: https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction?select=data.csv
    - Download: `data.csv`
    - Rename to: `bankruptcy.csv`

Put all 5 CSV files into the `data/` folder.

Your folder structure should look like this:

```text
project/
|--- data/
|    |--- bankruptcy.csv
|    |--- diabetes_binary_health_indicators_BRFSS2015.csv
|    |--- online_shoppers_intention.csv
|    |--- SeoulBikeData.csv
|    |--- superconductivity.csv
|--- DATASETS.md            <-- this file
|--- load_data.py
|--- preprocess.py
```

### Step 2 - Run the script

```bash
python preprocess.py
```

This preprocesses all 5 datasets and saves train/val/test splits to `preprocessed_data/`.

---

## Loading Datasets In Notebooks

Use `load_data.py`:

```python
from load_data import load_dataset

X_train, X_val, X_test, y_train, y_val, y_test = load_dataset("superconductivity")
```

---

## Preprocessing Details

- Train/val/test split: 70/15/15 with random seed 42.
- Stratification: applied for classification datasets (Online Shoppers, CDC Diabetes, Bankruptcy); not used for regression datasets.
- Scaling: `StandardScaler` is fit on training only, then applied to validation and test.
- Categorical encoding: one-hot encoding is used for categorical columns in mixed-type datasets, with encoder fit on training data only and `handle_unknown="ignore"`.
- Seoul Bike date column: dropped, because time information is represented by other available features.
- Boolean handling: boolean columns in Online Shoppers are converted to 0/1 before encoding.
- Duplicate rows: duplicates are kept as-is (no deduplication step).
- NaN handling: Bankruptcy features are passed through `np.nan_to_num(..., nan=0.0)` as a baseline safeguard.
- Class imbalance: for Online Shoppers, Bankruptcy, and Diabetes, use AUC-ROC as a main metric (alongside accuracy).


---

