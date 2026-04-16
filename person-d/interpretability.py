import numpy as np
import torch
import matplotlib.pyplot as plt
from xrfm import xRFM
from load_data import load_dataset
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

# Load data 
X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_dataset(
    "superconductivity", return_feature_names=True
)
print(f"Features: {len(feature_names)}")
print(f"First 5 features: {feature_names[:5]}")

# Train full xRFM model 
import pickle
import os

if os.path.exists("xrfm_superconductor.pkl"):
    print("Loading saved model...")
    with open("xrfm_superconductor.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded!")
else:
    print("\nTraining xRFM...")
    model = xRFM()
    model.fit(X_train, y_train, X_val, y_val)
    with open("xrfm_superconductor.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved!")

# 1. AGOP diagonal 
print("\nExtracting AGOP...")
agops = model.collect_best_agops()

# Average across leaves if multiple
agop_matrix = torch.stack(agops).mean(dim=0).numpy()
agop_diag = np.diag(agop_matrix)
agop_importance = agop_diag / agop_diag.sum()

print(f"Number of leaves: {len(agops)}")
print(f"AGOP shape: {agop_matrix.shape}")

# 2. PCA loadings 
print("Computing PCA...")
pca = PCA(n_components=1)
pca.fit(X_train)
pca_importance = np.abs(pca.components_[0])
pca_importance = pca_importance / pca_importance.sum()

# 3. Mutual information 
print("Computing Mutual Information...")
mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
mi_importance = mi_scores / mi_scores.sum()

# 4. Permutation importance 
print("Computing Permutation Importance...")
from sklearn.metrics import make_scorer, mean_squared_error

def rmse_scorer(estimator, X, y):
    preds = estimator.predict(X).squeeze()
    return -np.sqrt(mean_squared_error(y, preds))  

perm = permutation_importance(
    model, X_val, y_val,
    scoring=rmse_scorer,
    n_repeats=10,
    random_state=42
)
perm_importance = perm.importances_mean
perm_importance = np.clip(perm_importance, 0, None)
perm_importance = perm_importance / perm_importance.sum() if perm_importance.sum() > 0 else perm_importance

# Compile results 
import pandas as pd

importance_df = pd.DataFrame({
    "feature":      feature_names,
    "AGOP":         agop_importance,
    "PCA":          pca_importance,
    "MI":           mi_importance,
    "Permutation":  perm_importance,
})

print("\n--- Top 10 features by AGOP ---")
print(importance_df.nlargest(10, "AGOP")[["feature", "AGOP", "PCA", "MI", "Permutation"]].to_string(index=False))

# Plot
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

methods = ["AGOP", "PCA", "MI", "Permutation"]
colors  = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]
titles  = [
    "AGOP Diagonal (xRFM)",
    "PCA Loadings (unsupervised)",
    "Mutual Information",
    "Permutation Importance"
]

for ax, method, color, title in zip(axes.flatten(), methods, colors, titles):
    top = importance_df.nlargest(15, method)
    ax.barh(top["feature"], top[method], color=color, edgecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Normalised Importance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle(
    "Superconductor: Feature Importance Comparison\nAGOP vs PCA vs Mutual Information vs Permutation Importance",
    fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("superconductor_interpretability.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to superconductor_interpretability.png")

print("\n--- Top 10 by each method ---")
for method in ["AGOP", "PCA", "MI", "Permutation"]:
    print(f"\n{method}:")
    top = importance_df.nlargest(10, method)[["feature", method]]
    print(top.to_string(index=False))
importance_df.to_csv("superconductor_importance_comparison.csv", index=False)
print("Saved importance_comparison.csv")
