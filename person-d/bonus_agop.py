import numpy as np
import torch
import matplotlib.pyplot as plt
from xrfm import xRFM
from load_data import load_dataset
import pickle
import os

# ── Load data ──────────────────────────────────────────────────────
X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_dataset(
    "superconductivity", return_feature_names=True
)

# ── Load saved model ───────────────────────────────────────────────
print("Loading saved model...")
with open("xrfm_superconductor.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded!")

# ── Standard AGOP ─────────────────────────────────────────────────
print("\nExtracting standard AGOP...")
agops = model.collect_best_agops()
agop_matrix = torch.stack(agops).mean(dim=0).numpy()
agop_diag_standard = np.diag(agop_matrix)

# ── Residual Weighted AGOP ─────────────────────────────────────────
print("Computing residual-weighted AGOP...")

def residual_weighted_agop(model, X, y, phi="squared"):
    """
    Compute residual-weighted AGOP.
    
    Standard AGOP:
        AGOP = (1/n) * sum_i [ grad_f(x_i) @ grad_f(x_i).T ]
    
    Residual-weighted AGOP:
        w_i = phi(r_i)  where r_i = y_i - f(x_i)
        AGOP_res = sum_i [ w_i * grad_f(x_i) @ grad_f(x_i).T ] / sum_i(w_i)
    
    phi options:
        'squared' : w_i = r_i^2  (canonical choice from spec)
        'abs'     : w_i = |r_i|
    """
    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_tensor.requires_grad_(True)

    # Get predictions and residuals
    y_pred = model.predict(X).squeeze()
    residuals = y - y_pred
    
    # Compute weights
    if phi == "squared":
        weights = residuals ** 2
    elif phi == "abs":
        weights = np.abs(residuals)
    else:
        raise ValueError(f"Unknown phi: {phi}")
    
    weights = weights / weights.sum()  # normalise weights
    
    print(f"  Residual stats: mean={residuals.mean():.3f}, std={residuals.std():.3f}")
    print(f"  Weight stats:   max={weights.max():.6f}, min={weights.min():.6f}")
    
    # Compute gradients at each training point
    # We use finite differences since xRFM predict() doesn't expose gradients directly
    eps = 1e-4
    n, d = X.shape
    gradients = np.zeros((n, d))
    
    print(f"  Computing gradients for {n} points (this takes a moment)...")
    
    # Compute gradient via finite differences
    y_base = model.predict(X).squeeze()
    
    for j in range(d):
        X_plus = X.copy()
        X_plus[:, j] += eps
        y_plus = model.predict(X_plus).squeeze()
        gradients[:, j] = (y_plus - y_base) / eps
        
        if j % 10 == 0:
            print(f"  Feature {j+1}/{d}...")
    
    # Compute weighted AGOP
    agop_res = np.zeros((d, d))
    for i in range(n):
        g = gradients[i:i+1].T  # column vector (d, 1)
        agop_res += weights[i] * (g @ g.T)
    
    return agop_res, gradients, weights, residuals


# Use a subset for speed (finite differences on 14k points × 81 features is slow)
# Use 2000 training points
N_SUBSET = 2000
np.random.seed(42)
idx = np.random.choice(len(X_train), N_SUBSET, replace=False)
X_sub = X_train[idx]
y_sub = y_train[idx]

print(f"\nUsing subset of {N_SUBSET} training points for gradient computation...")
agop_res_matrix, gradients, weights, residuals = residual_weighted_agop(
    model, X_sub, y_sub, phi="squared"
)

# Diagonal of residual-weighted AGOP
agop_diag_residual = np.diag(agop_res_matrix)

# Normalise both for comparison
agop_diag_standard_norm = agop_diag_standard / agop_diag_standard.sum()
agop_diag_residual_norm = agop_diag_residual / agop_diag_residual.sum()

# ── Compare split directions ───────────────────────────────────────
print("\n--- Top eigenvectors (split directions) ---")

# Standard AGOP top eigenvector
eigvals_std, eigvecs_std = np.linalg.eigh(agop_matrix)
top_eigvec_std = eigvecs_std[:, -1]  # largest eigenvalue

# Residual AGOP top eigenvector  
eigvals_res, eigvecs_res = np.linalg.eigh(agop_res_matrix)
top_eigvec_res = eigvecs_res[:, -1]  # largest eigenvalue

# Cosine similarity between the two split directions
cosine_sim = np.abs(np.dot(top_eigvec_std, top_eigvec_res))
print(f"Cosine similarity between split directions: {cosine_sim:.4f}")
print(f"(1.0 = identical direction, 0.0 = completely different)")

if cosine_sim < 0.9:
    print(">>> DISAGREEMENT FOUND - directions differ meaningfully!")
else:
    print(">>> Directions are similar")

# Top features in each eigenvector
print("\nTop 5 features in standard AGOP eigenvector:")
top_std_idx = np.argsort(np.abs(top_eigvec_std))[::-1][:5]
for i in top_std_idx:
    print(f"  {feature_names[i]}: {top_eigvec_std[i]:.4f}")

print("\nTop 5 features in residual AGOP eigenvector:")
top_res_idx = np.argsort(np.abs(top_eigvec_res))[::-1][:5]
for i in top_res_idx:
    print(f"  {feature_names[i]}: {top_eigvec_res[i]:.4f}")

# ── Plot comparison ────────────────────────────────────────────────
import pandas as pd

comparison_df = pd.DataFrame({
    "feature": feature_names,
    "Standard AGOP": agop_diag_standard_norm,
    "Residual AGOP": agop_diag_residual_norm,
})

fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# Plot 1 - Standard AGOP diagonal
top_std = comparison_df.nlargest(15, "Standard AGOP")
axes[0].barh(top_std["feature"], top_std["Standard AGOP"], color="#e63946")
axes[0].set_title("Standard AGOP\n(uniform weights)", fontweight="bold")
axes[0].invert_yaxis()
axes[0].set_xlabel("Normalised Importance")

# Plot 2 - Residual AGOP diagonal
top_res = comparison_df.nlargest(15, "Residual AGOP")
axes[1].barh(top_res["feature"], top_res["Residual AGOP"], color="#2a9d8f")
axes[1].set_title("Residual-weighted AGOP\n(φ(r) = r²)", fontweight="bold")
axes[1].invert_yaxis()
axes[1].set_xlabel("Normalised Importance")

# Plot 3 - Residual distribution
axes[2].hist(residuals, bins=50, color="#457b9d", edgecolor="white", alpha=0.8)
axes[2].set_title("Residual Distribution\n(high |r| = upweighted points)", fontweight="bold")
axes[2].set_xlabel("Residual (y - ŷ)")
axes[2].set_ylabel("Count")
axes[2].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero residual")
axes[2].legend()

plt.suptitle(
    "Bonus: Standard AGOP vs Residual-Weighted AGOP\nSuperconductor Dataset",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("bonus_agop_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to bonus_agop_comparison.png")

# Save results
comparison_df.to_csv("bonus_agop_comparison.csv", index=False)
print("Saved bonus_agop_comparison.csv")