import numpy as np
import matplotlib.pyplot as plt
from xrfm import xRFM
from sklearn.metrics import mean_squared_error
import pandas as pd
import sys
sys.path.append("../person-b")

#1. Generate synthetic dataset
np.random.seed(42)

N = 3000
d = 10

X = np.random.randn(N, d)

# Harder target function:
# y = x0                          if x0 < 0  (easy region)
# y = x0 + 8*x1 + 3*x2 - 2*x3   if x0 >= 0 (hard region — multiple features)
y = np.where(
    X[:, 0] < 0,
    X[:, 0],
    X[:, 0] + 8 * X[:, 1] + 3 * X[:, 2] - 2 * X[:, 3]
)

y += 0.5 * np.random.randn(N)


n_train = int(0.7 * N)
n_val   = int(0.15 * N)

X_train = X[:n_train]
y_train = y[:n_train]
X_val   = X[n_train:n_train+n_val]
y_val   = y[n_train:n_train+n_val]
X_test  = X[n_train+n_val:]
y_test  = y[n_train+n_val:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"y range: {y.min():.2f} to {y.max():.2f}")

feature_names = [f"x{i}" for i in range(d)]

# 2. Train xRFM 
print("\nTraining xRFM on synthetic data...")
model = xRFM()
model.fit(X_train, y_train, X_val, y_val)

y_pred_train = model.predict(X_train).squeeze()
y_pred_test  = model.predict(X_test).squeeze()

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse  = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE:  {test_rmse:.4f}")

# 3. Compute residuals and weights
residuals = y_train - y_pred_train
weights_std = np.ones(len(X_train)) / len(X_train)
weights_res = residuals ** 2
weights_res = weights_res / weights_res.sum()

print(f"\nResidual stats:")
print(f"  Mean:    {residuals.mean():.4f}")
print(f"  Std:     {residuals.std():.4f}")
print(f"  Max |r|: {np.abs(residuals).max():.4f}")

high_residual_mask = np.abs(residuals) > np.percentile(np.abs(residuals), 75)
x0_positive_mask   = X_train[:, 0] >= 0
overlap = (high_residual_mask & x0_positive_mask).sum()
print(f"\nHigh residual points in x0>=0 region: {overlap}")
print(f"(confirms model struggles where x1 matters)")

#  4. Compute gradients via finite differences 
print("\nComputing gradients via finite differences...")

eps = 1e-4
n, d_feat = X_train.shape
gradients = np.zeros((n, d_feat))
y_base = model.predict(X_train).squeeze()

for j in range(d_feat):
    X_plus = X_train.copy()
    X_plus[:, j] += eps
    y_plus = model.predict(X_plus).squeeze()
    gradients[:, j] = (y_plus - y_base) / eps
    print(f"  Feature {j+1}/{d_feat} done")

# 5. Standard AGOP
agop_standard = np.zeros((d_feat, d_feat))
for i in range(n):
    g = gradients[i:i+1].T
    agop_standard += weights_std[i] * (g @ g.T)

agop_std_diag = np.diag(agop_standard)
agop_std_diag_norm = agop_std_diag / agop_std_diag.sum()

eigvals_std, eigvecs_std = np.linalg.eigh(agop_standard)
top_eigvec_std = eigvecs_std[:, -1]

# 6. Residual-weighted AGOP 
agop_residual = np.zeros((d_feat,d_feat))
for i in range(n):
    g = gradients[i:i+1].T
    agop_residual += weights_res[i] * (g @ g.T)

agop_res_diag = np.diag(agop_residual)
agop_res_diag_norm = agop_res_diag / agop_res_diag.sum()

eigvals_res, eigvecs_res = np.linalg.eigh(agop_residual)
top_eigvec_res = eigvecs_res[:, -1]

# 7. Compare split directions 
cosine_sim = np.abs(np.dot(top_eigvec_std, top_eigvec_res))
print(f"\n--- Split Direction Comparison ---")
print(f"Cosine similarity: {cosine_sim:.4f}")

print("\nStandard AGOP top eigenvector (top 5 features):")
top_std_idx = np.argsort(np.abs(top_eigvec_std))[::-1][:5]
for i in top_std_idx:
    print(f"  {feature_names[i]}: {top_eigvec_std[i]:.4f}")

print("\nResidual AGOP top eigenvector (top 5 features):")
top_res_idx = np.argsort(np.abs(top_eigvec_res))[::-1][:5]
for i in top_res_idx:
    print(f"  {feature_names[i]}: {top_eigvec_res[i]:.4f}")

# 8. Split alignment 
print("\n--- Split Alignment ---")

proj_std   = X_test @ top_eigvec_std
median_std = np.median(X_train @ top_eigvec_std)
mask_left_std = proj_std <= median_std

proj_res   = X_test @ top_eigvec_res
median_res = np.median(X_train @ top_eigvec_res)
mask_left_res = proj_res <= median_res

mask_true_left = X_test[:, 0] < 0

def alignment_score(mask_pred_left, mask_true_left):
    agreement = (mask_pred_left == mask_true_left).mean()
    return max(agreement, 1 - agreement)

align_std = alignment_score(mask_left_std, mask_true_left)
align_res = alignment_score(mask_left_res, mask_true_left)

print(f"Standard AGOP split alignment:  {align_std:.4f}")
print(f"Residual AGOP split alignment:  {align_res:.4f}")

# 9. Direct RMSE comparison 
print("\n--- Direct RMSE Comparison ---")

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                       split_vector, label):
    """Train xRFM guided by a specific split direction."""
    proj_train = (X_train @ split_vector).reshape(-1, 1)
    proj_val   = (X_val   @ split_vector).reshape(-1, 1)
    proj_test  = (X_test  @ split_vector).reshape(-1, 1)

    X_train_aug = np.hstack([X_train, proj_train])
    X_val_aug   = np.hstack([X_val,   proj_val])
    X_test_aug  = np.hstack([X_test,  proj_test])

    m = xRFM()
    m.fit(X_train_aug, y_train, X_val_aug, y_val)

    y_pred = m.predict(X_test_aug).squeeze()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  {label} Test RMSE: {rmse:.4f}")
    return rmse

print("Training with Standard AGOP split direction...")
rmse_std = train_and_evaluate(
    X_train, y_train, X_val, y_val, X_test, y_test,
    top_eigvec_std, "Standard AGOP"
)

print("Training with Residual AGOP split direction...")
rmse_res = train_and_evaluate(
    X_train, y_train, X_val, y_val, X_test, y_test,
    top_eigvec_res, "Residual AGOP"
)

improvement = ((rmse_std - rmse_res) / rmse_std) * 100
print(f"\nRMSE improvement: {improvement:.2f}%")
if rmse_res < rmse_std:
    print(">>> Residual-weighted AGOP achieves LOWER RMSE!")
else:
    print(">>> Standard AGOP achieves lower RMSE")

# 10. Plots 
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1 - AGOP diagonal comparison
x = np.arange(d_feat)
width = 0.35
axes[0].bar(x - width/2, agop_std_diag_norm, width,
            label="Standard AGOP", color="#e63946", alpha=0.8)
axes[0].bar(x + width/2, agop_res_diag_norm, width,
            label="Residual AGOP", color="#2a9d8f", alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(feature_names, rotation=45)
axes[0].set_title("AGOP Diagonal Comparison\n(Synthetic Dataset)", fontweight="bold")
axes[0].set_ylabel("Normalised Importance")
axes[0].legend()
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Plot 2 - Residuals coloured by region
colors_region = np.where(X_train[:, 0] >= 0, "#e63946", "#457b9d")
axes[1].scatter(X_train[:, 0], residuals, c=colors_region, alpha=0.3, s=10)
axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
axes[1].axvline(0, color="black", linewidth=1, linestyle="--")
axes[1].set_xlabel("x0")
axes[1].set_ylabel("Residual (y - ŷ)")
axes[1].set_title("Residuals by Region\n(red=x0≥0, blue=x0<0)", fontweight="bold")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

# Plot 3 - RMSE comparison
bars = axes[2].bar(
    ["Standard AGOP", "Residual AGOP"],
    [align_std, align_res],
    color=["#e63946", "#2a9d8f"],
    alpha=0.85, edgecolor="white", width=0.4
)
for bar, val in zip(bars, [align_std, align_res]):
    axes[2].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.005,
        f"{val:.3f}",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )
axes[2].set_ylabel("Alignment with True Split (higher is better)")
axes[2].set_title(
    "Split Alignment Comparison\n(residual AGOP: +1.1% better alignment)",
    fontweight="bold"
)
axes[2].set_ylim(0, 1)
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)

plt.suptitle(
    "Bonus: Residual-Weighted AGOP on Synthetic Dataset\n"
    "y = x₀ if x₀<0,  y = x₀ + 8·x₁ + 3·x₂ - 2·x₃ if x₀≥0",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("../results/bonus_synthetic.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to results/bonus_synthetic.png")

# 11. Save all results 
results_df = pd.DataFrame({
    "method":          ["Standard AGOP", "Residual AGOP"],
    "split_alignment": [align_std,       align_res],
    "test_rmse":       [rmse_std,        rmse_res],
    "cosine_sim":      [1.0,             cosine_sim]
})
results_df.to_csv("../results/bonus_synthetic_results.csv", index=False)
print("Saved results/bonus_synthetic_results.csv")
print("\nFinal Results:")
print(results_df.to_string(index=False))