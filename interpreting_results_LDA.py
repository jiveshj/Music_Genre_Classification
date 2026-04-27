import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

with open("Results/ClassifierResults.json") as f:
    results = json.load(f)

with open("features/automatic_features.json") as f:
    feat_data = json.load(f)

genres = sorted({feat_data[k]["label"] for k in feat_data})

cm = np.array(results["DecisionTree"]["mert"]["linear"]["aggregated"]["confusion_matrix"])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

filenames  = list(feat_data.keys())
labels     = [feat_data[f]["label"] for f in filenames]

centroids = {}
for g in genres:
    idxs = [i for i, l in enumerate(labels) if l == g]
    vecs = np.array([feat_data[filenames[i]]["mert"] for i in idxs])
    centroids[g] = vecs.mean(axis=0)

global_mu = np.array([feat_data[f]["mert"] for f in filenames]).mean(axis=0)

sb_per_genre = {}
for g in genres:
    n_c = sum(1 for l in labels if l == g)
    sb_per_genre[g] = n_c * np.linalg.norm(centroids[g] - global_mu) ** 2

centroid_mat = np.array([centroids[g] for g in genres])
dist_matrix  = cdist(centroid_mat, centroid_mat, metric="euclidean")

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

im0 = axes[0].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
axes[0].set_xticks(range(len(genres))); axes[0].set_xticklabels(genres, rotation=45, ha="right")
axes[0].set_yticks(range(len(genres))); axes[0].set_yticklabels(genres)
axes[0].set_title("Confusion Matrix\nDecision Tree + MERT + linear", fontsize=12)
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
for i in range(len(genres)):
    for j in range(len(genres)):
        axes[0].text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                     fontsize=7, color="white" if cm_norm[i,j] > 0.5 else "black")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(dist_matrix, cmap="viridis_r")
axes[1].set_xticks(range(len(genres))); axes[1].set_xticklabels(genres, rotation=45, ha="right")
axes[1].set_yticks(range(len(genres))); axes[1].set_yticklabels(genres)
axes[1].set_title("LDA Between-Class Distance\n(lighter = genres closer in MERT space)", fontsize=12)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

triu           = np.triu_indices(len(genres), k=1)
dist_vals      = dist_matrix[triu]
confusion_vals = cm_norm[triu]

axes[2].scatter(dist_vals, confusion_vals, alpha=0.6, edgecolors="k", linewidths=0.5)
for idx in np.argsort(confusion_vals)[-5:]:
    i, j = triu[0][idx], triu[1][idx]
    axes[2].annotate(f"{genres[i]}↔{genres[j]}",
                     (dist_vals[idx], confusion_vals[idx]),
                     fontsize=7, textcoords="offset points", xytext=(5, 3))

m, b = np.polyfit(dist_vals, confusion_vals, 1)
x_line = np.linspace(dist_vals.min(), dist_vals.max(), 100)
axes[2].plot(x_line, m * x_line + b, "r--", linewidth=1.5, label=f"slope={m:.4f}")
axes[2].set_xlabel("Centroid Distance (MERT space)")
axes[2].set_ylabel("Confusion Rate")
axes[2].set_title("Does Distance Predict Confusion?\n(negative slope = our hypothesis holds)", fontsize=12)
axes[2].legend(); axes[2].grid(alpha=0.3)

fig.suptitle("Decision Tree + MERT — Confusion vs LDA Between-Class Scatter", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("confusion_vs_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nGenres by LDA S_B (higher = more displaced from global mean = more distinctive):")
for g, v in sorted(sb_per_genre.items(), key=lambda x: -x[1]):
    print(f"  {g:12s}  {v:.2e}")
