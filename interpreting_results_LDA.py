import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

with open("features/automatic_features.json") as f:
    data = json.load(f)

filenames = list(data.keys())
labels    = [data[f]["label"] for f in filenames]
genres    = sorted(set(labels))
MODELS    = ["wav2vec", "hubert", "mert"]

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

centroids = {}  # centroids[model][genre] = mean embedding
for model in MODELS:
    centroids[model] = {}
    for g in genres:
        idxs = [i for i, l in enumerate(labels) if l == g]
        vecs = np.array([data[filenames[i]][model] for i in idxs])
        centroids[model][g] = vecs.mean(axis=0)

model_pairs = list(combinations(MODELS, 2))  # (wav2vec,hubert), (wav2vec,mert), (hubert,mert)

genre_agreement = {}   # genre -> mean cosine sim across the 3 pairs
pair_sims       = {f"{a}_{b}": [] for a, b in model_pairs}

for g in genres:
    sims = []
    for a, b in model_pairs:
        s = cosine_sim(centroids[a][g], centroids[b][g])
        sims.append(s)
        pair_sims[f"{a}_{b}"].append(s)
    genre_agreement[g] = np.mean(sims)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

agreement_vals = [genre_agreement[g] for g in genres]
colors = plt.cm.RdYlGn(np.array(agreement_vals) / max(agreement_vals))
bars = axes[0].bar(genres, agreement_vals, color=colors)
axes[0].set_title("Cross-Model Agreement per Genre\n(higher = models agree = less informative features)", fontsize=12)
axes[0].set_xlabel("Genre")
axes[0].set_ylabel("Mean Cosine Similarity (across model pairs)")
axes[0].tick_params(axis="x", rotation=45)
axes[0].set_ylim(min(agreement_vals) - 0.02, 1.0)
axes[0].grid(axis="y", alpha=0.3)
for bar, val in zip(bars, agreement_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

x = np.arange(len(genres))
width = 0.25
pair_labels = [f"{a}↔{b}" for a, b in model_pairs]
for i, (pair_key, pair_label) in enumerate(zip(pair_sims, pair_labels)):
    axes[1].bar(x + i * width, pair_sims[pair_key], width, label=pair_label, alpha=0.8)

axes[1].set_title("Per Model-Pair Agreement\n(which pairs agree most on each genre?)", fontsize=12)
axes[1].set_xlabel("Genre")
axes[1].set_ylabel("Cosine Similarity")
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(genres, rotation=45, ha="right")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("Cross-Model Feature Agreement by Genre\n(genres where all models agree have less distinctive features → harder to classify)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("cross_model_agreement.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nGenres ranked by cross-model agreement (hardest → easiest to classify):")
for g, v in sorted(genre_agreement.items(), key=lambda x: -x[1]):
    print(f"  {g:12s}  {v:.4f}")
