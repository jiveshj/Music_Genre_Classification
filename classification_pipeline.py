"""
Output format
-------------
{
  "KNN": {
    "spectrogram": {
      "linear": {
        "aggregated": {"TPR": 0.85, "FPR": 0.02, "confusion_matrix": [[...]], "ground_truth": [...], "predictions": [...]},
        "folds": [
          {"fold": 0, "TPR": ..., "FPR": ..., "confusion_matrix": [[...]], "ground_truth": [...], "predictions": [...]},
          ...
        ]
      },
      "rbf": {...},
      "poly": {...}
    },
    ...
  },
  ...
}
"""

import json
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler, PolynomialCountSketch
import seaborn as sns
import matplotlib.pyplot as plt


N_CLASSES = 10
def kernel_linear(X: np.ndarray) -> np.ndarray:
    return X 

def kernel_rbf(gamma: float) -> np.ndarray:
    rbf = RBFSampler(gamma=gamma, n_components=512, random_state=42)
    return rbf


def kernel_poly(degree: int = 2) -> np.ndarray:
    poly = PolynomialCountSketch(degree=degree, n_components=512, random_state=42)
    return poly

KERNELS = {
    "linear": kernel_linear,
    "rbf":    kernel_rbf,
    "poly":   kernel_poly,
}


def compute_tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    total_tp, total_fn, total_fp, total_tn = 0, 0, 0, 0
    for c in range(N_CLASSES):
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        total_tp += tp
        total_fn += fn
        total_fp += fp
        total_tn += tn

    tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
    return float(tpr), float(fpr)

def load_features(path: str, feature_key: str):
    with open(path) as f:
        data = json.load(f)

    le = LabelEncoder()
    labels = []
    for v in data.values():
        if "label" in v:
            labels.append(v["label"])
        elif "genre" in v:
            labels.append(v["genre"])
    le.fit(labels)

    X, y = [], []
    for entry in data.values():
        if feature_key not in entry:
            continue
        X.append(entry[feature_key])
        y.append(entry["label"] if "label" in entry else entry["genre"])

    X = np.array(X, dtype=np.float32)
    y = le.transform(y)
    return X, y, le


def evaluate(X: np.ndarray, y: np.ndarray, classifier, kernel_fn, genre_names: list, n_splits: int = 5, clf_name: str = "", feat_key: str = "", kernel_name: str = "") -> list:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        if kernel_fn is not kernel_linear:
            if kernel_fn == kernel_rbf:
                gamma = 1.0 / X_train.shape[1]
                transformer = kernel_fn(gamma=gamma)
            else:
                transformer = kernel_fn()
            X_train = transformer.fit_transform(X_train)
            X_test  = transformer.transform(X_test)

        classifier.fit(X_train, y_train)
        preds = classifier.predict(X_test)

        tpr, fpr = compute_tpr_fpr(y_test, preds)
        cm = confusion_matrix(y_test, preds, labels = list(np.arange(N_CLASSES)))

        # saving visualization for KNN with MFCC and linear kernel just for now
        if clf_name == "KNN" and feat_key == "MFCC" and kernel_name == "linear":
       
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=genre_names, yticklabels=genre_names)
            plt.title(f"{clf_name} — {feat_key} — {kernel_name} — Fold {fold_idx}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(f"{clf_name}_{feat_key}_{kernel_name}_fold{fold_idx}.png", dpi=100)
            plt.close()


        fold_results.append({
            "fold": fold_idx,
            "TPR": tpr,
            "FPR": fpr,
            "confusion_matrix": cm.tolist(),
            "ground_truth": y_test.tolist(),
            "predictions": preds.tolist(),
        })
    
    all_gt   = []   # Just combining all the folds' results
    all_pred = []
    cms, tprs, fprs = [], [], []
    for fold in fold_results:
        all_gt.extend(fold["ground_truth"])
        all_pred.extend(fold["predictions"])
        cms.append(np.array(fold["confusion_matrix"]))
        tprs.append(fold["TPR"])
        fprs.append(fold["FPR"])
 
    return {
        "aggregated": {
            "TPR": float(np.mean(tprs)),
            "FPR": float(np.mean(fprs)),
            "confusion_matrix": np.sum(cms, axis=0).tolist(),
            "ground_truth": all_gt,
            "predictions": all_pred,
        },
        "folds": fold_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_features", required=True, help="Path to manual_features.json")
    parser.add_argument("--auto_features", required=True, help="Path to auto_features.json")
    parser.add_argument("--out",      default="ClassifierResults.json")
    parser.add_argument("--cv",       type=int, default=2, help="Number of CV folds")
    args = parser.parse_args()

    autofeature_keys = ["wav2vec", "hubert", "mert"]
    manualfeature_keys = ["MFCC", "spectrogram", "mel_spectrogram", "log_mel_spectrogram"]
    results = {}
    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(max_depth=20, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    }


    for clf_name, clf in classifiers.items():
        print(f"\n── {clf_name} ──")
        results[clf_name] = {}

        for feat_key in manualfeature_keys:
            print(f"  Feature: {feat_key}")
            X, y, le = load_features(args.manual_features, feat_key)
            genre_names = le.classes_.tolist()
            results[clf_name][feat_key] = {}
            for kernel_name, kernel_fn in tqdm(KERNELS.items(), desc=f"    Kernels", leave=False):
                metrics = evaluate(X, y, clf, kernel_fn, genre_names, n_splits=args.cv, clf_name=clf_name, feat_key=feat_key, kernel_name=kernel_name)
        
                results[clf_name][feat_key][kernel_name] = metrics
                agg = metrics["aggregated"]
                print(f"[{kernel_name}] TPR={agg['TPR']:.3f}  FPR={agg['FPR']:.3f}")
        
        for feat_key in autofeature_keys:
            print(f"  Feature: {feat_key}")
            X, y, le = load_features(args.auto_features, feat_key)
            results[clf_name][feat_key] = {}
            genre_names = le.classes_.tolist()

            for kernel_name, kernel_fn in tqdm(KERNELS.items(), desc=f"    Kernels", leave=False):
                metrics = evaluate(X, y, clf, kernel_fn, genre_names,n_splits=args.cv, clf_name=clf_name, feat_key=feat_key, kernel_name=kernel_name)
                
                results[clf_name][feat_key][kernel_name] = metrics
                agg = metrics["aggregated"]
                print(f"[{kernel_name}] TPR={agg['TPR']:.3f}  FPR={agg['FPR']:.3f}")
                    
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Results -> {args.out}")


if __name__ == "__main__":
    main()