"""
Output format
-------------
{
  "KNN": {
    "wav2vec": {
      "linear":   {"TPR": 0.85, "FPR": 0.02, "ground_truth": [...], "predictions": [...]},
      "rbf":      {...},
      "poly":     {...}
    },
    "hubert":  {...},
    "mert":    {...}
    "MFCC": {...},
    "spectrogram": {...},
    "mel_spectrogram": {...},
    "log_mel_spectrogram": {...}
  },
  "DecisionTree": { ... },
  "LogisticRegression": { ... }
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
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import PolynomialCountSketch



def kernel_linear(X: np.ndarray) -> np.ndarray:
    return X


def kernel_rbf(X: np.ndarray, gamma: float = None) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    rbf = RBFSampler(gamma=gamma, n_components=512, random_state=42)
    return rbf.fit_transform(X)


def kernel_poly(X: np.ndarray, degree: int = 2) -> np.ndarray:
    poly = PolynomialCountSketch(degree=degree, n_components=512, random_state=42)
    return poly.fit_transform(X)


KERNELS = {
    "linear": kernel_linear,
    "rbf":    kernel_rbf,
    "poly":   kernel_poly,
}


def compute_tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, float]:
    tprs, fprs = [], []
    for c in range(n_classes):
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    return float(np.mean(tprs)), float(np.mean(fprs))


def load_features(path: str, feature_key: str) -> tuple[np.ndarray, np.ndarray]:
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


def evaluate(X: np.ndarray, y: np.ndarray, classifier, n_splits: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_gt, all_pred = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        classifier.fit(X_train, y_train)
        preds = classifier.predict(X_test)

        all_gt.extend(y_test.tolist())
        all_pred.extend(preds.tolist())

    all_gt   = np.array(all_gt)
    all_pred = np.array(all_pred)
    n_classes = len(np.unique(all_gt))
    tpr, fpr = compute_tpr_fpr(all_gt, all_pred, n_classes)

    return {
        "TPR":          tpr,
        "FPR":          fpr,
        "ground_truth": all_gt.tolist(),
        "predictions":  all_pred.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_features", required=True, help="Path to manual_features.json")
    parser.add_argument("--auto_features", required=True, help="Path to auto_features.json")
    parser.add_argument("--out",      default="ClassifierResults.json")
    parser.add_argument("--cv",       type=int, default=5, help="Number of CV folds")
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
            results[clf_name][feat_key] = {}

            for kernel_name, kernel_fn in tqdm(KERNELS.items(), desc=f"    Kernels", leave=False):
                X_aug = kernel_fn(X)
                metrics = evaluate(X_aug, y, clf, n_splits=args.cv)
                results[clf_name][feat_key][kernel_name] = metrics
                print(f"    [{kernel_name}] TPR={metrics['TPR']:.3f}  FPR={metrics['FPR']:.3f}")
        
        for feat_key in autofeature_keys:
            print(f"  Feature: {feat_key}")
            X, y, le = load_features(args.auto_features, feat_key)
            results[clf_name][feat_key] = {}

            for kernel_name, kernel_fn in tqdm(KERNELS.items(), desc=f"    Kernels", leave=False):
                X_aug = kernel_fn(X)
                metrics = evaluate(X_aug, y, clf, n_splits=args.cv)
                results[clf_name][feat_key][kernel_name] = metrics
                print(f"    [{kernel_name}] TPR={metrics['TPR']:.3f}  FPR={metrics['FPR']:.3f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Results -> {args.out}")


if __name__ == "__main__":
    main()