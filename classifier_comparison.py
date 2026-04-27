import json
import numpy as np
from itertools import product
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_PATH = SCRIPT_DIR / "features/features.json"
OUTPUT_PATH = SCRIPT_DIR / "results.json"

# ---------------------------------------------------------------------------
# What we're comparing
# ---------------------------------------------------------------------------
MANUAL_FEATURES = ("MFCC", "spectrogram", "mel_spectrogram", "log_mel_spectrogram")
AUTOMATIC_FEATURES = ("wav2vec", "hubert", "mert")

FEATURE_NAMES = MANUAL_FEATURES + AUTOMATIC_FEATURES
FEATURE_PAIRS = list(product(MANUAL_FEATURES, AUTOMATIC_FEATURES))

CLASSIFIERS = {
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=2000, random_state=42),
    "SVM": lambda: SVC(kernel="rbf", random_state=42),
    "XGBoost": lambda: XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                                      random_state=42, verbosity=0),
}

N_FOLDS = 5


def load_data(path: Path):
    """
    Load the merged features JSON and return parallel lists of
    feature dicts and genre strings, plus the sorted list of keys.
    """
    with path.open() as f:
        features = json.load(f)

    keys = sorted(features.keys())
    genres = [k.rsplit(".", 1)[0] for k in keys]
    return keys, features, genres


def extract_feature_matrix(features: dict, keys: list[str], feature_name: str):
    """
    Pull a single feature type out of the feature dicts and stack into
    a numpy array. Returns None if the feature is missing for any key.
    """
    vectors = []
    for k in keys:
        vec = features[k].get(feature_name)
        if vec is None:
            return None
        # Some features might be nested lists (e.g. MFCCs as time × coeffs).
        # Flatten to 1-D so every classifier gets a consistent input.
        flat = np.array(vec).flatten()
        vectors.append(flat)
    return np.vstack(vectors)


def concatenate_feature_matrices(*matrices):
    """
    Concatenate per-sample feature vectors horizontally. Returns None
    if any input matrix is None (i.e. a constituent feature was missing).
    """
    if any(m is None for m in matrices):
        return None
    return np.hstack(matrices)


def compute_macro_rates(y_true, y_pred, n_classes):
    """
    Compute macro-averaged TPR and FPR from a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    tprs, fprs = [], []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    return float(np.mean(tprs)), float(np.mean(fprs))


def evaluate(X, y_all, folds, n_classes, clf_factory):
    """
    Run k-fold CV for one (X, classifier) pair and return aggregated
    metrics plus the concatenated ground-truth and predictions.
    """
    all_true, all_pred = [], []
    for train_idx, test_idx in folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = clf_factory()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())

    tpr, fpr = compute_macro_rates(all_true, all_pred, n_classes)
    macro_f1 = f1_score(all_true, all_pred, average="macro")
    accuracy = accuracy_score(all_true, all_pred)
    return {
        "TPR": round(tpr, 4),
        "FPR": round(fpr, 4),
        "F1": round(macro_f1, 4),
        "Accuracy": round(accuracy, 4),
        "Ground_truth": all_true,
        "Predictions": all_pred,
    }


def run_experiment():
    keys, features, genres = load_data(FEATURES_PATH)

    le = LabelEncoder()
    y_all = le.fit_transform(genres)
    n_classes = len(le.classes_)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(skf.split(keys, y_all))

    # Cache single-feature matrices once — reused for both single and paired runs
    feature_matrices = {
        name: extract_feature_matrix(features, keys, name)
        for name in FEATURE_NAMES
    }

    results = {}

    for clf_name, clf_factory in CLASSIFIERS.items():
        print(f"\n{'='*60}")
        print(f"  {clf_name}")
        print(f"{'='*60}")

        # ---- Single features ----
        feature_performance = {}
        for feat_name in FEATURE_NAMES:
            X = feature_matrices[feat_name]
            if X is None:
                print(f"  [{feat_name}] skipped — not present in data")
                continue

            metrics = evaluate(X, y_all, folds, n_classes, clf_factory)
            print(f"  [{feat_name}] TPR={metrics['TPR']:.3f}  "
                  f"FPR={metrics['FPR']:.3f}  F1={metrics['F1']:.3f}  "
                  f"Acc={metrics['Accuracy']:.3f}")
            feature_performance[feat_name] = metrics

        # ---- Paired features (manual × automatic) ----
        feature_pair_performance = {}
        for manual, automatic in FEATURE_PAIRS:
            X = concatenate_feature_matrices(
                feature_matrices[manual], feature_matrices[automatic]
            )
            if X is None:
                print(f"  [{manual} + {automatic}] skipped — missing constituent")
                continue

            metrics = evaluate(X, y_all, folds, n_classes, clf_factory)
            pair_key = f"{manual}+{automatic}"
            print(f"  [{pair_key}] TPR={metrics['TPR']:.3f}  "
                  f"FPR={metrics['FPR']:.3f}  F1={metrics['F1']:.3f}  "
                  f"Acc={metrics['Accuracy']:.3f}")
            feature_pair_performance[pair_key] = metrics

        results[clf_name] = {
            "feature_performance": feature_performance,
            "feature_pair_performance": feature_pair_performance,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Results written to {OUTPUT_PATH}")
    print(f"       Classes: {list(le.classes_)}")
    print(f"       Single features: {len(FEATURE_NAMES)}")
    print(f"       Feature pairs:   {len(FEATURE_PAIRS)}")


if __name__ == "__main__":
    run_experiment()