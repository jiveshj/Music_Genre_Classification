import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MANUAL_FEATURES_PATH = SCRIPT_DIR / "manual_features.json"
AUTO_FEATURES_PATH = SCRIPT_DIR / "automatic_features.json"
OUTPUT_PATH = SCRIPT_DIR / "features.json"

# Feature keys we pull from the automatic extraction
AUTO_FEATURE_NAMES = ("wav2vec", "hubert", "mert", "label")


def normalize_auto_key(key: str) -> str:
    """
    Strip directory prefixes and the .wav extension so that keys like
    'blues/blues.00000.wav' become just 'blues.00000'.
    """
    filename = key.replace("\\", "/").split("/")[-1]
    if filename.endswith(".wav"):
        filename = filename[:-4]
    return filename


def merge_feature_jsons(manual_path: Path, auto_path: Path, out_path: Path) -> None:
    """
    Combine hand-annotated features with automatically extracted ones.

    The manual file is treated as the source of truth for structure and
    ordering. Automatic embeddings (wav2vec, hubert, mert, label) are
    folded in where a matching key exists.
    """
    with manual_path.open() as f:
        manual = json.load(f)

    with auto_path.open() as f:
        raw_auto = json.load(f)

    # Re-key the automatic features so they match the manual naming scheme
    auto = {normalize_auto_key(k): v for k, v in raw_auto.items()}

    merged = {}
    missing = []
    unmatched_auto_keys = set(auto)

    for key, manual_entry in manual.items():
        entry = dict(manual_entry)
        auto_entry = auto.get(key)

        if auto_entry is None:
            missing.append(key)
        else:
            for feat in AUTO_FEATURE_NAMES:
                if feat in auto_entry:
                    entry[feat] = auto_entry[feat]
            unmatched_auto_keys.discard(key)

        merged[key] = entry

    # Write out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(merged, f, indent=2)

    # Summary
    print(f"[DONE] Wrote merged file to: {out_path}")
    print(f"       manual={len(manual)}  auto={len(raw_auto)}  merged={len(merged)}")
    if missing:
        print(f"[WARN] {len(missing)} manual keys had no matching auto entry")
    if unmatched_auto_keys:
        print(f"[WARN] {len(unmatched_auto_keys)} auto keys had no manual counterpart")


if __name__ == "__main__":
    merge_feature_jsons(MANUAL_FEATURES_PATH, AUTO_FEATURES_PATH, OUTPUT_PATH)