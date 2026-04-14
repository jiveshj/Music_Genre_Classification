# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")

# print("Path to dataset files:", path)

"""
Music Genre Classification — Automatic Feature Extraction (Jivesh)
==================================================================
Extracts Wav2Vec 2.0 embeddings from GTZAN audio files.
"""

import os
import json
import argparse
import warnings
from pathlib import Path

import librosa
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

warnings.filterwarnings("ignore")

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
TARGET_SR = 16_000  # Wav2Vec 2.0 expects 16 kHz


def load_model(device: str):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    model.eval()
    return processor, model


def extract_wav2vec(y, sr, processor, model, device: str) -> list:
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    inputs = processor(y, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state: (1, T, 768) -> mean pool over T -> (768,)
    embedding = outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()
    return embedding.tolist()


def collect_files(gtzan_root: Path):
    pairs = []
    for genre in GENRES:
        genre_dir = gtzan_root / genre
        if not genre_dir.is_dir():
            print(f"[WARN] Missing genre folder: {genre_dir}")
            continue
        for f in sorted(genre_dir.iterdir()):
            if f.suffix in {".wav", ".au", ".mp3", ".ogg", ".flac"}:
                pairs.append((f"{genre}/{f.name}", str(f)))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtzan_root", required=True)
    parser.add_argument("--out", default="features.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 "mps"  if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"[INFO] Using device: {device}")

    processor, model = load_model(device)

    results = {}
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
        print(f"[INFO] Resuming — {len(results)} files already done")

    files = collect_files(Path(args.gtzan_root))
    print(f"[INFO] Found {len(files)} audio files")

    errors = []
    for audio_id, path in tqdm(files, desc="Extracting"):
        if args.resume and audio_id in results:
            continue
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
            results[audio_id] = {
                "label":   audio_id.split("/")[0],
                "wav2vec": extract_wav2vec(y, sr, processor, model, device),
            }
        except Exception as e:
            errors.append(audio_id)
            tqdm.write(f"[ERROR] {audio_id}: {e}")

        if len(results) % 10 == 0:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
    if len(results) % 10 != 0:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n[DONE] {len(results)} files -> {args.out}")
    if errors:
        print(f"[WARN] {len(errors)} failed: {errors}")


if __name__ == "__main__":
    main()