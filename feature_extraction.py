import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

import librosa
import numpy as np

from config import SAMPLING_RATE, N_MFCC, N_FFT, HOP_LENGTH

DATA_DIR = Path("data/genres_original")
FEATURES_JSON = Path("features/manual_features.json")


def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLING_RATE)

    # https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
    # shape(mfcc) = (13, 647)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLING_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # https://librosa.org/doc/main/generated/librosa.stft.html
    # shape(stft) = (1025, 647)
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))

    # https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
    # shape(mel) = (128, 647)
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLING_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
    # shape(log_mel) = (128, 647)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # We average across time to get fixed length features 
    # in the lecture we saw how stationary features worked for this case
    return {
        "MFCC": np.mean(mfcc, axis=1).tolist(),
        "spectrogram": np.mean(stft, axis=1).tolist(),
        "mel_spectrogram": np.mean(mel, axis=1).tolist(),
        "log_mel_spectrogram": np.mean(log_mel, axis=1).tolist(),
    }


def main(data_path, output_path):
    wav_files = sorted(data_path.rglob("*.wav"))
    #wav_files = wav_files[:2]  # testing

    print(f"Found {len(wav_files)} wav files")
    features = {}

    for fpath in tqdm(wav_files, total=len(wav_files)):
        try:
            filename = fpath.stem
            features[filename] = extract_features(str(fpath))
            genre, number = filename.split(".")
            features[filename]["genre"] = genre
            features[filename]["number"] = number
        except Exception as e:
            print(f"  ERROR: {e}")

    with open(output_path, "w") as f:
        json.dump(features, f, indent=4)

if __name__ == "__main__":
    main(DATA_DIR, FEATURES_JSON)