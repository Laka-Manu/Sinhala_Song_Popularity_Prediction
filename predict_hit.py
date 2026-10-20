"""
Usage:
    python predict_hit.py train
    python predict_hit.py predict "path\\to\\song.mp3"
"""
import sys
import os
import numpy as np
import pandas as pd
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET = "dataset_features.csv"
MODEL_FILE = "hit_model.pkl"
CLIP_SECONDS = 60      # must match features.py
SR = 22050             # must match features.py

FEATURE_COLS = (
    ["Tempo", "Energy", "Loudness"]
    + [f"MFCC_{i}" for i in range(1, 14)]
    + [f"Chroma_{i}" for i in range(1, 13)]
)


# ---------------------------------------------------------------
# Feature extraction (same method used to build the dataset)
# ---------------------------------------------------------------
def extract_features(path):
    total = librosa.get_duration(path=path)
    offset = max(0, total / 2 - CLIP_SECONDS / 2)
    y, sr = librosa.load(path, sr=SR, mono=True, offset=offset, duration=CLIP_SECONDS)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    rms = librosa.feature.rms(y=y)[0]
    energy = float(np.mean(rms))
    loudness = float(np.mean(librosa.amplitude_to_db(rms, ref=1.0)))

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

    feats = {"Tempo": tempo, "Energy": energy, "Loudness": loudness}
    for i, v in enumerate(mfcc, 1):
        feats[f"MFCC_{i}"] = float(v)
    for i, v in enumerate(chroma, 1):
        feats[f"Chroma_{i}"] = float(v)
    return feats


# ---------------------------------------------------------------
# Train
# ---------------------------------------------------------------
def train():
    if not os.path.exists(DATASET):
        print(f"Error: '{DATASET}' not found in current folder.")
        return

    df = pd.read_csv(DATASET)
    df = df.dropna(subset=FEATURE_COLS + ["Label"])

    X = df[FEATURE_COLS]
    y = df["Label"]

    counts = y.value_counts()
    print("Class counts:\n", counts, "\n")
    if len(counts) < 2 or counts.min() < 3:
        print("Not enough songs in one of the classes to train. "
              "You need both Hit and Non-Hit songs (at least a few of each).")
        return

    model = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",   # helps when Hit / Non-Hit are unbalanced
            random_state=42,
        )),
    ])

    # Cross-validation gives a more honest score on small datasets
    n_splits = min(5, counts.min())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"Cross-validation accuracy ({n_splits}-fold): "
          f"{scores.mean():.2%} (+/- {scores.std():.2%})")

    baseline = counts.max() / counts.sum()
    print(f"Baseline (always guess the bigger class): {baseline:.2%}\n")

    # Hold-out test report
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print("Test set report:")
    print(classification_report(y_te, pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_te, pred), "\n")

    # Final model on ALL data
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to '{MODEL_FILE}'")

    # Which features mattered most
    imp = pd.Series(model.named_steps["rf"].feature_importances_, index=FEATURE_COLS)
    print("\nTop 5 important features:")
    print(imp.sort_values(ascending=False).head(5).round(4))


# ---------------------------------------------------------------
# Predict
# ---------------------------------------------------------------
def predict(audio_path):
    if not os.path.exists(MODEL_FILE):
        print(f"Model '{MODEL_FILE}' not found. Run:  python predict_hit.py train")
        return
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    print(f"Analysing: {audio_path} ...")
    feats = extract_features(audio_path)

    print("\nExtracted features:")
    print(f"  Tempo    : {feats['Tempo']:.1f} BPM")
    print(f"  Energy   : {feats['Energy']:.4f}")
    print(f"  Loudness : {feats['Loudness']:.2f} dB")
    print(f"  MFCC_1-3 : {feats['MFCC_1']:.2f}, {feats['MFCC_2']:.2f}, {feats['MFCC_3']:.2f}")
    print(f"  Chroma_1-3: {feats['Chroma_1']:.3f}, {feats['Chroma_2']:.3f}, {feats['Chroma_3']:.3f}")

    model = joblib.load(MODEL_FILE)
    X = pd.DataFrame([feats])[FEATURE_COLS]

    label = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    classes = model.named_steps["rf"].classes_

    print("\n==============================")
    print(f"Prediction: {label.upper()}")
    for c, p in zip(classes, probs):
        print(f"  {c}: {p:.1%}")
    print("==============================")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) >= 3 and sys.argv[1] == "predict":
        predict(sys.argv[2])
    else:
        print(__doc__)
