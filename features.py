import os
import time
import shutil
import numpy as np
import pandas as pd
import librosa
import yt_dlp

CSV_IN = "metadata_updated.csv"
CSV_OUT = "dataset_features.csv"
TEMP_DIR = "temp_audio"
HIT_THRESHOLD = 1_000_000      # views >= 1M -> Hit, otherwise Non-Hit
CLIP_SECONDS = 60              # analyse 60 s from the middle of each song
SR = 22050

if not os.path.exists(CSV_IN):
    print(f"Error: '{CSV_IN}' not found in current folder.")
    raise SystemExit

if shutil.which("ffmpeg") is None:
    print("Error: ffmpeg not found. Install it first:")
    print("   conda install -c conda-forge ffmpeg")
    raise SystemExit

df = pd.read_csv(CSV_IN, dtype=str)
df.columns = df.columns.str.strip()
print("Columns found:", list(df.columns))


def find_col(*keywords):
    for c in df.columns:
        if any(k in c.lower() for k in keywords):
            return c
    return None


link_col = find_col("link", "url")
song_col = find_col("song", "title", "name")
views_col = find_col("views")

if link_col is None or views_col is None:
    print("Need a link column and a views column in the CSV.")
    raise SystemExit

os.makedirs(TEMP_DIR, exist_ok=True)

ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": os.path.join(TEMP_DIR, "%(id)s.%(ext)s"),
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
    }],
    # If you get "Sign in to confirm you're not a bot", uncomment ONE:
    # "cookiesfrombrowser": ("chrome",),
    # "cookiefile": "cookies.txt",
}


def download_audio(ydl, url):
    """Download audio as WAV and return its file path."""
    info = ydl.extract_info(url, download=True)
    if not info:
        return None
    path = os.path.join(TEMP_DIR, f"{info['id']}.wav")
    return path if os.path.exists(path) else None


def extract_features(path):
    total = librosa.get_duration(path=path)
    offset = max(0, total / 2 - CLIP_SECONDS / 2)
    y, sr = librosa.load(path, sr=SR, mono=True, offset=offset, duration=CLIP_SECONDS)

    # Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # Energy = mean RMS ; Loudness = mean RMS in dB
    rms = librosa.feature.rms(y=y)[0]
    energy = float(np.mean(rms))
    loudness = float(np.mean(librosa.amplitude_to_db(rms, ref=1.0)))

    # MFCC (13 coefficients, averaged over time)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    # Chroma (12 pitch classes, averaged over time)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

    feats = {"Tempo": tempo, "Energy": energy, "Loudness": loudness}
    for i, v in enumerate(mfcc, 1):
        feats[f"MFCC_{i}"] = float(v)
    for i, v in enumerate(chroma, 1):
        feats[f"Chroma_{i}"] = float(v)
    return feats


rows = []
total = len(df)

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for i, row in df.iterrows():
        url = str(row[link_col]).strip() if pd.notna(row[link_col]) else ""
        song = row[song_col] if song_col else f"Row {i + 1}"
        views = pd.to_numeric(row[views_col], errors="coerce")

        print(f"[{i + 1}/{total}] {song}")

        if not url.startswith("http") or pd.isna(views):
            print("   ! Skipped (missing link or views)")
            continue

        wav_path = None
        try:
            wav_path = download_audio(ydl, url)
            if wav_path is None:
                print("   ! Download failed")
                continue

            feats = extract_features(wav_path)
            record = {"Song": song, "Artist": row.get("Artist", ""),
                      "Year": row.get("Year", ""), "Views": int(views)}
            record.update(feats)
            record["Label"] = "Hit" if views >= HIT_THRESHOLD else "Non-Hit"
            rows.append(record)

        except Exception as e:
            print(f"   ! Failed: {str(e)[:150]}")
        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

        time.sleep(1)

        # Save progress every 10 songs so nothing is lost if it stops
        if len(rows) % 10 == 0 and rows:
            pd.DataFrame(rows).to_csv(CSV_OUT, index=False)

result = pd.DataFrame(rows)
result.to_csv(CSV_OUT, index=False)
shutil.rmtree(TEMP_DIR, ignore_errors=True)

print(f"\nDone! {len(result)}/{total} songs processed. Saved to '{CSV_OUT}'.")
if len(result):
    print(result["Label"].value_counts())
