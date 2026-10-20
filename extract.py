import os
import time
import pandas as pd
import yt_dlp

CSV_IN = "metadata.csv"
CSV_OUT = "metadata_updated.csv"

if not os.path.exists(CSV_IN):
    print(f"Error: '{CSV_IN}' not found in current folder.")
    raise SystemExit

# Read everything as text so pandas never complains about float columns
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

if link_col is None:
    print("Could not find a column containing 'link' or 'url'. Rename it and retry.")
    raise SystemExit
print(f"Using link column: '{link_col}' | song column: '{song_col}'")

# Make sure output columns exist and are plain text (object) columns
for col in ["Artist", "Year", "YouTube Views"]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].fillna("").astype(object)

ydl_opts = {
    "skip_download": True,
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    # If you see "Sign in to confirm you're not a bot", uncomment ONE line:
    # "cookiesfrombrowser": ("chrome",),   # or "firefox", "edge"
    # "cookiefile": "cookies.txt",
}


def clean_artist(name):
    if not name:
        return None
    name = name.replace(" - Topic", "")
    for suffix in ("VEVO", "Vevo", "Official"):
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def get_details(ydl, url):
    try:
        info = ydl.extract_info(url, download=False)
    except Exception as e:
        print(f"   ! Failed: {str(e)[:150]}")
        return None, None, None

    if not info:
        return None, None, None

    # Artist: best sources first
    artist = info.get("artist")
    if isinstance(artist, list):
        artist = ", ".join(artist)
    if not artist:
        artist = info.get("creator")
    if not artist:
        artist = clean_artist(info.get("channel") or info.get("uploader"))

    # Year: release year if available, otherwise upload year
    year = info.get("release_year")
    if not year:
        date = info.get("release_date") or info.get("upload_date")
        year = str(date)[:4] if date else None

    return artist, year, info.get("view_count")


success = 0
total = len(df)

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for i, row in df.iterrows():
        raw = row[link_col]
        url = str(raw).strip() if pd.notna(raw) else ""

        if not url.startswith("http"):
            print(f"[{i + 1}/{total}] Skipped (no valid link)")
            continue

        song = row[song_col] if song_col else f"Row {i + 1}"
        print(f"[{i + 1}/{total}] {song}")

        artist, year, views = get_details(ydl, url)

        if artist:
            df.at[i, "Artist"] = str(artist)
        if year:
            df.at[i, "Year"] = str(year)
        if views is not None:
            df.at[i, "YouTube Views"] = str(views)
            success += 1

        time.sleep(1)  # avoid rate limits

df.to_csv(CSV_OUT, index=False)
print(f"\nDone! Updated {success}/{total} rows. Saved to '{CSV_OUT}'.")
