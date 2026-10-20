import os
import shutil
import tempfile

import joblib
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yt_dlp

# Re-use the exact same feature extraction used for training
from predict_hit import extract_features, FEATURE_COLS, MODEL_FILE, CLIP_SECONDS, SR

# ------------------------------------------------------------------
# Page setup + styling
# ------------------------------------------------------------------
st.set_page_config(page_title="Hit Song Predictor", page_icon="🎵", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    h1, h2, h3, h4, p, label, span, div { color: #f1f1f6; }
    .hero { text-align: center; padding: 10px 0 25px 0; }
    .hero h1 { font-size: 3rem; margin-bottom: 0; }
    .hero p { font-size: 1.1rem; color: #b8b5d6; }
    .card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px; padding: 22px; margin-bottom: 15px;
        backdrop-filter: blur(6px);
    }
    .verdict-hit {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: #06260f !important; font-size: 2.2rem; font-weight: 800;
        text-align: center; padding: 18px; border-radius: 16px;
    }
    .verdict-non {
        background: linear-gradient(90deg, #ff512f, #dd2476);
        color: white !important; font-size: 2.2rem; font-weight: 800;
        text-align: center; padding: 18px; border-radius: 16px;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        padding: 15px; border-radius: 14px;
    }
    div[data-testid="stMetricValue"] { color: #ffd166 !important; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f1f1f6"),
    margin=dict(l=10, r=10, t=40, b=10),
)

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None


def load_clip(path):
    """Load the same middle clip used for feature extraction (for the charts)."""
    y_full, sr = librosa.load(path, sr=SR, mono=True)
    duration = len(y_full) / sr
    clip_len = CLIP_SECONDS * sr
    start = max(0, len(y_full) // 2 - clip_len // 2)
    return y_full[start:start + clip_len], sr, duration


def gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 48}},
        title={"text": "Hit Probability", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ffd166"},
            "steps": [
                {"range": [0, 40], "color": "rgba(221,36,118,0.55)"},
                {"range": [40, 60], "color": "rgba(255,209,102,0.45)"},
                {"range": [60, 100], "color": "rgba(56,239,125,0.55)"},
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "value": 50},
        },
    ))
    fig.update_layout(height=300, **PLOT_LAYOUT)
    return fig


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎵 Hit Song Predictor</h1>
        <p>Upload an MP3 and discover whether its sound profile looks like a Hit or a Non-Hit</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_model()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app extracts audio features with **librosa** and uses a "
        "**Random Forest** model trained on your dataset."
    )
    st.markdown(
        """
        **Features used**
        - Tempo (BPM)
        - Energy (RMS)
        - Loudness (dB)
        - 13 MFCC coefficients
        - 12 Chroma values
        """
    )
    st.markdown("**Label rule:** 1,000,000+ YouTube views = **Hit**")
    st.divider()
    if model is None:
        st.error("Model not found")
    else:
        st.success("Model loaded ✅")
    st.caption(
        "⚠️ Audio alone cannot fully predict popularity. "
        "Treat the result as a rough estimate."
    )

if model is None:
    st.error(
        f"`{MODEL_FILE}` not found. Train the model first by running "
        "`python predict_hit.py train` in this folder, then refresh this page."
    )
    st.stop()

# ------------------------------------------------------------------
# Input: upload a file OR paste a YouTube link
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_youtube(url):
    """Download the audio of a YouTube video as mp3 and return bytes + info."""
    tmp_dir = tempfile.mkdtemp()
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmp_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        # If you get "Sign in to confirm you're not a bot", uncomment ONE:
        # "cookiesfrombrowser": ("chrome",),   # or "firefox", "edge"
        # "cookiefile": "cookies.txt",
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        path = os.path.join(tmp_dir, f"{info['id']}.mp3")
        with open(path, "rb") as f:
            data = f.read()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    date = info.get("upload_date")
    return {
        "bytes": data,
        "title": info.get("title", "YouTube audio"),
        "channel": info.get("channel") or info.get("uploader") or "Unknown",
        "views": info.get("view_count"),
        "year": str(date)[:4] if date else None,
        "thumbnail": info.get("thumbnail"),
    }


mode = st.radio(
    "Choose how to add a song",
    ["📁 Upload a file", "▶️ YouTube link"],
    horizontal=True,
)

audio_bytes = None
display_name = None
suffix = ".mp3"
meta = None

if mode == "📁 Upload a file":
    uploaded = st.file_uploader(
        "Drop your song here",
        type=["mp3", "wav", "m4a", "flac", "ogg"],
        help="Any common audio format works",
    )
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        display_name = uploaded.name
        suffix = os.path.splitext(uploaded.name)[1] or ".mp3"
else:
    c_url, c_btn = st.columns([5, 1])
    url = c_url.text_input(
        "YouTube link",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
    if c_btn.button("Analyse", type="primary", use_container_width=True) and url.strip():
        st.session_state["yt_url"] = url.strip()

    active_url = st.session_state.get("yt_url")
    if active_url:
        try:
            with st.spinner("Downloading audio from YouTube..."):
                meta = download_youtube(active_url)
            audio_bytes = meta["bytes"]
            display_name = meta["title"]
            suffix = ".mp3"
        except Exception as e:
            st.error(f"Could not download this video: {str(e)[:300]}")
            st.caption("Check the link, update yt-dlp (`pip install -U yt-dlp`), "
                       "or enable the cookies option in app.py if YouTube asks to sign in.")
            st.stop()

if audio_bytes is None:
    st.info("👆 Add a song to begin.")
    st.stop()

st.audio(audio_bytes)

# ------------------------------------------------------------------
# Analyse
# ------------------------------------------------------------------
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(audio_bytes)
    tmp_path = tmp.name

try:
    with st.spinner("Analysing the song... this can take 10-30 seconds"):
        feats = extract_features(tmp_path)
        y, sr, duration = load_clip(tmp_path)
        mfcc_mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma_mat = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96), ref=np.max
        )
except Exception as e:
    st.error(f"Could not analyse this file: {e}")
    st.stop()
finally:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

X = pd.DataFrame([feats])[FEATURE_COLS]
label = model.predict(X)[0]
probs = model.predict_proba(X)[0]
classes = list(model.named_steps["rf"].classes_)
hit_prob = probs[classes.index("Hit")] if "Hit" in classes else 0.0

# ------------------------------------------------------------------
# Result
# ------------------------------------------------------------------
st.markdown(f"### 🎧 {display_name}")

if meta:
    i1, i2 = st.columns([1, 3])
    with i1:
        if meta.get("thumbnail"):
            st.image(meta["thumbnail"], use_container_width=True)
    with i2:
        st.markdown(f"**Channel:** {meta['channel']}")
        v = meta.get("views")
        if v is not None:
            actual = "Hit" if v >= 1_000_000 else "Non-Hit"
            st.markdown(f"**YouTube views:** {v:,}  →  actual label by the 1M rule: **{actual}**")
        if meta.get("year"):
            st.markdown(f"**Uploaded:** {meta['year']}")
        st.caption("If this song was in your training dataset, the model has already "
                   "seen it, so the prediction is not a fair test.")

left, right = st.columns([1, 1], gap="large")
with left:
    css = "verdict-hit" if label == "Hit" else "verdict-non"
    icon = "🔥" if label == "Hit" else "📉"
    st.markdown(f'<div class="{css}">{icon} {label.upper()}</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("**Model confidence**")
    for c, p in zip(classes, probs):
        st.write(f"{c}")
        st.progress(float(p), text=f"{p:.1%}")
with right:
    st.plotly_chart(gauge(hit_prob), use_container_width=True)

# ------------------------------------------------------------------
# Key metrics
# ------------------------------------------------------------------
chroma_mean = np.array([feats[f"Chroma_{i}"] for i in range(1, 13)])
dominant_note = NOTES[int(np.argmax(chroma_mean))]
mins, secs = divmod(int(duration), 60)

st.markdown("### 📊 Audio Details")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Tempo", f"{feats['Tempo']:.0f} BPM")
m2.metric("Energy", f"{feats['Energy']:.3f}")
m3.metric("Loudness", f"{feats['Loudness']:.1f} dB")
m4.metric("Dominant Note", dominant_note)
m5.metric("Duration", f"{mins}:{secs:02d}")

# ------------------------------------------------------------------
# Tabs with charts
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🌊 Waveform", "🎛️ Spectrogram", "🧬 MFCC", "🎹 Chroma", "📋 Feature Table"]
)

with tab1:
    step = max(1, len(y) // 3000)
    t = np.arange(0, len(y), step) / sr
    fig = go.Figure(go.Scatter(
        x=t, y=y[::step], mode="lines", fill="tozeroy",
        line=dict(color="#ffd166", width=1),
    ))
    fig.update_layout(
        title=f"Waveform (analysed {len(y)/sr:.0f}s clip from the middle of the song)",
        xaxis_title="Time (s)", yaxis_title="Amplitude", height=380, **PLOT_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure(go.Heatmap(z=mel[:, ::4], colorscale="Magma", colorbar=dict(title="dB")))
    fig.update_layout(
        title="Mel Spectrogram (energy across frequencies over time)",
        xaxis_title="Time frames", yaxis_title="Mel bands", height=400, **PLOT_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Heatmap(
            z=mfcc_mat[:, ::4], colorscale="Viridis", colorbar=dict(title="Value"),
        ))
        fig.update_layout(
            title="MFCC over time", xaxis_title="Time frames",
            yaxis_title="Coefficient", height=380, **PLOT_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        mfcc_mean = [feats[f"MFCC_{i}"] for i in range(1, 14)]
        fig = go.Figure(go.Bar(
            x=[f"MFCC {i}" for i in range(1, 14)], y=mfcc_mean,
            marker_color="#7f7fff",
        ))
        fig.update_layout(title="Average MFCC values", height=380, **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("MFCCs describe the timbre (tone colour) of the song.")

with tab4:
    colors = ["#38ef7d" if i == int(np.argmax(chroma_mean)) else "#ff6fb5" for i in range(12)]
    fig = go.Figure(go.Bar(x=NOTES, y=chroma_mean, marker_color=colors))
    fig.update_layout(
        title="Chroma: strength of each musical note", yaxis_title="Average strength",
        height=380, **PLOT_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"The most prominent note in this clip is **{dominant_note}**.")

with tab5:
    table = pd.DataFrame({"Feature": list(feats.keys()), "Value": list(feats.values())})
    table["Value"] = table["Value"].round(4)
    st.dataframe(table, use_container_width=True, height=450, hide_index=True)
    st.download_button(
        "⬇️ Download features as CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"{os.path.splitext(display_name)[0]}_features.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Built with Streamlit, librosa and scikit-learn • For educational use")
