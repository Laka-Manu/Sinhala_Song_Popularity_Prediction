\# 🎵 Sinhala Song Popularity Prediction



Predict whether a Sinhala song is a \*\*Hit\*\* or \*\*Non-Hit\*\* using only its audio.

A song with \*\*1,000,000+ YouTube views\*\* is labelled \*\*Hit\*\*, otherwise \*\*Non-Hit\*\*.



\## 🔗 Live App

\*\*\[👉 Open the Sinhala Song Popularity Prediction](http://localhost:8501/)\*\*



\## 📌 Overview

1\. Collect song names and YouTube links in a CSV file

2\. Extract artist, upload year and view count with \*\*yt-dlp\*\*

3\. Download the audio and extract features with \*\*librosa\*\*

4\. Label each song as Hit / Non-Hit using the 1M views rule

5\. Train a \*\*Random Forest\*\* classifier

6\. Predict new songs through a \*\*Streamlit\*\* web app



\## ✨ Features

\- Upload an audio file (MP3, WAV, M4A, FLAC, OGG) or paste a YouTube link

\- Hit / Non-Hit prediction with a confidence score and probability gauge

\- Audio details: tempo (BPM), energy, loudness, dominant note, duration

\- Interactive charts: waveform, mel spectrogram, MFCC, chroma

\- Download the extracted features as CSV



\## 🎼 Audio Features Used

| Feature | Description |

|---|---|

| Tempo | Speed of the song in BPM |

| Energy | Mean RMS energy |

| Loudness | Mean RMS level in dB |

| MFCC (13) | Timbre / tone colour of the sound |

| Chroma (12) | Strength of each musical note (C to B) |



\## 🗂️ Project Structure

| File | Purpose |

|---|---|

| `extract.py` | Gets Artist, Year and YouTube views from the song links |

| `features.py` | Downloads audio, extracts librosa features, adds the label |

| `predict\_hit.py` | Trains the model and predicts from the command line |

| `app.py` | Streamlit web app |

| `dataset\_features.csv` | Final dataset used for training |

| `hit\_model.pkl` | Trained model |

| `requirements.txt` | Python packages |

| `packages.txt` | System packages for Streamlit Cloud (ffmpeg) |



\## 🛠️ Tech Stack

Python, pandas, librosa, scikit-learn, yt-dlp, Streamlit, Plotly



\## 🚀 Run Locally

1\. Install \*\*ffmpeg\*\* on your computer.

2\. Install the packages:



&#x20;      pip install -r requirements.txt



3\. Train the model (only if `hit\_model.pkl` is missing):



&#x20;      python predict\_hit.py train



4\. Start the app:



&#x20;      streamlit run app.py





