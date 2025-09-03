import streamlit as st
import assemblyai as aai
import tempfile
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import pandas as pd

# Download VADER lexicon if not already present
nltk.download("vader_lexicon")

# Directly set your API key here
aai.settings.api_key = "YOUR_API_KEY"

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Audio Transcription & Sentiment Analysis",
    page_icon="🎙️",
    layout="wide"
)

# ------------------- Dark Mode Toggle -------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = toggle

# ------------------- Custom CSS -------------------
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #000000 100%);
        color: #f5f5f5 !important;
        font-family: "Segoe UI", sans-serif;
    }
    .title {
        text-align: center;
        color: #60a5fa !important;
        font-size: 40px !important;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px !important;
        color: #d1d5db !important;
        margin-bottom: 40px;
    }
    .stDownloadButton button, .stButton button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 18px !important;
        font-weight: bold !important;
        transition: 0.3s ease-in-out;
    }
    .stDownloadButton button:hover, .stButton button:hover {
        background-color: #1d4ed8 !important;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f9f9f9 0%, #e6f0ff 100%);
        color: #1e293b !important;
        font-family: "Segoe UI", sans-serif;
    }
    .title {
        text-align: center;
        color: #1e3a8a !important;
        font-size: 40px !important;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px !important;
        color: #444 !important;
        margin-bottom: 40px;
    }
    .stDownloadButton button, .stButton button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 18px !important;
        font-weight: bold !important;
        transition: 0.3s ease-in-out;
    }
    .stDownloadButton button:hover, .stButton button:hover {
        background-color: #1d4ed8 !important;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------- App Header -------------------
st.markdown('<h1 class="title">🎙️ Audio Transcription & Sentiment Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload audio → Get transcription → Sentiment insights → Word Cloud</p>', unsafe_allow_html=True)

# ------------------- File Uploader -------------------
uploaded_file = st.file_uploader("📂 Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_filename = tmp_file.name

    # ---- Audio Playback ----
    st.subheader("▶️ Play Uploaded Audio")
    st.audio(temp_filename, format="audio/mp3")

    # (rest of your transcription + sentiment + wordcloud code stays the same)
