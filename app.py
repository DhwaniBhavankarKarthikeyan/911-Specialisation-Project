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
aai.settings.api_key = "YOUR_API_KEY"  # replace with your AssemblyAI key

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
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9 !important;
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
        color: #cbd5e1 !important;
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

    try:
        with st.spinner("⏳ Transcribing audio... please wait"):
            # Configure transcription
            config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.universal)
            transcriber = aai.Transcriber(config=config)
            transcript = transcriber.transcribe(temp_filename)

        if transcript.status == "error":
            st.error(f"❌ Transcription failed: {transcript.error}")
        else:
            st.success("✅ Transcription complete!")

            # ---- Layout: Two Columns ----
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📝 Transcribed Text")
                st.text_area("", transcript.text, height=400)

                # ---- Download Options ----
                if transcript.text.strip():
                    st.download_button(
                        label="📥 Download Transcript as TXT",
                        data=transcript.text,
                        file_name="transcript.txt",
                        mime="text/plain"
                    )

                    df = pd.DataFrame([{"Transcript": transcript.text}])
                    st.download_button(
                        label="📥 Download Transcript as CSV",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="transcript.csv",
                        mime="text/csv"
                    )

            with col2:
                # ---- Sentiment Analysis ----
                st.subheader("📊 Sentiment Analysis")
                sia = SentimentIntensityAnalyzer()
                scores = sia.polarity_scores(transcript.text)
                compound = scores["compound"]

                if compound >= 0.05:
                    sentiment = "✅ Positive (No human harm)"
                    sentiment_color = "#15803d" if st.session_state.dark_mode else "#d1fae5"
                elif compound <= -0.05:
                    sentiment = "❌ Negative (Human life is in danger)"
                    sentiment_color = "#b91c1c" if st.session_state.dark_mode else "#fee2e2"
                else:
                    sentiment = "⚠️ Neutral (No life in danger, but needs attention)"
                    sentiment_color = "#a16207" if st.session_state.dark_mode else "#fef9c3"

                st.markdown(f"""
                <div style="background-color:{sentiment_color};
                            padding:15px;
                            border-radius:10px;
                            font-size:18px;
                            color:white if st.session_state.dark_mode else black;">
                    <b>Overall Sentiment:</b> {sentiment}
                </div>
                """, unsafe_allow_html=True)

                # ---- Word Cloud ----
                st.subheader("☁️ Word Cloud")
                if transcript.text.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(transcript.text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Clean up temp file
    os.remove(temp_filename)
