import streamlit as st
import assemblyai as aai
import tempfile
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

# Download VADER lexicon if not already present
nltk.download("vader_lexicon")

# Directly set your API key here
aai.settings.api_key = "4c5a787a36634384b228b0531ebd8c5d"

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Audio Transcription & Sentiment Analysis",
    page_icon="🎙️",
    layout="wide"
)

# ------------------- Custom CSS -------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f9f9f9 0%, #e6f0ff 100%);
    font-family: "Segoe UI", sans-serif;
}
.title {
    text-align: center;
    color: #1e3a8a;
    font-size: 40px !important;
    font-weight: bold;
    margin-bottom: 20px;
}
.subtitle {
    text-align: center;
    font-size: 20px !important;
    color: #444;
    margin-bottom: 40px;
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

    try:
        with st.spinner("⏳ Transcribing audio... please wait"):
            # Configure transcription
            config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.universal)

            # Run transcription
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

            with col2:
                # ---- Sentiment Analysis ----
                st.subheader("📊 Sentiment Analysis")

                sia = SentimentIntensityAnalyzer()
                scores = sia.polarity_scores(transcript.text)
                compound = scores["compound"]

                if compound >= 0.05:
                    sentiment = "✅ Positive (No human harm)"
                    sentiment_color = "#d1fae5"  # green
                elif compound <= -0.05:
                    sentiment = "❌ Negative (Human life is in danger)"
                    sentiment_color = "#fee2e2"  # red
                else:
                    sentiment = "⚠️ Neutral (No life in danger, but needs attention)"
                    sentiment_color = "#fef9c3"  # yellow

                st.markdown(f"""
                <div style="background-color:{sentiment_color};
                            padding:15px;
                            border-radius:10px;
                            font-size:18px;">
                    <b>Overall Sentiment:</b> {sentiment}
                </div>
                """, unsafe_allow_html=True)

                st.json(scores)

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
