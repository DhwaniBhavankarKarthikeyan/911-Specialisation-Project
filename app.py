import streamlit as st
import assemblyai as aai
import tempfile
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from transformers import pipeline

# Download VADER lexicon if not already present
nltk.download("vader_lexicon")

# Direct API key
aai.settings.api_key = "4c5a787a36634384b228b0531ebd8c5d"

# App Title
st.title("🎙️ Audio Transcription, Sentiment & Summary")
st.write("Upload an audio file → Get transcription → Sentiment → Word Cloud → Summary")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_filename = tmp_file.name

    st.info("Transcribing... please wait ⏳")

    try:
        # Configure transcription
        config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.universal)

        # Run transcription
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(temp_filename)

        if transcript.status == "error":
            st.error(f"❌ Transcription failed: {transcript.error}")
        else:
            st.success("✅ Transcription complete!")
            st.text_area("📝 Transcribed Text", transcript.text, height=300)

            # ---- Sentiment Analysis ----
            st.subheader("📊 Sentiment Analysis")

            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(transcript.text)
            compound = scores["compound"]

            if compound >= 0.05:
                sentiment = "✅ Positive (No human harm)"
            elif compound <= -0.05:
                sentiment = "❌ Negative (Human life is in danger)"
            else:
                sentiment = "⚠️ Neutral (No life in danger, but needs attention)"

            st.write(f"**Overall Sentiment:** {sentiment}")
            st.json(scores)

            # ---- Word Cloud ----
            st.subheader("☁️ Word Cloud")
            if transcript.text.strip():
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(transcript.text)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

            # ---- Summarization ----
            st.subheader("📰 Situation Summary (2–3 lines)")
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

            try:
                summary = summarizer(transcript.text, max_length=60, min_length=20, do_sample=False)
                st.write(summary[0]['summary_text'])
            except Exception as e:
                st.warning(f"Summarization skipped: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Clean up temp file
    os.remove(temp_filename)
