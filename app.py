import streamlit as st
import speech_recognition as sr
from transformers import pipeline
import tempfile

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="🎤 Audio Transcription & Sentiment", layout="centered")

# -------------------------------
# Dark/Light Theme CSS
# -------------------------------
def set_custom_css(dark_mode=False):
    if dark_mode:
        st.markdown("""
        <style>
        .stApp { background-color: #1E1E1E; color: #FFFFFF; }
        .stButton>button {
            background-color: #3A3A3A;
            color: #FFFFFF;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton>button:hover { background-color: #575757; }
        .stDownloadButton>button {
            background-color: #3A3A3A;
            color: #FFFFFF;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stDownloadButton>button:hover { background-color: #575757; }
        audio { filter: invert(1); }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp { background-color: #FAFAFA; color: #000000; }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton>button:hover { background-color: #45a049; }
        .stDownloadButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stDownloadButton>button:hover { background-color: #45a049; }
        </style>
        """, unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ Settings")
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)
set_custom_css(dark_mode)

# -------------------------------
# Header
# -------------------------------
st.title("🎤 Audio Transcription & Sentiment Analysis")
st.markdown("Upload an audio file, play it, transcribe, and check overall sentiment.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file:
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Audio Playback
    st.audio(file_path)

    # -------------------------------
    # Transcription
    # -------------------------------
    if st.button("Transcribe"):
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                st.subheader("📝 Transcription:")
                st.success(text)

                # -------------------------------
                # Sentiment
                # -------------------------------
                sentiment_pipeline = pipeline("sentiment-analysis")
                result = sentiment_pipeline(text)[0]
                label = result["label"]

                if label == "POSITIVE":
                    sentiment_label = "✅ Positive"
                elif label == "NEGATIVE":
                    sentiment_label = "❌ Negative"
                else:
                    sentiment_label = "😐 Neutral"

                st.subheader("📊 Overall Sentiment:")
                st.info(sentiment_label)

                # -------------------------------
                # Download Transcript
                # -------------------------------
                st.download_button(
                    label="📥 Download Transcript",
                    data=text,
                    file_name="transcript.txt",
                    mime="text/plain"
                )

            except sr.UnknownValueError:
                st.error("❌ Could not understand the audio")
            except sr.RequestError:
                st.error("⚠️ Error connecting to the speech recognition service")
