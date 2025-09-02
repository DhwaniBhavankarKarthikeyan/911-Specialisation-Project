import streamlit as st
import assemblyai as aai
import tempfile
import os

# ⚠️ Important: Don't hardcode API keys when deploying publicly.
# Use Streamlit secrets instead: st.secrets["ASSEMBLYAI_API_KEY"]
aai.settings.api_key = "4c5a787a36634384b228b0531ebd8c5d"

# App Title
st.title("🎙️ Audio Transcription App")
st.write("Upload an audio file and get the transcription using AssemblyAI.")

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
            st.text_area("Transcribed Text", transcript.text, height=300)

    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Clean up temp file
    os.remove(temp_filename)
