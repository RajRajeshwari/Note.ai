import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import whisper
import tempfile
import queue
import threading
import requests
import av
import numpy as np

# Load Whisper model once
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

model = load_whisper()

# Queue to hold recorded audio frames
audio_queue = queue.Queue()

# Audio processor for streamlit-webrtc
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recording = True

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        if self.recording:
            pcm = frame.to_ndarray(format="s16")
            audio_queue.put(pcm.tobytes())
        return frame

def save_audio_from_queue(filename: str):
    import wave

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bit = 2 bytes
        wf.setframerate(48000)
        while True:
            try:
                data = audio_queue.get(timeout=5)
                if data == b"STOP":
                    break
                wf.writeframes(data)
            except queue.Empty:
                break

def transcribe_audio_file(filename: str):
    result = model.transcribe(filename)
    return result["text"]

def generate_mom(text: str) -> str:
    prompt = f"""
You are an expert meeting assistant. Given the meeting transcript below, generate:

1. Summary
2. Key Discussion Points (bullet list)
3. Decisions made (bullet list)
4. Action Items Table (with columns Action Item | Owner | ETA | Comments - leave owner, ETA, comments blank)

Meeting Transcript:
{text}
"""
    # Replace this with your open-source LLM API endpoint
    #Get grok API key from Secrets
    groq_api_key = st.secrets.get("GROQ_API_KEY")

    # Example does not require API key, adjust if yours does
    response = requests.post(groq_api_key, json={"inputs": prompt, "parameters": {"max_new_tokens": 500}})
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "Failed to generate MOM. Check your LLM API."

st.title("Versatile Meeting MOM Generator with Live Transcription")

st.markdown("""
**Select input mode:**

- Live Transcription via Microphone (default, listens live)
- Paste Transcript Text
- Upload Transcript Text File
- Upload Audio File
""")

input_mode = st.radio("Input mode", ["Live Transcription", "Paste Transcript Text", "Upload Transcript Text File", "Upload Audio File"])

transcript_text = ""

if input_mode == "Live Transcription":

    st.write("Click Start to begin live transcription and Stop to end and generate MOM.")
    audio_processor = AudioProcessor()
    ctx = webrtc_streamer(
        key="live-transcription",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=lambda: audio_processor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if st.button("Stop Recording and Transcribe & Generate MOM"):
        audio_processor.recording = False
        audio_queue.put(b"STOP")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_path = f.name
        save_audio_from_queue(wav_path)
        st.success("Audio recorded, now transcribing...")
        transcript_text = transcribe_audio_file(wav_path)
        st.subheader("Transcript")
        st.write(transcript_text)

        st.subheader("Generating Minutes of Meeting...")
        mom = generate_mom(transcript_text)
        st.text_area("Minutes of Meeting", mom, height=400)

elif input_mode == "Paste Transcript Text":
    transcript_text = st.text_area("Paste your meeting transcript here:")
    if transcript_text and st.button("Generate MOM"):
        mom = generate_mom(transcript_text)
        st.subheader("Minutes of Meeting")
        st.text_area("MOM Output", mom, height=400)

elif input_mode == "Upload Transcript Text File":
    uploaded_file = st.file_uploader("Upload transcript text file (txt only)", type=["txt"])
    if uploaded_file:
        transcript_text = uploaded_file.read().decode("utf-8")
        st.text_area("Loaded Transcript Text", transcript_text, height=200)
        if st.button("Generate MOM"):
            mom = generate_mom(transcript_text)
            st.subheader("Minutes of Meeting")
            st.text_area("MOM Output", mom, height=400)

elif input_mode == "Upload Audio File":
    audio_file = st.file_uploader("Upload audio file (mp3, wav)", type=["mp3", "wav"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        with st.spinner("Transcribing audio..."):
            transcript_text = transcribe_audio_file(tmp_path)
        st.subheader("Transcript from Audio")
        st.write(transcript_text)
        if st.button("Generate MOM"):
            mom = generate_mom(transcript_text)
            st.subheader("Minutes of Meeting")
            st.text_area("MOM Output", mom, height=400)

st.markdown("""
--------

Use this app to transcribe live meeting audio or upload/paste transcripts and get automated Minutes of Meeting (summary, discussions, decisions, action items).

---

**Instructions:**

- For live transcription: click 'Start' to allow microphone and speak. When done, click 'Stop Recording...' to transcribe and generate MOM.
- For paste/upload: simply paste or upload and generate MOM.

---
""")
