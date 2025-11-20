import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import whisper
import tempfile
import queue
import threading
import requests
import av  # Audio processing library used by streamlit-webrtc

# -----------------------------------
# Load Whisper Model Once (cached for performance)
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")  # small and efficient

model = load_whisper()

# -----------------------------------
# Queue to hold audio frames coming from mic
audio_queue = queue.Queue()

# -----------------------------------
# Audio processor class to capture audio chunks
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recording = True

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Called for every audio frame from mic
        if self.recording:
            # Convert audio frame to raw bytes and put into queue
            pcm = frame.to_ndarray(format="s16")
            audio_queue.put(pcm.tobytes())
        return frame

# -----------------------------------
# Function to save all audio chunks from queue as wav file
def save_audio_from_queue(filename: str):
    import wave
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # mono audio
        wf.setsampwidth(2)  # 16 bit audio (2 bytes)
        wf.setframerate(48000)  # Sampling rate expected by streamlit-webrtc
        while True:
            try:
                data = audio_queue.get(timeout=5)
                if data == b"STOP":
                    break
                wf.writeframes(data)
            except queue.Empty:
                break

# -----------------------------------
# Transcribe audio file using Whisper
def transcribe_audio_file(filename: str) -> str:
    result = model.transcribe(filename)
    return result["text"]

# -----------------------------------
# Generate MOM by calling Groq API
def generate_mom(text: str) -> str:
    prompt = f"""
You are an expert meeting assistant. Based on the meeting transcript below, generate:

1. Summary
2. Key Discussion Points (bullet list)
3. Decisions made (bullet list)
4. Action Items Table (Action Item | Owner | ETA | Comments - leave Owner, ETA, Comments blank)

Meeting Transcript:
{text}
"""

    url = "https://api.groq.com/openai/v1"
    groq_api_key = st.secrets.get("GROQ_API_KEY")

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mixtral-8x7b-32768",  # Update based on Groq docs or your plan
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Failed to generate MOM: {response.status_code} - {response.text}"

# -----------------------------------
# Streamlit UI starts here
st.title("Versatile Meeting MOM Generator with Live Transcription")

st.markdown("""
This app can transcribe live meeting audio from your microphone using Whisper and generate a professional Minutes of Meeting (MOM) using Groq’s LLM.

Also supports pasting transcripts or uploading audio/text files.
""")

input_mode = st.radio("Select input mode", ["Live Transcription", "Paste Transcript Text", "Upload Transcript Text File", "Upload Audio File"])

transcript_text = ""

if input_mode == "Live Transcription":
    st.write("Click Start to begin live transcription and Stop to end and generate MOM.")
    audio_processor = AudioProcessor()
    webrtc_ctx = webrtc_streamer(
        key="live-transcription",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=lambda: audio_processor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if st.button("Stop Recording and Generate MOM"):
        audio_processor.recording = False
        audio_queue.put(b"STOP")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name

        save_audio_from_queue(wav_path)
        st.success("Audio recorded. Transcribing...")

        transcript_text = transcribe_audio_file(wav_path)
        st.subheader("Transcript")
        st.write(transcript_text)

        st.subheader("Generating Minutes of Meeting...")
        mom_text = generate_mom(transcript_text)
        st.text_area("Minutes of Meeting", mom_text, height=400)

elif input_mode == "Paste Transcript Text":
    transcript_text = st.text_area("Paste your transcript here")
    if transcript_text and st.button("Generate MOM"):
        mom_text = generate_mom(transcript_text)
        st.subheader("Minutes of Meeting")
        st.text_area("MOM Output", mom_text, height=400)

elif input_mode == "Upload Transcript Text File":
    uploaded_file = st.file_uploader("Upload transcript text file (.txt only)", type=["txt"])
    if uploaded_file:
        transcript_text = uploaded_file.read().decode("utf-8")
        st.text_area("Loaded Transcript", transcript_text, height=200)
        if st.button("Generate MOM"):
            mom_text = generate_mom(transcript_text)
            st.subheader("Minutes of Meeting")
            st.text_area("MOM Output", mom_text, height=400)

elif input_mode == "Upload Audio File":
    audio_file = st.file_uploader("Upload audio file (mp3, wav)", type=["mp3", "wav"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            audio_path = tmp.name
        with st.spinner("Transcribing audio..."):
            transcript_text = transcribe_audio_file(audio_path)
        st.subheader("Transcript")
        st.write(transcript_text)
        if st.button("Generate MOM"):
            mom_text = generate_mom(transcript_text)
            st.subheader("Minutes of Meeting")
            st.text_area("MOM Output", mom_text, height=400)
