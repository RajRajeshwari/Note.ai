import streamlit as st
import requests
import base64
import tempfile
import os

# Function for Google Speech-to-Text API
def transcribe_audio_google(audio_path):
    with open(audio_path, "rb") as f:
        audio_content = base64.b64encode(f.read()).decode('utf-8')
    api_key = st.secrets["GOOGLE_API_KEY"]
    url = f"https://speech.googleapis.com/v1/speech:recognize?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": 16000,
            "languageCode": "en-US"
        },
        "audio": {"content": audio_content}
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        results = response.json().get("results", [])
        transcript = " ".join([res["alternatives"][0]["transcript"] for res in results])
        return transcript
    else:
        return f"Error: {response.text}"

# Function for MOM generation via Groq API
def generate_mom(text):
    groq_api_key = st.secrets["GROQ_API_KEY"]
    url = "https://api.groq.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    prompt = f"""
You are an expert meeting assistant. Generate:
1. Summary
2. Key discussion points
3. Decisions taken
4. Action items (with Owner, ETA, Comments)

Transcript:
{text}
"""
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.3
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {resp.text}"

# Streamlit Interface
st.title("Meeting Transcription & MOM (Google Cloud + Groq)")

mode = st.radio("Select input method:", ["Upload Audio", "Paste Text", "Upload Transcript File"])

transcript = ""

if mode == "Upload Audio":
    uploaded_audio = st.file_uploader("Upload audio file (wav/mp3)", type=["wav", "mp3"])
    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_audio.read())
            tmp_path = tmp_file.name
        st.info("Transcribing audio with Google Cloud...")
        transcript = transcribe_audio_google(tmp_path)
        os.remove(tmp_path)

elif mode == "Paste Text":
    transcript = st.text_area("Paste meeting transcript here:")

elif mode == "Upload Transcript File":
    uploaded_txt = st.file_uploader("Upload transcript file (.txt)", type=["txt"])
    if uploaded_txt:
        transcript = uploaded_txt.read().decode("utf-8")

# Show the transcript and MOM button
if transcript:
    st.subheader("Transcript")
    st.write(transcript)
    if st.button("Generate MOM"):
        result = generate_mom(transcript)
        st.subheader("Minutes of Meeting")
        st.text_area("MOM Output", result, height=400)
