import os
import tempfile
import streamlit as st
import soundfile as sf
import librosa

from yt_dlp import YoutubeDL
from moviepy.editor import VideoFileClip

import whisper
import whisper.tokenizer as tok
from speechbrain.pretrained import EncoderClassifier
import numpy as np
from audio_recorder_streamlit import audio_recorder

# ───────────────────────────────────────────────
# 1) Page config & Dark Theme Styling
# ───────────────────────────────────────────────
st.set_page_config(page_title="English & Accent Detector", page_icon="🎤", layout="wide")
st.markdown("""
    <style>
      body, .stApp { background-color: #121212; color: #e0e0e0; overflow-y: scroll; }
      .stButton>button {
        background-color: #1f77b4; color: #fff;
        border-radius:8px; padding:0.6em 1.2em; font-size:1rem;
      }
      .stButton>button:hover { background-color: #105b88; }
      .stVideo > video { max-width: 300px !important; border: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# 2) Load models once
# ───────────────────────────────────────────────
wmodel = whisper.load_model("tiny")
classifier = EncoderClassifier.from_hparams(
    source="Jzuluaga/accent-id-commonaccent_ecapa",
    savedir="pretrained_models/accent-id-commonaccent_ecapa"
)

# ───────────────────────────────────────────────
# 3) Accent grouping map
# ───────────────────────────────────────────────
GROUP_MAP = {
    "england": "British", "us": "American", "canada": "American",
    "australia": "Australian", "newzealand": "Australian",
    "indian": "Indian", "scotland": "Scottish", "ireland": "Irish",
    "wales": "Welsh", "african": "African", "malaysia": "Malaysian",
    "bermuda": "Bermudian", "philippines": "Philippine",
    "hongkong": "Hong Kong", "singapore": "Singaporean",
    "southatlandtic": "Other"
}
def group_accents(raw_list):
    return [(GROUP_MAP.get(r, r.capitalize()), p) for r, p in raw_list]

# ───────────────────────────────────────────────
# 4) Helper functions
# ───────────────────────────────────────────────
def download_extract_audio(url, out_vid="clip.mp4", out_wav="clip.wav",
                           max_duration=60, sr=16000):
    if os.path.exists(out_vid): os.remove(out_vid)
    with YoutubeDL({"outtmpl": out_vid, "merge_output_format": "mp4"}) as ydl:
        ydl.download([url])
    clip = VideoFileClip(out_vid)
    used = min(clip.duration, max_duration)
    sub = clip.subclip(0, used)
    sub.audio.write_audiofile(out_wav, fps=sr, codec="pcm_s16le")
    clip.close(); sub.close()
    wav, rate = librosa.load(out_wav, sr=sr, mono=True)
    return wav, rate, out_wav, out_vid

def detect_language_whisper(wav_path):
    audio = whisper.load_audio(wav_path, sr=16000)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(wmodel.device)
    _, probs = wmodel.detect_language(mel)
    lang = max(probs, key=probs.get)
    conf = probs.get("en", 0.0) * 100
    return lang, conf

def classify_clip_topk(wav_path, k=3):
    out_prob, _, _, _ = classifier.classify_file(wav_path)
    probs = out_prob.squeeze().cpu().numpy()
    idxs = probs.argsort()[-k:][::-1]
    return [(classifier.hparams.label_encoder.ind2lab[i], float(probs[i]))
            for i in idxs]

# ───────────────────────────────────────────────
# 5) Streamlit UI
# ───────────────────────────────────────────────
st.title("🎤 English & Accent Detector")
st.write("""
    This tool helps you determine if a speaker is speaking English and identifies their accent.
    
    🧭 **How to use:**
    - Use **URL** for public **YouTube**, **Loom**, or any **MP4-accessible video link**.
    - Use **Upload** to submit local video files (MP4, MOV, WEBM, MKV).
    - Use **Record** to record short audio snippets directly from your browser microphone.
    
    
""")

st.sidebar.header("📥 Input")
method = st.sidebar.radio("Input method", ["URL", "Upload", "Record"])

url = None
uploaded = None
audio_bytes = None

if method == "URL":
    url = st.sidebar.text_input("Video URL (e.g. YouTube, Loom, MP4 link)")
elif method == "Upload":
    uploaded = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "webm", "mkv"])
elif method == "Record":
    st.sidebar.write("🎙️ Click below to start recording (wait for microphone access prompt):")
    audio_bytes = audio_recorder()
    if not audio_bytes:
        st.sidebar.info("Waiting for you to record your voice...")
    else:
        st.sidebar.success("Audio recorded successfully! You can now classify it.")

if st.sidebar.button("Classify Accent"):
    with st.spinner("🔊 Extracting audio..."):
        if method == "URL" and url:
            wav, sr, wav_path, vid_path = download_extract_audio(url)
        elif method == "Upload" and uploaded:
            vid_path = tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(uploaded.name)[1], delete=False
            ).name
            with open(vid_path, "wb") as f:
                f.write(uploaded.read())
            clip = VideoFileClip(vid_path)
            wav_path = "clip.wav"
            clip.audio.write_audiofile(wav_path, fps=16000, codec="pcm_s16le")
            clip.close()
            wav, sr = librosa.load(wav_path, sr=16000, mono=True)
        elif method == "Record" and audio_bytes:
            wav_path = "recorded.wav"
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)
            wav, sr = librosa.load(wav_path, sr=16000, mono=True)
            vid_path = None
        else:
            st.error("Please supply a valid input.")
            st.stop()

    left, right = st.columns([1, 2])
    with left:
        st.subheader("📺 Preview")
        if method == "Record":
            st.audio(audio_bytes, format="audio/wav")
        elif vid_path:
            with open(vid_path, "rb") as f:
                st.video(f.read())

    with right:
        with st.spinner("🔎 Detecting English..."):
            lang_code, eng_conf = detect_language_whisper(wav_path)

        if eng_conf >= 4.0:
            st.markdown(
                "<div style='background-color:#1b5e20; color:#a5d6a7; padding:8px;"
                " border-radius:5px;'>✅ <strong>English detected – classifying accent...</strong></div>",
                unsafe_allow_html=True
            )
            with st.spinner("🎯 Classifying accent..."):
                raw3 = classify_clip_topk(wav_path, k=3)
            grouped = group_accents(raw3)

            st.subheader("🗣️ Accent Classification")
            cols = st.columns(len(grouped))
            for c, (lbl, p) in zip(cols, grouped):
                c.markdown(
                    f"""<div style=\"border:1px solid #444; border-radius:8px; padding:15px; text-align:center;\">
                          <div style=\"font-size:1.1em; font-weight:bold; color:#90caf9\">{lbl}</div>
                          <div style=\"font-size:1.8em; color:#29b6f6;\">{p*100:5.1f}%</div>
                        </div>""",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "<div style='background-color:#b71c1c; color:#ffcdd2; padding:8px;"
                " border-radius:5px;'>❌ <strong>English not detected</strong></div>",
                unsafe_allow_html=True
            )
            name = tok.LANGUAGES.get(lang_code, lang_code).capitalize()
            st.write(f"**Top detected language:** {name} ({eng_conf:.1f}% English)")

    for p in (wav_path, vid_path):
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except:
                pass




















