import os
import sys
import ffmpeg
import librosa
import numpy as np
import soundfile as sf
import whisper
from transformers import pipeline
from tqdm import tqdm
import tempfile

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
STRIDE_SECONDS = 2


# ------------------------------------------------------------
# STEP 1: Extract Audio from Video
# ------------------------------------------------------------
def extract_audio(video_path, out_path="temp.wav"):
    """Extract audio as mono 16kHz WAV."""
    (
        ffmpeg
        .input(video_path)
        .output(out_path, ac=1, ar=SAMPLE_RATE, format="wav", loglevel="error")
        .overwrite_output()
        .run()
    )
    return out_path


# ------------------------------------------------------------
# STEP 2: Split Audio into Chunks
# ------------------------------------------------------------
def chunk_audio(audio, sr, chunk_seconds=CHUNK_SECONDS, stride_seconds=STRIDE_SECONDS):
    chunk_len = int(chunk_seconds * sr)
    stride = int(stride_seconds * sr)
    total = len(audio)
    for start in range(0, total, stride):
        end = start + chunk_len
        if end > total:
            break
        yield audio[start:end], start / sr, end / sr


# ------------------------------------------------------------
# STEP 3: Compute Loudness + Pitch
# ------------------------------------------------------------
def compute_acoustics(chunk, sr):
    rms = np.mean(librosa.feature.rms(y=chunk))
    try:
        f0 = librosa.yin(chunk, fmin=50, fmax=500, sr=sr)
        f0 = f0[~np.isnan(f0)]
        pitch = np.mean(f0) if f0.size > 0 else 0
    except Exception:
        pitch = 0
    return float(rms), float(pitch)


# ------------------------------------------------------------
# STEP 4: Transcribe & Sentiment
# ------------------------------------------------------------
def load_models():
    print("📦 Loading Whisper + BERT models (first time may take a bit)...")
    whisper_model = whisper.load_model("base")
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return whisper_model, sentiment_pipe


def transcribe_chunk(model, chunk, sr):
    """Transcribe a numpy chunk using Whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, chunk, sr)
        result = model.transcribe(f.name, fp16=False)
        os.remove(f.name)
    return result.get("text", "").strip()


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def main(video_path):
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)

    # 1. Extract audio
    print(f"🎥 Extracting audio from {video_path} ...")
    wav_path = "temp_audio.wav"
    extract_audio(video_path, wav_path)

    # 2. Load audio
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

    # 3. Load models
    whisper_model, sentiment_pipe = load_models()

    # 4. Process in chunks
    print("\n🎧 Transcribing and analyzing...\n")
    for chunk, start, end in tqdm(chunk_audio(audio, sr)):
        # Acoustic features
        rms, pitch = compute_acoustics(chunk, sr)
        # Transcription
        text = transcribe_chunk(whisper_model, chunk, sr)
        # Sentiment
        sent = sentiment_pipe(text[:512])[0] if text else {"label": "neutral", "score": 0.0}

        print(f"🕒 {start:.1f}s–{end:.1f}s")
        print(f"  Text: {text if text else '[no speech detected]'}")
        print(f"  Loudness (RMS): {rms:.4f}")
        print(f"  Pitch (Hz): {pitch:.2f}")
        print(f"  Sentiment: {sent['label'].upper()} (score={sent['score']:.2f})")
        print("-" * 60)

    print("✅ Done! Analysis complete.")


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_audio.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])
