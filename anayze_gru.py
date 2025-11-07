import os
import sys
import ffmpeg
import librosa
import numpy as np
import soundfile as sf
import whisper
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tqdm import tqdm
import tempfile
import torch
import torch.nn as nn
import time

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
STRIDE_SECONDS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# GRU MODEL
# ------------------------------------------------------------
class EmotionGRU(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        out = self.fc(out[:, -1, :])
        return self.softmax(out), hidden

gru_model = EmotionGRU().to(DEVICE)
gru_model.eval()

# ------------------------------------------------------------
# STEP 1: Extract Audio
# ------------------------------------------------------------
def extract_audio(video_path, out_path="temp.wav"):
    (
        ffmpeg
        .input(video_path)
        .output(out_path, ac=1, ar=SAMPLE_RATE, format="wav", loglevel="error")
        .overwrite_output()
        .run()
    )
    return out_path

# ------------------------------------------------------------
# STEP 2: Split Audio
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
# STEP 3: Audio Features
# ------------------------------------------------------------
def compute_audio_features(chunk, sr, text, sentiment):
    rms = np.mean(librosa.feature.rms(y=chunk))
    try:
        f0 = librosa.yin(chunk, fmin=50, fmax=500, sr=sr)
        f0 = f0[~np.isnan(f0)]
        pitch = np.mean(f0) if f0.size > 0 else 0
    except Exception:
        pitch = 0

    energy = librosa.feature.rms(y=chunk).flatten()
    pause_ratio = np.sum(energy < 0.005) / len(energy)
    words = len(text.split()) if text else 0
    speech_rate = words / CHUNK_SECONDS

    s_score = float(sentiment.get("score", 0.0))
    s_label = sentiment.get("label", "").lower()
    if "pos" in s_label:
        s_label_val = 1
    elif "neg" in s_label:
        s_label_val = -1
    else:
        s_label_val = 0

    return [rms, pitch, speech_rate, pause_ratio, s_score, s_label_val]

# ------------------------------------------------------------
# STEP 4: Text Embedding (Sentence Transformer + PCA)
# ------------------------------------------------------------
print("🧠 Loading text embedding model (SentenceTransformer)...")
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-fit PCA once on a few random embeddings (for simplicity)
dummy_sentences = ["hello", "I am okay", "I feel bad", "today is great", "nothing to say"]
embeddings = text_model.encode(dummy_sentences)
pca = PCA(n_components=2)
pca.fit(embeddings)

def compute_text_embedding_features(text):
    if not text:
        return [0.0, 0.0]
    emb = text_model.encode(text)
    reduced = pca.transform(emb.reshape(1, -1))[0]
    return reduced.tolist()

# ------------------------------------------------------------
# STEP 5: Models (Whisper + Sentiment)
# ------------------------------------------------------------
def load_models():
    print("📦 Loading Whisper + Sentiment models...")
    whisper_model = whisper.load_model("base")
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return whisper_model, sentiment_pipe

def transcribe_chunk(model, chunk, sr):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, chunk, sr)
        result = model.transcribe(f.name, fp16=False)
        os.remove(f.name)
    return result.get("text", "").strip()

# ------------------------------------------------------------
# STEP 6: Real-Time Simulation
# ------------------------------------------------------------
def main(video_path):
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)

    print(f"🎥 Extracting audio from {video_path} ...")
    wav_path = "temp_audio.wav"
    extract_audio(video_path, wav_path)
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

    whisper_model, sentiment_pipe = load_models()

    hidden_state = None
    print("\n🎧 Simulated Real-Time Analysis Starting...\n")

    for chunk, start, end in chunk_audio(audio, sr):
        # simulate waiting for real-time input (optional)
        # time.sleep(CHUNK_SECONDS)

        text = transcribe_chunk(whisper_model, chunk, sr)
        sentiment = sentiment_pipe(text[:512])[0] if text else {"label": "neutral", "score": 0.0}

        audio_feats = compute_audio_features(chunk, sr, text, sentiment)
        text_feats = compute_text_embedding_features(text)

        fused_features = np.concatenate([audio_feats, text_feats])  # total 8 features

        # run GRU incrementally
        x = torch.tensor(fused_features, dtype=torch.float32).view(1, 1, -1).to(DEVICE)
        with torch.no_grad():
            probs, hidden_state = gru_model(x, hidden_state)
            probs = probs[0].cpu().numpy()

        classes = ["BAD", "NORMAL", "GOOD"]
        pred_idx = int(np.argmax(probs))
        confidence = probs[pred_idx]

        print(f"🕒 {start:.1f}s–{end:.1f}s")
        print(f"  Text: {text if text else '[no speech detected]'}")
        print(f"  Audio Features: {np.round(audio_feats,3)} | Text Embedding: {np.round(text_feats,3)}")
        print(f"  Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")
        print(f"  → Real-Time Mood: {classes[pred_idx]} (confidence={confidence:.2f})")
        print("-" * 60)

    print("✅ Real-Time Simulation Complete.\n")

# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_realtime_fusion.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])
