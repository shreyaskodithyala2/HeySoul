import os, sys, ffmpeg, tempfile, time
import numpy as np, soundfile as sf, librosa
from tqdm import tqdm
import whisper
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from collections import deque

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
STRIDE_SECONDS = 2
WINDOW_SIZE = 5          # sliding window (chunks)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------------------------------------
# STEP 1: AUDIO EXTRACTION
# ------------------------------------------------------------
def extract_audio(video_path, out_path="temp.wav"):
    (
        ffmpeg.input(video_path)
        .output(out_path, ac=1, ar=SAMPLE_RATE, format="wav", loglevel="error")
        .overwrite_output()
        .run()
    )
    return out_path

# ------------------------------------------------------------
# STEP 2: SPLIT AUDIO
# ------------------------------------------------------------
def chunk_audio(audio, sr, chunk_seconds=CHUNK_SECONDS, stride_seconds=STRIDE_SECONDS):
    chunk_len = int(chunk_seconds * sr)
    stride = int(stride_seconds * sr)
    total = len(audio)
    for start in range(0, total, stride):
        end = start + chunk_len
        if end > total: break
        yield audio[start:end], start / sr, end / sr

# ------------------------------------------------------------
# STEP 3: FEATURE EXTRACTION
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
    if "pos" in s_label: s_label_val = 1
    elif "neg" in s_label: s_label_val = -1
    else: s_label_val = 0

    return dict(rms=rms, pitch=pitch, speech_rate=speech_rate,
                pause_ratio=pause_ratio, sentiment_score=s_score,
                sentiment_label=s_label_val)

# ------------------------------------------------------------
# STEP 4: TEXT EMBEDDING
# ------------------------------------------------------------
print("🧠 Loading text embedding model (SentenceTransformer)...")
text_model = SentenceTransformer("all-MiniLM-L6-v2")
dummy = text_model.encode(["happy","sad","angry","tired","neutral"])
pca = PCA(n_components=2).fit(dummy)

def text_embedding(text):
    if not text: return [0.,0.]
    emb = text_model.encode(text)
    return pca.transform(emb.reshape(1,-1))[0].tolist()

# ------------------------------------------------------------
# STEP 5: LOAD MODELS
# ------------------------------------------------------------
def load_models():
    print("📦 Loading Whisper + Sentiment models...")
    w = whisper.load_model("base")
    s = pipeline("sentiment-analysis",
                 model="distilbert-base-uncased-finetuned-sst-2-english")
    return w, s

def transcribe_chunk(model, chunk, sr):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, chunk, sr)
        result = model.transcribe(f.name, fp16=False)
        os.remove(f.name)
    return result.get("text", "").strip()

# ------------------------------------------------------------
# STEP 6: RULE-BASED EMOTION ENGINE
# ------------------------------------------------------------
def detect_danger_words(text):
    danger = ["kill", "suicide", "die", "hurt myself", "end my life"]
    if not text: return 0
    return int(any(w in text.lower() for w in danger))

def compute_rule_score(feat, text_vec, danger_flag):
    # Weighted fusion
    score = (
        2*feat["sentiment_score"]
        + 3*(feat["rms"]-0.05)
        + 1.5*(feat["speech_rate"]-1.0)
        - 2*(feat["pause_ratio"]-0.2)
        - 5*danger_flag
    )
    valence = feat["sentiment_score"]
    arousal = (feat["rms"]*5 + feat["speech_rate"]*2 - feat["pause_ratio"]*2)
    return score, valence, arousal

def map_emotion(valence, arousal):
    if valence > 0.4 and arousal > 0.5:  return "Excited/Happy"
    if valence > 0.4 and arousal <=0.5:  return "Calm/Content"
    if valence < -0.4 and arousal > 0.5: return "Angry/Anxious"
    if valence < -0.4 and arousal <=0.5: return "Sad/Withdrawn"
    return "Neutral"

# ------------------------------------------------------------
# STEP 7: MAIN (SIMULATED REAL-TIME)
# ------------------------------------------------------------
def main(video_path):
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}"); sys.exit(1)

    print(f"🎥 Extracting audio from {video_path} ...")
    wav_path = "temp_audio.wav"
    extract_audio(video_path, wav_path)
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

    whisper_model, sentiment_pipe = load_models()
    window_scores = deque(maxlen=WINDOW_SIZE)

    print("\n🎧 Real-Time Rule-Based Emotion Analysis\n")
    for chunk, start, end in tqdm(chunk_audio(audio, sr)):
        # time.sleep(CHUNK_SECONDS)  # mimic live
        text = transcribe_chunk(whisper_model, chunk, sr)
        sentiment = sentiment_pipe(text[:512])[0] if text else {"label":"neutral","score":0.0}

        feat = compute_audio_features(chunk, sr, text, sentiment)
        txt_vec = text_embedding(text)
        danger_flag = detect_danger_words(text)
        score, valence, arousal = compute_rule_score(feat, txt_vec, danger_flag)
        emotion = map_emotion(valence, arousal)

        window_scores.append(score)
        avg_score = np.mean(window_scores)
        trend = "GOOD" if avg_score>1.5 else "BAD" if avg_score<-1.5 else "NORMAL"

        print(f"🕒 {start:.1f}-{end:.1f}s")
        print(f"  Text: {text if text else '[no speech detected]'}")
        print(f"  Sentiment: {sentiment['label']} ({sentiment['score']:.2f}) | RMS:{feat['rms']:.3f} | Rate:{feat['speech_rate']:.2f}")
        print(f"  → Instant Emotion: {emotion}")
        print(f"  → Sliding-Window Trend (5 chunks): {trend} (avg_score={avg_score:.2f})")
        print("-"*70)

    print("✅ Analysis complete.\n")

# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python analyze_rule_based_emotions.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])
