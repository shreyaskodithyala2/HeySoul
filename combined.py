#!/usr/bin/env python3
"""
Comprehensive Real-time Activity Recognition with 4 Canvases + Full CSV Logging
All metrics printed to terminal and CSV for every frame
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from fer import FER
import os
os.environ['YOLO_VERBOSE'] = 'False'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load
from ultralytics import YOLO
import threading
from queue import Queue
import csv
import librosa
import soundfile as sf
import tempfile
import whisper
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

class AudioEmotionAnalyzer:
    """Real-time audio emotion analysis"""
    
    def __init__(self):
        import threading  # Add this for the audio processing
        print("Loading audio emotion models...")
        self.whisper_model = whisper.load_model("base")
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        print("Loading SentenceTransformer...")
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Settings
        self.sample_rate = 16000
        self.chunk_seconds = 3
        self.chunk_size = int(self.chunk_seconds * self.sample_rate)  # 48000 samples
        
        # Audio buffer for accumulation
        self.audio_buffer = []
        self.processing = False
        
        # Current state
        self.current_emotion = "Neutral"
        self.current_text = ""
        self.sentiment_label = "NEUTRAL"
        self.sentiment_score = 0.0
        self.rms = 0.0
        self.pitch = 0.0
        self.speech_rate = 0.0
        self.pause_ratio = 0.0
        self.valence = 0.0
        self.arousal = 0.0
        self.danger_flag = 0
        
        # Sliding window for trend
        self.score_window = deque(maxlen=5)
        self.window_trend = "NORMAL"
        
        print("Audio emotion models loaded!")
    
    def detect_danger_words(self, text):
        danger = ["kill", "suicide", "die", "hurt myself", "end my life", "depressed"]
        if not text: return 0
        return int(any(w in text.lower() for w in danger))
    
    def compute_audio_features(self, chunk, text, sentiment):
        try:
            # RMS
            self.rms = float(np.mean(librosa.feature.rms(y=chunk)))
            
            # Pitch
            try:
                f0 = librosa.yin(chunk, fmin=50, fmax=500, sr=self.sample_rate)
                f0 = f0[~np.isnan(f0)]
                self.pitch = float(np.mean(f0)) if f0.size > 0 else 0.0
            except:
                self.pitch = 0.0
            
            # Speech rate
            energy = librosa.feature.rms(y=chunk).flatten()
            self.pause_ratio = float(np.sum(energy < 0.005) / len(energy)) if len(energy) > 0 else 0.0
            words = len(text.split()) if text else 0
            self.speech_rate = float(words / self.chunk_seconds)
            
            # Sentiment
            self.sentiment_score = float(sentiment.get("score", 0.0))
            s_label = sentiment.get("label", "").lower()
            
            if "pos" in s_label:
                s_label_val = 1
            elif "neg" in s_label:
                s_label_val = -1
            else:
                s_label_val = 0
            
            return {
                'rms': self.rms,
                'pitch': self.pitch,
                'speech_rate': self.speech_rate,
                'pause_ratio': self.pause_ratio,
                'sentiment_score': self.sentiment_score,
                'sentiment_label': s_label_val
            }
        except Exception as e:
            return {
                'rms': 0, 'pitch': 0, 'speech_rate': 0,
                'pause_ratio': 0, 'sentiment_score': 0, 'sentiment_label': 0
            }
    
    def compute_emotion(self, feat, text, danger_flag):
        # Weighted fusion
        score = (
            2 * feat["sentiment_score"] +
            3 * (feat["rms"] - 0.05) +
            1.5 * (feat["speech_rate"] - 1.0) -
            2 * (feat["pause_ratio"] - 0.2) -
            5 * danger_flag
        )
        
        self.valence = float(feat["sentiment_score"])
        self.arousal = float(feat["rms"] * 5 + feat["speech_rate"] * 2 - feat["pause_ratio"] * 2)
        
        # Map to emotion
        if self.valence > 0.4 and self.arousal > 0.5:
            emotion = "Excited/Happy"
        elif self.valence > 0.4 and self.arousal <= 0.5:
            emotion = "Calm/Content"
        elif self.valence < -0.4 and self.arousal > 0.5:
            emotion = "Angry/Anxious"
        elif self.valence < -0.4 and self.arousal <= 0.5:
            emotion = "Sad/Withdrawn"
        else:
            emotion = "Neutral"
        
        # Update sliding window
        self.score_window.append(score)
        avg_score = np.mean(self.score_window)
        
        if avg_score > 1.5:
            self.window_trend = "GOOD"
        elif avg_score < -1.5:
            self.window_trend = "BAD"
        else:
            self.window_trend = "NORMAL"
        
        return emotion, score
    
    def add_audio_chunk(self, chunk):
        """Add audio chunk to buffer and process when ready"""
        if chunk is None or len(chunk) == 0:
            return
        
        self.audio_buffer.extend(chunk)
        
        # Process when we have enough audio (3 seconds worth)
        if len(self.audio_buffer) >= self.chunk_size and not self.processing:
            # Get exactly chunk_size samples
            audio_to_process = np.array(self.audio_buffer[:self.chunk_size])
            # Remove processed samples from buffer
            self.audio_buffer = self.audio_buffer[self.chunk_size:]
            
            # Process in background thread
            self.processing = True
            threading.Thread(
                target=self._process_and_reset,
                args=(audio_to_process,),
                daemon=True
            ).start()
    
    def _process_and_reset(self, audio_chunk):
        """Process audio and reset flag"""
        self.process_audio_chunk(audio_chunk)
        self.processing = False
    
    def process_audio_chunk(self, audio_chunk):
        try:
            if len(audio_chunk) < self.sample_rate * 0.5:
                return
            
            # Transcribe - fix for Windows file locking
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_path = temp_file.name
            temp_file.close()  # Close it first before writing
            
            try:
                sf.write(temp_path, audio_chunk, self.sample_rate)
                result = self.whisper_model.transcribe(temp_path, fp16=False)
            finally:
                # Try to delete, but don't fail if we can't
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass  # Ignore deletion errors on Windows
            
            text = result.get("text", "").strip()
            if not text:
                return
            
            # Sentiment
            sentiment = self.sentiment_pipe(text[:512])[0]
            self.sentiment_label = sentiment['label']
            
            # Features
            feat = self.compute_audio_features(audio_chunk, text, sentiment)
            self.danger_flag = self.detect_danger_words(text)
            
            # Emotion
            emotion, score = self.compute_emotion(feat, text, self.danger_flag)
            
            # Update state
            self.current_emotion = emotion
            self.current_text = text
            
            print(f"🎤 Audio processed: {emotion} | Text: {text[:50]}...")
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            import traceback
            traceback.print_exc()

class ActivityRecognizer:
    def __init__(self, enable_audio=True):
        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        
        # MediaPipe Face
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
        # FER
        print("Loading facial emotion detection...")
        self.emotion_detector = FER(mtcnn=False)
        print("Facial emotion loaded!")
        
        # YOLO
        print("Loading YOLOv8...")
        self.yolo_model = YOLO('yolov8n.pt')
        print("YOLOv8 loaded!")
        
        # Audio
        self.enable_audio = enable_audio
        self.audio_analyzer = None
        if enable_audio:
            self.audio_analyzer = AudioEmotionAnalyzer()
        
        # Tracking
        self.activity_history = deque(maxlen=30)
        self.movement_history = deque(maxlen=10)
        self.emotion_history = deque(maxlen=5)
        self.cached_emotion = "neutral"
        self.cached_emotion_confidence = 0.0
        
        self.start_time = None
        self.canvas_size = 400
        
        # Frame skip
        self.frame_count = 0
        self.yolo_skip = 3
        self.emotion_skip = 2
        self.audio_skip = 90
        
        # Cached
        self.cached_objects = []
        
        # YOLO threading
        self.yolo_queue = Queue(maxsize=1)
        self.yolo_result_queue = Queue(maxsize=1)
        self.yolo_thread = None
        self.yolo_running = False
        
        # CSV
        self.setup_csv_logging()
    
    def setup_csv_logging(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"activity_log.csv"
        self.csv_file = open(self.csv_filename, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        headers = [
            'timestamp', 'frame_num', 'fps',
            'body_activity', 'body_confidence',
            'facial_expression', 'face_confidence',
            'num_objects', 'detected_objects',
            'audio_text', 'audio_sentiment_label', 'audio_sentiment_score',
            'audio_instant_emotion', 'audio_rms', 'audio_pitch',
            'audio_speech_rate', 'audio_pause_ratio',
            'audio_valence', 'audio_arousal',
            'audio_sliding_window_trend', 'audio_danger_flag'
        ]
        self.csv_writer.writerow(headers)
        self.csv_file.flush()
        print(f"📊 CSV logging to: {self.csv_filename}")
    
    def log_to_csv(self, data):
        try:
            self.csv_writer.writerow(data)
            self.csv_file.flush()
        except Exception as e:
            print(f"CSV error: {e}")
    
    def detect_all_objects(self, frame):
        try:
            scale = 0.5
            small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            results = self.yolo_model(small_frame, verbose=False, conf=0.25)
            
            detected_objects = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]
                    
                    detected_objects.append({
                        'name': class_name,
                        'confidence': confidence,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'class_id': class_id
                    })
            return detected_objects
        except:
            return []
    
    def yolo_worker(self):
        while self.yolo_running:
            try:
                if not self.yolo_queue.empty():
                    frame = self.yolo_queue.get(timeout=0.1)
                    detected_objects = self.detect_all_objects(frame)
                    
                    if not self.yolo_result_queue.full():
                        if not self.yolo_result_queue.empty():
                            try:
                                self.yolo_result_queue.get_nowait()
                            except:
                                pass
                        self.yolo_result_queue.put(detected_objects)
                else:
                    time.sleep(0.01)
            except:
                time.sleep(0.1)
    
    def start_yolo_thread(self):
        self.yolo_running = True
        self.yolo_thread = threading.Thread(target=self.yolo_worker, daemon=True)
        self.yolo_thread.start()
    
    def stop_yolo_thread(self):
        self.yolo_running = False
        if self.yolo_thread:
            self.yolo_thread.join(timeout=1.0)
    
    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def detect_activity(self, landmarks, frame_height):
        if not landmarks:
            return "No Person Detected", 0.0
        
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
        
        torso_height = abs((left_shoulder.y + right_shoulder.y) / 2 - (left_hip.y + right_hip.y) / 2)
        
        hip_center_x = (left_hip.x + right_hip.x) / 2
        self.movement_history.append(hip_center_x)
        
        movement_variance = np.var(list(self.movement_history)) if len(self.movement_history) >= 5 else 0
        
        if torso_height < 0.15:
            return "Sleeping/Lying Down", 0.9
        elif avg_knee_angle < 120 and avg_hip_angle < 120 and torso_height > 0.15:
            return "Sitting", 0.85
        elif avg_knee_angle > 150 and torso_height > 0.25 and movement_variance < 0.001:
            return "Standing", 0.8
        elif torso_height > 0.2 and movement_variance > 0.0005:
            return "Walking", 0.85
        else:
            return "Transitioning", 0.5
    
    def smooth_activity(self, current_activity):
        self.activity_history.append(current_activity)
        
        if len(self.activity_history) < 5:
            return current_activity
        
        recent = list(self.activity_history)[-10:]
        activity_counts = {}
        for act in recent:
            activity_counts[act] = activity_counts.get(act, 0) + 1
        
        return max(activity_counts, key=activity_counts.get)
    
    def detect_facial_expression(self, face_frame):
        try:
            if face_frame is None or face_frame.size == 0:
                return self.cached_emotion, self.cached_emotion_confidence
            
            if np.mean(face_frame) < 10:
                return "No Face", 0.0
            
            small_face = cv2.resize(face_frame, (200, 200))
            emotions = self.emotion_detector.detect_emotions(small_face)
            
            if emotions and len(emotions) > 0:
                emotion_scores = emotions[0]['emotions']
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[dominant_emotion]
                
                self.emotion_history.append(dominant_emotion)
                self.cached_emotion = dominant_emotion
                self.cached_emotion_confidence = confidence
                
                if len(self.emotion_history) >= 3:
                    emotion_counts = {}
                    for em in list(self.emotion_history):
                        emotion_counts[em] = emotion_counts.get(em, 0) + 1
                    smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
                    return smoothed_emotion, confidence
                
                return dominant_emotion, confidence
            else:
                return self.cached_emotion, self.cached_emotion_confidence
        except:
            return self.cached_emotion, self.cached_emotion_confidence
    
    def extract_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        face_canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        raw_face = None
        
        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            
            padding = 0.3
            x = int(max(0, (bboxC.xmin - padding * bboxC.width) * w))
            y = int(max(0, (bboxC.ymin - padding * bboxC.height) * h))
            width = int(min(w - x, bboxC.width * (1 + 2 * padding) * w))
            height = int(min(h - y, bboxC.height * (1 + 2 * padding) * h))
            
            face_roi = frame[y:y+height, x:x+width]
            
            if face_roi.size > 0:
                face_canvas = cv2.resize(face_roi, (self.canvas_size, self.canvas_size))
                raw_face = face_canvas.copy()
                cv2.rectangle(face_canvas, (10, 10), (self.canvas_size-10, self.canvas_size-10),
                            (0, 255, 0), 2)
        else:
            cv2.putText(face_canvas, "No Face Detected", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(face_canvas, "Face View", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return face_canvas, raw_face
    
    def create_yolo_canvas(self, frame, detected_objects):
        yolo_frame = cv2.resize(frame.copy(), (self.canvas_size, self.canvas_size))
        
        original_h, original_w = frame.shape[:2]
        scale_w = self.canvas_size / original_w
        scale_h = self.canvas_size / original_h
        
        for obj in detected_objects:
            ox1, oy1, ox2, oy2 = obj['bbox']
            x1 = int(ox1 * scale_w)
            y1 = int(oy1 * scale_h)
            x2 = int(ox2 * scale_w)
            y2 = int(oy2 * scale_h)
            
            x1 = max(0, min(x1, self.canvas_size-1))
            y1 = max(0, min(y1, self.canvas_size-1))
            x2 = max(0, min(x2, self.canvas_size-1))
            y2 = max(0, min(y2, self.canvas_size-1))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            if obj['name'] == 'person':
                color = (0, 255, 0)
            elif obj['name'] in ['cell phone', 'laptop', 'keyboard', 'mouse']:
                color = (255, 0, 255)
            else:
                color = (0, 165, 255)
            
            cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{obj['name']} {obj['confidence']:.2f}"
            cv2.putText(yolo_frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        overlay = yolo_frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.canvas_size, 60), (0, 0, 0), -1)
        yolo_frame = cv2.addWeighted(overlay, 0.7, yolo_frame, 0.3, 0)
        
        cv2.putText(yolo_frame, "Objects", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return yolo_frame
    
    def create_audio_canvas(self):
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        
        if not self.audio_analyzer:
            cv2.putText(canvas, "Audio Disabled", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
            return canvas
        
        cv2.rectangle(canvas, (0, 0), (self.canvas_size, 60), (40, 40, 40), -1)
        cv2.putText(canvas, "Audio Emotion", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        emotion = self.audio_analyzer.current_emotion
        cv2.putText(canvas, emotion, (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        y_pos = 160
        metrics = [
            f"RMS: {self.audio_analyzer.rms:.3f}",
            f"Valence: {self.audio_analyzer.valence:.2f}",
            f"Arousal: {self.audio_analyzer.arousal:.2f}",
            f"Trend: {self.audio_analyzer.window_trend}",
        ]
        
        for metric in metrics:
            cv2.putText(canvas, metric, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 30
        
        text = self.audio_analyzer.current_text
        if text:
            cv2.rectangle(canvas, (0, 280), (self.canvas_size, self.canvas_size),
                         (30, 30, 30), -1)
            cv2.putText(canvas, "Speech:", (10, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            words = text.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if len(test_line) > 35:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)
            
            y = 325
            for line in lines[:3]:
                cv2.putText(canvas, line, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y += 20
        
        return canvas
    
    def add_overlay(self, frame, activity, confidence):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (390, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, f"Activity: {activity}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        bar_length = int(confidence * 250)
        cv2.rectangle(frame, (20, 90), (20 + bar_length, 105), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, 90), (270, 105), (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame, audio_chunk=None):
        self.frame_count += 1
        
        # Body
        frame_resized = cv2.resize(frame, (self.canvas_size, self.canvas_size))
        
        # Face
        face_frame, raw_face = self.extract_face(frame)
        
        emotion = self.cached_emotion
        emotion_confidence = self.cached_emotion_confidence
        
        if raw_face is not None and self.frame_count % self.emotion_skip == 0:
            emotion, emotion_confidence = self.detect_facial_expression(raw_face)
        
        cv2.putText(face_frame, f"Emotion: {emotion}", (10, 360),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(face_frame, f"Conf: {emotion_confidence:.2f}", (10, 390),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Pose
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        activity = "No Person Detected"
        confidence = 0.0
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_resized, results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            landmarks = results.pose_landmarks.landmark
            activity, confidence = self.detect_activity(landmarks, frame_resized.shape[0])
            activity = self.smooth_activity(activity)
        
        self.add_overlay(frame_resized, activity, confidence)
        
        # YOLO
        detected_objects = self.cached_objects
        
        if not self.yolo_result_queue.empty():
            try:
                detected_objects = self.yolo_result_queue.get_nowait()
                self.cached_objects = detected_objects
            except:
                pass
        
        if self.frame_count % self.yolo_skip == 0:
            if self.yolo_queue.empty():
                try:
                    self.yolo_queue.put_nowait(frame.copy())
                except:
                    pass
        
        yolo_canvas = self.create_yolo_canvas(frame, detected_objects)
        
        # Audio - feed every frame's audio chunk to the buffer
        if self.audio_analyzer and audio_chunk is not None:
            self.audio_analyzer.add_audio_chunk(audio_chunk)
        
        audio_canvas = self.create_audio_canvas()
        
        return frame_resized, face_frame, yolo_canvas, audio_canvas, activity, confidence, emotion, emotion_confidence, detected_objects
    
    def extract_audio_from_video(self, video_path):
        try:
            import ffmpeg
            audio_path = "temp_extracted_audio.wav"
            (
                ffmpeg.input(video_path)
                .output(audio_path, ac=1, ar=16000, format="wav", loglevel="error")
                .overwrite_output()
                .run()
            )
            return audio_path
        except Exception as e:
            print(f"Could not extract audio: {e}")
            return None
    
    def print_frame_data(self, frame_idx, elapsed_time, fps, activity, confidence,
                        emotion, emotion_confidence, detected_objects):
        """Print comprehensive data for current frame"""
        print("\n" + "="*80)
        print(f"FRAME {frame_idx} | Time: {elapsed_time:.2f}s | FPS: {fps}")
        print("="*80)
        
        # Body Activity
        print(f"\n🏃 BODY ACTIVITY:")
        print(f"   Activity: {activity}")
        print(f"   Confidence: {confidence:.3f}")
        
        # Facial Expression
        print(f"\n😊 FACIAL EXPRESSION:")
        print(f"   Expression: {emotion}")
        print(f"   Confidence: {emotion_confidence:.3f}")
        
        # Objects
        print(f"\n📦 OBJECTS DETECTED: {len(detected_objects)}")
        if detected_objects:
            for i, obj in enumerate(detected_objects[:5], 1):
                print(f"   {i}. {obj['name']} (conf: {obj['confidence']:.2f})")
            if len(detected_objects) > 5:
                print(f"   ... and {len(detected_objects)-5} more")
        
        # Audio
        if self.audio_analyzer:
            print(f"\n🎤 AUDIO ANALYSIS:")
            print(f"   Text: {self.audio_analyzer.current_text if self.audio_analyzer.current_text else '[no speech]'}")
            print(f"   Sentiment: {self.audio_analyzer.sentiment_label} ({self.audio_analyzer.sentiment_score:.3f})")
            print(f"   Instant Emotion: {self.audio_analyzer.current_emotion}")
            print(f"   RMS: {self.audio_analyzer.rms:.4f}")
            print(f"   Pitch: {self.audio_analyzer.pitch:.2f} Hz")
            print(f"   Speech Rate: {self.audio_analyzer.speech_rate:.2f} words/sec")
            print(f"   Pause Ratio: {self.audio_analyzer.pause_ratio:.3f}")
            print(f"   Valence: {self.audio_analyzer.valence:.3f}")
            print(f"   Arousal: {self.audio_analyzer.arousal:.3f}")
            print(f"   Sliding Window Trend: {self.audio_analyzer.window_trend}")
            print(f"   Danger Flag: {self.audio_analyzer.danger_flag}")
        
        print("="*80)
    
    def run_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            return
        
        # Audio extraction
        audio_path = None
        audio_data = None
        audio_sr = None
        
        if self.enable_audio:
            print("Extracting audio...")
            audio_path = self.extract_audio_from_video(video_path)
            if audio_path and os.path.exists(audio_path):
                audio_data, audio_sr = librosa.load(audio_path, sr=16000)
                print(f"Audio loaded: {len(audio_data)/audio_sr:.1f} seconds")
        
        print("\n🎬 Starting 4-Canvas Analysis + CSV Logging")
        print("Press 'q' to quit\n")
        
        self.start_yolo_thread()
        self.start_time = time.time()
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        
        # Windows
        cv2.namedWindow('1. Body', cv2.WINDOW_NORMAL)
        cv2.namedWindow('2. Face', cv2.WINDOW_NORMAL)
        cv2.namedWindow('3. Objects', cv2.WINDOW_NORMAL)
        cv2.namedWindow('4. Audio', cv2.WINDOW_NORMAL)
        cv2.moveWindow('1. Body', 50, 50)
        cv2.moveWindow('2. Face', 500, 50)
        cv2.moveWindow('3. Objects', 950, 50)
        cv2.moveWindow('4. Audio', 1400, 50)
        
        frame_idx = 0
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            elapsed_time = time.time() - self.start_time
            
            # Audio chunk - extract audio for this frame
            audio_chunk = None
            if audio_data is not None and video_fps > 0:
                # Calculate audio samples for this frame
                samples_per_frame = int(audio_sr / video_fps)
                audio_start = int(frame_idx * samples_per_frame)
                audio_end = int((frame_idx + 1) * samples_per_frame)
                
                if audio_end <= len(audio_data):
                    audio_chunk = audio_data[audio_start:audio_end]
            
            # Process
            body_frame, face_frame, yolo_frame, audio_frame, activity, confidence, emotion, emotion_confidence, detected_objects = self.process_frame(frame, audio_chunk)
            
            # FPS
            fps_counter += 1
            if time.time() - fps_time > 1:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            cv2.putText(body_frame, f"FPS: {fps}", (body_frame.shape[1] - 100, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Display
            cv2.imshow('1. Body', body_frame)
            cv2.imshow('2. Face', face_frame)
            cv2.imshow('3. Objects', yolo_frame)
            cv2.imshow('4. Audio', audio_frame)
            
            # Get audio data
            audio_text = self.audio_analyzer.current_text if self.audio_analyzer else ""
            audio_sentiment_label = self.audio_analyzer.sentiment_label if self.audio_analyzer else "N/A"
            audio_sentiment_score = self.audio_analyzer.sentiment_score if self.audio_analyzer else 0.0
            audio_emotion = self.audio_analyzer.current_emotion if self.audio_analyzer else "N/A"
            audio_rms = self.audio_analyzer.rms if self.audio_analyzer else 0.0
            audio_pitch = self.audio_analyzer.pitch if self.audio_analyzer else 0.0
            audio_speech_rate = self.audio_analyzer.speech_rate if self.audio_analyzer else 0.0
            audio_pause_ratio = self.audio_analyzer.pause_ratio if self.audio_analyzer else 0.0
            audio_valence = self.audio_analyzer.valence if self.audio_analyzer else 0.0
            audio_arousal = self.audio_analyzer.arousal if self.audio_analyzer else 0.0
            audio_trend = self.audio_analyzer.window_trend if self.audio_analyzer else "N/A"
            audio_danger = self.audio_analyzer.danger_flag if self.audio_analyzer else 0
            
            objects_str = ";".join([o['name'] for o in detected_objects[:10]])
            
            # CSV row
            csv_row = [
                elapsed_time, frame_idx, fps,
                activity, confidence,
                emotion, emotion_confidence,
                len(detected_objects), objects_str,
                audio_text, audio_sentiment_label, audio_sentiment_score,
                audio_emotion, audio_rms, audio_pitch,
                audio_speech_rate, audio_pause_ratio,
                audio_valence, audio_arousal,
                audio_trend, audio_danger
            ]
            self.log_to_csv(csv_row)
            
            # Terminal output (every 30 frames)
            if frame_idx % 30 == 0:
                self.print_frame_data(frame_idx, elapsed_time, fps, activity, confidence,
                                     emotion, emotion_confidence, detected_objects)
            
            frame_idx += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.stop_yolo_thread()
        cap.release()
        cv2.destroyAllWindows()
        
        if self.csv_file:
            self.csv_file.close()
            print(f"\n✅ Complete! Data saved to: {self.csv_filename}")
        
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
    
    def cleanup(self):
        self.stop_yolo_thread()
        self.pose.close()
        self.face_detection.close()
        if self.csv_file:
            self.csv_file.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Activity Recognition')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--yolo-skip', type=int, default=3)
    parser.add_argument('--emotion-skip', type=int, default=2)
    parser.add_argument('--audio-skip', type=int, default=90)
    parser.add_argument('--disable-audio', action='store_true')
    
    args = parser.parse_args()
    
    recognizer = ActivityRecognizer(enable_audio=not args.disable_audio)
    recognizer.yolo_skip = args.yolo_skip
    recognizer.emotion_skip = args.emotion_skip
    recognizer.audio_skip = args.audio_skip
    
    try:
        recognizer.run_video(args.video_path)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recognizer.cleanup()

if __name__ == "__main__":
    main()