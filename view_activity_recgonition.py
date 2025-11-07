#!/usr/bin/env python3
"""
Real-time Activity Recognition System with 3 Parallel Canvases
1. Body Activity - MediaPipe Pose tracking
2. Face View - Facial expression detection
3. Objects Detection - YOLO detecting EVERYTHING in frame
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from fer import FER
from ultralytics import YOLO

class ActivityRecognizer:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
        # Initialize FER for emotion detection
        print("Loading emotion detection model...")
        self.emotion_detector = FER(mtcnn=False)
        print("Emotion detection model loaded!")
        
        # Initialize YOLOv8 for comprehensive object detection
        print("Loading YOLOv8 object detection model...")
        self.yolo_model = YOLO('yolov8n.pt')
        print("YOLOv8 model loaded!")
        
        # Tracking
        self.activity_history = deque(maxlen=30)
        self.movement_history = deque(maxlen=10)
        self.emotion_history = deque(maxlen=5)
        self.last_emotion = "neutral"
        self.last_emotion_confidence = 0.0
        
        # Timing
        self.start_time = None
        
        # Canvas sizes
        self.canvas_size = 400
        
    def detect_all_objects(self, frame):
        """
        Detect ALL objects in frame using YOLO - including people, furniture, items, everything!
        Returns: list of all detected objects
        """
        try:
            # Run YOLO detection with lower confidence to catch more objects
            results = self.yolo_model(frame, verbose=False, conf=0.25)
            
            detected_objects = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
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
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []
    
    def create_yolo_canvas(self, frame, detected_objects):
        """Create dedicated 400x400 canvas showing ALL YOLO detections"""
        # Resize frame to 400x400
        yolo_frame = cv2.resize(frame.copy(), (self.canvas_size, self.canvas_size))
        
        # Calculate scale factors
        original_h, original_w = frame.shape[:2]
        scale_w = self.canvas_size / original_w
        scale_h = self.canvas_size / original_h
        
        # Draw all detected objects
        for obj in detected_objects:
            ox1, oy1, ox2, oy2 = obj['bbox']
            
            # Scale coordinates
            x1 = int(ox1 * scale_w)
            y1 = int(oy1 * scale_h)
            x2 = int(ox2 * scale_w)
            y2 = int(oy2 * scale_h)
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, self.canvas_size-1))
            y1 = max(0, min(y1, self.canvas_size-1))
            x2 = max(0, min(x2, self.canvas_size-1))
            y2 = max(0, min(y2, self.canvas_size-1))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Color based on object type
            if obj['name'] == 'person':
                color = (0, 255, 0)  # Green for person
            elif obj['name'] in ['cell phone', 'laptop', 'keyboard', 'mouse', 'remote', 'tv']:
                color = (255, 0, 255)  # Magenta for electronics
            elif obj['name'] in ['bottle', 'cup', 'bowl', 'fork', 'knife', 'spoon']:
                color = (0, 255, 255)  # Yellow for food/drink
            elif obj['name'] in ['chair', 'couch', 'bed', 'dining table']:
                color = (255, 165, 0)  # Orange for furniture
            else:
                color = (0, 165, 255)  # Default orange
            
            # Draw bounding box
            cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{obj['name']} {obj['confidence']:.2f}"
            font_scale = 0.4
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Background for text
            text_y = max(text_height + 8, y1)
            cv2.rectangle(yolo_frame, (x1, text_y - text_height - 8),
                         (min(x1 + text_width + 6, self.canvas_size), text_y), color, -1)
            
            # Text
            cv2.putText(yolo_frame, label, (x1 + 3, text_y - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # Add header
        overlay = yolo_frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.canvas_size, 60), (0, 0, 0), -1)
        yolo_frame = cv2.addWeighted(overlay, 0.7, yolo_frame, 0.3, 0)
        
        cv2.putText(yolo_frame, "YOLO Object Detection", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(yolo_frame, f"Objects: {len(detected_objects)}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add legend at bottom
        if detected_objects:
            legend_y = 340
            cv2.rectangle(yolo_frame, (0, legend_y), (self.canvas_size, self.canvas_size), (0, 0, 0), -1)
            
            # Count by category
            people = [o for o in detected_objects if o['name'] == 'person']
            electronics = [o for o in detected_objects if o['name'] in ['cell phone', 'laptop', 'keyboard', 'mouse']]
            furniture = [o for o in detected_objects if o['name'] in ['chair', 'couch', 'bed', 'table']]
            
            y_pos = legend_y + 15
            if people:
                cv2.putText(yolo_frame, f"People: {len(people)}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                y_pos += 15
            if electronics:
                names = ', '.join([o['name'] for o in electronics[:2]])
                cv2.putText(yolo_frame, f"Electronics: {names}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                y_pos += 15
            if furniture:
                names = ', '.join([o['name'] for o in furniture[:2]])
                cv2.putText(yolo_frame, f"Furniture: {names}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
        
        return yolo_frame
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
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
        """Detect current activity based on pose landmarks"""
        if not landmarks:
            return "No Person Detected", 0.0
        
        # Extract key landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Calculate angles
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
        
        # Body orientation
        torso_height = abs((left_shoulder.y + right_shoulder.y) / 2 - (left_hip.y + right_hip.y) / 2)
        
        # Track movement
        hip_center_x = (left_hip.x + right_hip.x) / 2
        self.movement_history.append(hip_center_x)
        
        movement_variance = np.var(list(self.movement_history)) if len(self.movement_history) >= 5 else 0
        
        # Activity classification
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
        """Smooth activity detection"""
        self.activity_history.append(current_activity)
        
        if len(self.activity_history) < 5:
            return current_activity
        
        recent = list(self.activity_history)[-10:]
        activity_counts = {}
        for act in recent:
            activity_counts[act] = activity_counts.get(act, 0) + 1
        
        return max(activity_counts, key=activity_counts.get)
    
    def detect_facial_expression(self, face_frame):
        """Detect facial expression using FER"""
        try:
            if face_frame is None or face_frame.size == 0:
                return self.last_emotion, self.last_emotion_confidence
            
            if np.mean(face_frame) < 10:
                return "No Face", 0.0
            
            emotions = self.emotion_detector.detect_emotions(face_frame)
            
            if emotions and len(emotions) > 0:
                emotion_scores = emotions[0]['emotions']
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[dominant_emotion]
                
                self.emotion_history.append(dominant_emotion)
                self.last_emotion = dominant_emotion
                self.last_emotion_confidence = confidence
                
                if len(self.emotion_history) >= 3:
                    emotion_counts = {}
                    for em in list(self.emotion_history):
                        emotion_counts[em] = emotion_counts.get(em, 0) + 1
                    smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
                    return smoothed_emotion, confidence
                
                return dominant_emotion, confidence
            else:
                return self.last_emotion, self.last_emotion_confidence
            
        except Exception as e:
            return self.last_emotion, self.last_emotion_confidence
    
    def extract_face(self, frame):
        """Extract face region and resize to 400x400"""
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
    
    def get_emotion_color(self, emotion):
        """Get color for emotion display"""
        emotion_colors = {
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'angry': (0, 0, 255),
            'fear': (255, 0, 255),
            'surprise': (0, 255, 255),
            'disgust': (128, 0, 128),
            'neutral': (255, 255, 255)
        }
        return emotion_colors.get(emotion.lower(), (255, 255, 255))
    
    def add_overlay(self, frame, activity, confidence):
        """Add text overlay with activity information"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (390, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, f"Activity: {activity}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        bar_length = int(confidence * 250)
        cv2.rectangle(frame, (20, 90), (20 + bar_length, 105), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, 90), (270, 105), (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame):
        """Process frame and return 3 canvases"""
        # 1. Body Activity Canvas
        frame_resized = cv2.resize(frame, (self.canvas_size, self.canvas_size))
        
        # 2. Face Canvas
        face_frame, raw_face = self.extract_face(frame)
        
        emotion = "No Face"
        emotion_confidence = 0.0
        if raw_face is not None:
            emotion, emotion_confidence = self.detect_facial_expression(raw_face)
        
        emotion_color = self.get_emotion_color(emotion)
        cv2.putText(face_frame, f"Emotion: {emotion}", (10, 360),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
        cv2.putText(face_frame, f"Conf: {emotion_confidence:.2f}", (10, 390),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Process pose
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        activity = "No Person Detected"
        confidence = 0.0
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_resized,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            landmarks = results.pose_landmarks.landmark
            activity, confidence = self.detect_activity(landmarks, frame_resized.shape[0])
            activity = self.smooth_activity(activity)
        
        self.add_overlay(frame_resized, activity, confidence)
        
        # 3. YOLO Objects Canvas
        detected_objects = self.detect_all_objects(frame)
        yolo_canvas = self.create_yolo_canvas(frame, detected_objects)
        
        return frame_resized, face_frame, yolo_canvas, activity, confidence, emotion, emotion_confidence, detected_objects
    
    def run_webcam(self):
        """Run with 3 parallel windows"""
        cap = cv2.VideoCapture(0)
        
        print("Starting real-time recognition with 3 parallel canvases...")
        print("Press 'q' to quit")
        
        self.start_time = time.time()
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        
        # Create 3 windows
        cv2.namedWindow('1. Body Activity', cv2.WINDOW_NORMAL)
        cv2.namedWindow('2. Face View', cv2.WINDOW_NORMAL)
        cv2.namedWindow('3. YOLO Objects', cv2.WINDOW_NORMAL)
        cv2.moveWindow('1. Body Activity', 50, 50)
        cv2.moveWindow('2. Face View', 500, 50)
        cv2.moveWindow('3. YOLO Objects', 950, 50)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            elapsed_time = time.time() - self.start_time
            
            # Process frame - get all 3 canvases
            body_frame, face_frame, yolo_frame, activity, confidence, emotion, emotion_confidence, detected_objects = self.process_frame(frame)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time > 1:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Add FPS and time to body frame
            cv2.putText(body_frame, f"FPS: {fps}", (body_frame.shape[1] - 100, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(body_frame, f"Time: {elapsed_time:.1f}s", (body_frame.shape[1] - 120, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Show all 3 frames
            cv2.imshow('1. Body Activity', body_frame)
            cv2.imshow('2. Face View', face_frame)
            cv2.imshow('3. YOLO Objects', yolo_frame)
            
            # Console log
            objects_summary = f"{len(detected_objects)} objects"
            if detected_objects:
                top_objects = ', '.join([o['name'] for o in detected_objects[:5]])
                objects_summary = f"{len(detected_objects)} objects: {top_objects}"
            
            print(f"Time: {elapsed_time:.2f}s | Activity: {activity} | Conf: {confidence:.2f} | "
                  f"Emotion: {emotion} | Conf: {emotion_confidence:.2f} | {objects_summary}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def cleanup(self):
        """Cleanup resources"""
        self.pose.close()
        self.face_detection.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Activity Recognition with 3 Parallel Canvases')
    parser.add_argument('--source', type=int, default=0, help='Webcam source (default: 0)')
    
    args = parser.parse_args()
    
    recognizer = ActivityRecognizer()
    
    try:
        recognizer.run_webcam()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recognizer.cleanup()


if __name__ == "__main__":
    main()