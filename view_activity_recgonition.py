#!/usr/bin/env python3
"""
Real-time Activity Recognition System
Detects patient activities: walking, sitting, standing, sleeping/lying down
Uses MediaPipe Pose for skeletal tracking
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class ActivityRecognizer:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Face Detection for face extraction
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
        # Activity state tracking
        self.activity_history = deque(maxlen=30)  # Last 30 frames for smoothing
        self.movement_history = deque(maxlen=10)  # Track movement for walking detection
        
        # Timing
        self.start_time = None
        
        # Canvas size
        self.canvas_size = 400
        
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
        """
        Detect current activity based on pose landmarks
        Returns: activity name and confidence
        """
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
        
        # Calculate body orientation and position
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        avg_hip_y = (left_hip.y + right_hip.y) / 2
        avg_knee_y = (left_knee.y + right_knee.y) / 2
        avg_ankle_y = (left_ankle.y + right_ankle.y) / 2
        
        # Calculate knee angles
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        # Calculate hip angles
        left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
        
        # Torso orientation (vertical alignment)
        torso_height = abs(avg_shoulder_y - avg_hip_y)
        body_verticality = torso_height  # Higher value = more vertical
        
        # Track movement for walking detection
        hip_center_x = (left_hip.x + right_hip.x) / 2
        self.movement_history.append(hip_center_x)
        
        # Calculate horizontal movement
        if len(self.movement_history) >= 5:
            movement_variance = np.var(list(self.movement_history))
        else:
            movement_variance = 0
        
        # Activity Classification Logic
        confidence = 0.0
        
        # SLEEPING / LYING DOWN
        # - Body is horizontal (torso height is very small)
        # - Head and hips are at similar height
        if torso_height < 0.15:  # Body is nearly horizontal
            activity = "Sleeping/Lying Down"
            confidence = 0.9
            
        # SITTING
        # - Torso is more vertical
        # - Knees are bent significantly (angle < 120)
        # - Hips are bent
        elif avg_knee_angle < 120 and avg_hip_angle < 120 and body_verticality > 0.15:
            activity = "Sitting"
            confidence = 0.85
            
        # STANDING
        # - Body is vertical
        # - Knees are relatively straight (angle > 150)
        # - Minimal horizontal movement
        elif avg_knee_angle > 150 and body_verticality > 0.25 and movement_variance < 0.001:
            activity = "Standing"
            confidence = 0.8
            
        # WALKING
        # - Body is vertical
        # - Moderate knee bending and unbending
        # - Noticeable horizontal movement
        elif body_verticality > 0.2 and movement_variance > 0.0005:
            activity = "Walking"
            confidence = 0.85
            
        # UNCERTAIN / TRANSITIONING
        else:
            activity = "Transitioning"
            confidence = 0.5
        
        return activity, confidence
    
    def extract_face(self, frame):
        """Extract face region from frame and resize to 400x400"""
        # Convert BGR to RGB for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_frame)
        
        # Create a blank 400x400 canvas
        face_canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        
        if results.detections:
            # Get the first detected face
            detection = results.detections[0]
            
            # Get bounding box
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            
            # Calculate coordinates with some padding
            padding = 0.3  # 30% padding around face
            x = int(max(0, (bboxC.xmin - padding * bboxC.width) * w))
            y = int(max(0, (bboxC.ymin - padding * bboxC.height) * h))
            width = int(min(w - x, bboxC.width * (1 + 2 * padding) * w))
            height = int(min(h - y, bboxC.height * (1 + 2 * padding) * h))
            
            # Extract face region
            face_roi = frame[y:y+height, x:x+width]
            
            if face_roi.size > 0:
                # Resize face to 400x400
                face_canvas = cv2.resize(face_roi, (self.canvas_size, self.canvas_size))
                
                # Draw detection box on face canvas for reference
                cv2.rectangle(face_canvas, (10, 10), (self.canvas_size-10, self.canvas_size-10), 
                            (0, 255, 0), 2)
        else:
            # No face detected - show message
            cv2.putText(face_canvas, "No Face Detected", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add label
        cv2.putText(face_canvas, "Face View", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return face_canvas
    
    def smooth_activity(self, current_activity):
        """Smooth activity detection over multiple frames"""
        self.activity_history.append(current_activity)
        
        if len(self.activity_history) < 5:
            return current_activity
        
        # Get most common activity in recent history
        recent = list(self.activity_history)[-10:]
        activity_counts = {}
        for act in recent:
            activity_counts[act] = activity_counts.get(act, 0) + 1
        
        # Return most frequent activity
        smoothed = max(activity_counts, key=activity_counts.get)
        return smoothed
    
    def process_frame(self, frame):
        """Process a single frame and return annotated frame with activity + face extraction"""
        # Resize frame to 400x400 for body activity view
        frame_resized = cv2.resize(frame, (self.canvas_size, self.canvas_size))
        
        # Extract face before processing (use original frame for better quality)
        face_frame = self.extract_face(frame)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Initialize activity
        activity = "No Person Detected"
        confidence = 0.0
        
        # Draw pose landmarks and detect activity
        if results.pose_landmarks:
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                frame_resized,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Detect activity
            landmarks = results.pose_landmarks.landmark
            activity, confidence = self.detect_activity(landmarks, frame_resized.shape[0])
            activity = self.smooth_activity(activity)
        
        # Add text overlay
        self.add_overlay(frame_resized, activity, confidence)
        
        return frame_resized, face_frame, activity, confidence
    
    def add_overlay(self, frame, activity, confidence):
        """Add text overlay with activity information"""
        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Activity text
        cv2.putText(frame, f"Activity: {activity}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Confidence bar
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Confidence bar visualization
        bar_length = int(confidence * 300)
        cv2.rectangle(frame, (180, 100), (180 + bar_length, 110), (0, 255, 0), -1)
        cv2.rectangle(frame, (180, 100), (480, 110), (255, 255, 255), 2)
        
        return frame
    
    def run_webcam(self):
        """Run activity recognition on webcam feed"""
        cap = cv2.VideoCapture(0)
        
        print("Starting real-time activity recognition...")
        print("Press 'q' to quit")
        
        self.start_time = time.time()  # Initialize start time
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        
        # Position windows side by side
        cv2.namedWindow('Body Activity', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Face View', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Body Activity', 50, 50)
        cv2.moveWindow('Face View', 500, 50)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time
            
            # Process frame
            body_frame, face_frame, activity, confidence = self.process_frame(frame)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time > 1:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Display FPS on body frame
            cv2.putText(body_frame, f"FPS: {fps}", (body_frame.shape[1] - 120, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Display elapsed time on body frame
            cv2.putText(body_frame, f"Time: {elapsed_time:.1f}s", (body_frame.shape[1] - 120, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Show both frames
            cv2.imshow('Body Activity', body_frame)
            cv2.imshow('Face View', face_frame)
            
            # Log activity with timestamp
            print(f"Time: {elapsed_time:.2f}s | Activity: {activity} | Confidence: {confidence:.2f}")
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_video(self, video_path):
        """Run activity recognition on video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        print(f"Processing video: {video_path}")
        print("Press 'q' to quit")
        
        self.start_time = time.time()  # Initialize start time
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        
        # Position windows side by side
        cv2.namedWindow('Body Activity', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Face View', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Body Activity', 50, 50)
        cv2.moveWindow('Face View', 500, 50)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to grab frame")
                break
            
            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time
            
            # Process frame
            body_frame, face_frame, activity, confidence = self.process_frame(frame)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time > 1:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Display FPS on body frame
            cv2.putText(body_frame, f"FPS: {fps}", (body_frame.shape[1] - 120, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Display elapsed time on body frame
            cv2.putText(body_frame, f"Time: {elapsed_time:.1f}s", (body_frame.shape[1] - 120, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Show both frames
            cv2.imshow('Body Activity', body_frame)
            cv2.imshow('Face View', face_frame)
            
            # Log activity with timestamp
            print(f"Time: {elapsed_time:.2f}s | Activity: {activity} | Confidence: {confidence:.2f}")
            
            # Quit on 'q'
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
    
    parser = argparse.ArgumentParser(description='Real-time Activity Recognition')
    parser.add_argument('--video', type=str, help='Path to video file (optional, uses webcam if not provided)')
    parser.add_argument('--source', type=int, default=0, help='Webcam source (default: 0)')
    
    args = parser.parse_args()
    
    # Create recognizer
    recognizer = ActivityRecognizer()
    
    try:
        if args.video:
            recognizer.run_video(args.video)
        else:
            recognizer.run_webcam()
    except KeyboardInterrupt:
        print("\nStopping activity recognition...")
    finally:
        recognizer.cleanup()


if __name__ == "__main__":
    main()