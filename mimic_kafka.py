#!/usr/bin/env python3
"""
Real-Time Mental Health Monitoring System
Reads CSV file live, applies rule-based analysis, and displays mental health score (-1 to +1)
"""

import pandas as pd
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from datetime import datetime
import os

class MentalHealthMonitor:
    """Real-time mental health analysis with rule-based flagging"""
    
    def __init__(self, csv_path, window_size=30):
        self.csv_path = csv_path
        self.window_size = window_size  # frames to consider for trend
        
        # Sliding windows for each metric
        self.activity_window = deque(maxlen=window_size)
        self.face_emotion_window = deque(maxlen=window_size)
        self.audio_emotion_window = deque(maxlen=window_size)
        self.audio_sentiment_window = deque(maxlen=window_size)
        self.audio_valence_window = deque(maxlen=window_size)
        self.audio_arousal_window = deque(maxlen=window_size)
        self.rms_window = deque(maxlen=window_size)
        self.speech_rate_window = deque(maxlen=window_size)
        self.pause_ratio_window = deque(maxlen=window_size)
        
        # Mental health score history
        self.mh_score_history = deque(maxlen=200)  # Last 200 scores
        self.timestamp_history = deque(maxlen=200)
        
        # Alert flags
        self.current_alerts = []
        self.alert_history = deque(maxlen=50)
        
        # Last processed row
        self.last_processed_row = -1
        
        # Overall statistics
        self.total_frames = 0
        self.alert_count = 0
        
    def map_activity_score(self, activity):
        """Map activity to mental health score component"""
        activity_scores = {
            'Walking': 0.3,
            'Standing': 0.1,
            'Sitting': 0.0,
            'Sleeping/Lying Down': -0.2,
            'Transitioning': 0.0,
            'No Person Detected': -0.1
        }
        return activity_scores.get(activity, 0.0)
    
    def map_facial_emotion_score(self, emotion):
        """Map facial emotion to mental health score component"""
        emotion_scores = {
            'happy': 0.5,
            'neutral': 0.0,
            'surprise': 0.2,
            'sad': -0.4,
            'angry': -0.3,
            'fear': -0.5,
            'disgust': -0.3,
            'No Face': 0.0
        }
        return emotion_scores.get(emotion, 0.0)
    
    def map_audio_emotion_score(self, emotion):
        """Map audio emotion to mental health score component"""
        emotion_scores = {
            'Excited/Happy': 0.6,
            'Calm/Content': 0.4,
            'Neutral': 0.0,
            'Angry/Anxious': -0.5,
            'Sad/Withdrawn': -0.6,
            'N/A': 0.0
        }
        return emotion_scores.get(emotion, 0.0)
    
    def analyze_row(self, row):
        """Analyze a single CSV row and return mental health score + alerts"""
        alerts = []
        
        # Extract data
        activity = row.get('body_activity', 'No Person Detected')
        activity_conf = float(row.get('body_confidence', 0.0))
        face_emotion = row.get('facial_expression', 'neutral')
        face_conf = float(row.get('face_confidence', 0.0))
        audio_emotion = row.get('audio_instant_emotion', 'N/A')
        audio_sentiment = row.get('audio_sentiment_label', 'NEUTRAL')
        audio_sent_score = float(row.get('audio_sentiment_score', 0.0))
        audio_valence = float(row.get('audio_valence', 0.0))
        audio_arousal = float(row.get('audio_arousal', 0.0))
        audio_rms = float(row.get('audio_rms', 0.0))
        speech_rate = float(row.get('audio_speech_rate', 0.0))
        pause_ratio = float(row.get('audio_pause_ratio', 0.0))
        danger_flag = int(row.get('audio_danger_flag', 0))
        audio_trend = row.get('audio_sliding_window_trend', 'NORMAL')
        audio_text = row.get('audio_text', '')
        
        # Add to windows
        self.activity_window.append(self.map_activity_score(activity))
        self.face_emotion_window.append(self.map_facial_emotion_score(face_emotion))
        self.audio_emotion_window.append(self.map_audio_emotion_score(audio_emotion))
        
        # Sentiment mapping
        if audio_sentiment == 'POSITIVE':
            sent_score = audio_sent_score
        elif audio_sentiment == 'NEGATIVE':
            sent_score = -audio_sent_score
        else:
            sent_score = 0.0
        self.audio_sentiment_window.append(sent_score)
        
        self.audio_valence_window.append(audio_valence)
        self.audio_arousal_window.append(audio_arousal)
        self.rms_window.append(audio_rms)
        self.speech_rate_window.append(speech_rate)
        self.pause_ratio_window.append(pause_ratio)
        
        # =================================================================
        # RULE-BASED FLAGGING SYSTEM
        # =================================================================
        
        # CRITICAL FLAG 1: Danger words detected
        if danger_flag == 1:
            alerts.append({
                'level': 'CRITICAL',
                'message': f'⚠️ DANGER WORDS DETECTED: "{audio_text[:50]}"',
                'score_impact': -1.0
            })
        
        # CRITICAL FLAG 2: Persistent negative emotions (face + audio)
        if len(self.face_emotion_window) >= 10:
            recent_face = list(self.face_emotion_window)[-10:]
            recent_audio = list(self.audio_emotion_window)[-10:]
            if np.mean(recent_face) < -0.3 and np.mean(recent_audio) < -0.4:
                alerts.append({
                    'level': 'CRITICAL',
                    'message': '😟 PERSISTENT NEGATIVE EMOTIONS (Face: Sad, Audio: Withdrawn)',
                    'score_impact': -0.8
                })
        
        # HIGH FLAG 3: Very low energy + negative sentiment
        if len(self.rms_window) >= 5:
            recent_rms = list(self.rms_window)[-5:]
            recent_sent = list(self.audio_sentiment_window)[-5:]
            if np.mean(recent_rms) < 0.02 and np.mean(recent_sent) < -0.3:
                alerts.append({
                    'level': 'HIGH',
                    'message': '😔 LOW ENERGY + NEGATIVE SPEECH',
                    'score_impact': -0.6
                })
        
        # HIGH FLAG 4: Extended inactivity (lying down for extended period)
        if len(self.activity_window) >= 20:
            recent_activity = [a for a in list(self.activity_window)[-20:]]
            if np.mean(recent_activity) < -0.15:  # Mostly lying down
                alerts.append({
                    'level': 'HIGH',
                    'message': '🛏️ EXTENDED INACTIVITY (Lying down)',
                    'score_impact': -0.5
                })
        
        # MEDIUM FLAG 5: High pause ratio in speech (hesitation/lack of fluency)
        if len(self.pause_ratio_window) >= 5:
            recent_pause = list(self.pause_ratio_window)[-5:]
            if np.mean(recent_pause) > 0.4:
                alerts.append({
                    'level': 'MEDIUM',
                    'message': '🤐 HIGH SPEECH HESITATION',
                    'score_impact': -0.3
                })
        
        # MEDIUM FLAG 6: Negative valence + high arousal (stress/anxiety)
        if len(self.audio_valence_window) >= 5 and len(self.audio_arousal_window) >= 5:
            recent_valence = list(self.audio_valence_window)[-5:]
            recent_arousal = list(self.audio_arousal_window)[-5:]
            if np.mean(recent_valence) < -0.3 and np.mean(recent_arousal) > 0.5:
                alerts.append({
                    'level': 'MEDIUM',
                    'message': '😰 SIGNS OF STRESS/ANXIETY',
                    'score_impact': -0.4
                })
        
        # POSITIVE FLAG 7: Consistent positive emotions
        if len(self.face_emotion_window) >= 10:
            recent_face = list(self.face_emotion_window)[-10:]
            recent_audio = list(self.audio_emotion_window)[-10:]
            if np.mean(recent_face) > 0.3 and np.mean(recent_audio) > 0.3:
                alerts.append({
                    'level': 'POSITIVE',
                    'message': '😊 GOOD MOOD - Positive emotions detected',
                    'score_impact': 0.3
                })
        
        # POSITIVE FLAG 8: Active engagement (movement + speech)
        if len(self.activity_window) >= 5 and len(self.speech_rate_window) >= 5:
            recent_activity = list(self.activity_window)[-5:]
            recent_speech = list(self.speech_rate_window)[-5:]
            if np.mean(recent_activity) > 0.15 and np.mean(recent_speech) > 1.5:
                alerts.append({
                    'level': 'POSITIVE',
                    'message': '🎯 ACTIVE ENGAGEMENT',
                    'score_impact': 0.2
                })
        
        # =================================================================
        # CALCULATE OVERALL MENTAL HEALTH SCORE (-1 to +1)
        # =================================================================
        
        # Base components (weighted average)
        weights = {
            'activity': 0.10,
            'face_emotion': 0.20,
            'audio_emotion': 0.25,
            'audio_sentiment': 0.20,
            'audio_valence': 0.15,
            'energy': 0.10
        }
        
        # Get recent averages
        activity_score = np.mean(self.activity_window) if self.activity_window else 0
        face_score = np.mean(self.face_emotion_window) if self.face_emotion_window else 0
        audio_emo_score = np.mean(self.audio_emotion_window) if self.audio_emotion_window else 0
        sentiment_score = np.mean(self.audio_sentiment_window) if self.audio_sentiment_window else 0
        valence_score = np.mean(self.audio_valence_window) if self.audio_valence_window else 0
        
        # Energy score (normalized RMS)
        energy_score = np.mean(self.rms_window) if self.rms_window else 0
        energy_score = min(energy_score * 10, 1.0)  # Normalize to 0-1
        energy_score = (energy_score - 0.5) * 2  # Convert to -1 to +1
        
        # Weighted sum
        base_score = (
            weights['activity'] * activity_score +
            weights['face_emotion'] * face_score +
            weights['audio_emotion'] * audio_emo_score +
            weights['audio_sentiment'] * sentiment_score +
            weights['audio_valence'] * valence_score +
            weights['energy'] * energy_score
        )
        
        # Apply alert impacts
        alert_impact = sum([alert['score_impact'] for alert in alerts])
        
        # Final score
        final_score = np.clip(base_score + alert_impact * 0.3, -1.0, 1.0)
        
        return final_score, alerts
    
    def read_new_rows(self):
        """Read new rows from CSV file"""
        if not os.path.exists(self.csv_path):
            return []
        
        try:
            df = pd.read_csv(self.csv_path)
            
            # Get only new rows
            new_rows = df.iloc[self.last_processed_row + 1:]
            
            if len(new_rows) > 0:
                self.last_processed_row = len(df) - 1
                return new_rows.to_dict('records')
            
            return []
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return []
    
    def process_new_data(self):
        """Process all new rows from CSV"""
        new_rows = self.read_new_rows()
        
        for row in new_rows:
            score, alerts = self.analyze_row(row)
            
            # Store results
            timestamp = float(row.get('timestamp', time.time()))
            self.mh_score_history.append(score)
            self.timestamp_history.append(timestamp)
            self.total_frames += 1
            
            # Store alerts
            if alerts:
                self.current_alerts = alerts
                self.alert_count += len(alerts)
                for alert in alerts:
                    self.alert_history.append({
                        'timestamp': timestamp,
                        'alert': alert
                    })
            else:
                self.current_alerts = []
        
        return len(new_rows) > 0

class MentalHealthDashboard:
    """Live visualization dashboard"""
    
    def __init__(self, monitor):
        self.monitor = monitor
        
        # Setup plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 10))
        self.fig.suptitle('🧠 Real-Time Mental Health Monitor', fontsize=16, fontweight='bold')
        
        # Colors for zones
        self.colors = {
            'critical': '#ff4444',
            'poor': '#ff8844',
            'neutral': '#ffdd44',
            'good': '#88dd44',
            'excellent': '#44dd88'
        }
        
    def init_plot(self):
        """Initialize plot elements"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot 1: Mental Health Score Timeline
        self.ax1.set_title('Mental Health Score Timeline (-1 = Bad, +1 = Good)', fontsize=12, fontweight='bold')
        self.ax1.set_ylabel('Mental Health Score', fontsize=10)
        self.ax1.set_ylim(-1.1, 1.1)
        self.ax1.grid(True, alpha=0.3)
        
        # Add colored zones
        self.ax1.axhspan(-1.0, -0.6, alpha=0.1, color=self.colors['critical'], label='Critical')
        self.ax1.axhspan(-0.6, -0.2, alpha=0.1, color=self.colors['poor'], label='Poor')
        self.ax1.axhspan(-0.2, 0.2, alpha=0.1, color=self.colors['neutral'], label='Neutral')
        self.ax1.axhspan(0.2, 0.6, alpha=0.1, color=self.colors['good'], label='Good')
        self.ax1.axhspan(0.6, 1.0, alpha=0.1, color=self.colors['excellent'], label='Excellent')
        
        # Plot 2: Alerts and Statistics
        self.ax2.axis('off')
        
        return self.ax1, self.ax2
    
    def update(self, frame):
        """Update dashboard with new data"""
        # Process new data
        self.monitor.process_new_data()
        
        # Clear plots
        self.init_plot()
        
        # Plot mental health score
        if len(self.monitor.mh_score_history) > 0:
            timestamps = list(self.monitor.timestamp_history)
            scores = list(self.monitor.mh_score_history)
            
            # Normalize timestamps to start at 0
            if timestamps:
                base_time = timestamps[0]
                rel_times = [t - base_time for t in timestamps]
                
                # Plot line
                self.ax1.plot(rel_times, scores, 'b-', linewidth=2, label='MH Score')
                
                # Plot current point
                if scores:
                    current_score = scores[-1]
                    color = self.get_score_color(current_score)
                    self.ax1.plot(rel_times[-1], current_score, 'o', color=color, 
                                 markersize=12, markeredgecolor='black', markeredgewidth=2)
                
                self.ax1.set_xlabel('Time (seconds)', fontsize=10)
                self.ax1.legend(loc='upper left')
        
        # Display current status in second plot
        self.display_status()
        
        plt.tight_layout()
        return self.ax1, self.ax2
    
    def get_score_color(self, score):
        """Get color based on score"""
        if score < -0.6:
            return self.colors['critical']
        elif score < -0.2:
            return self.colors['poor']
        elif score < 0.2:
            return self.colors['neutral']
        elif score < 0.6:
            return self.colors['good']
        else:
            return self.colors['excellent']
    
    def display_status(self):
        """Display current status and alerts"""
        y_pos = 0.95
        
        # Current score
        if self.monitor.mh_score_history:
            current_score = self.monitor.mh_score_history[-1]
            score_text = f"Current Mental Health Score: {current_score:.3f}"
            color = self.get_score_color(current_score)
            
            self.ax2.text(0.05, y_pos, score_text, fontsize=16, fontweight='bold',
                         color=color, transform=self.ax2.transAxes)
            y_pos -= 0.12
            
            # Status description
            if current_score >= 0.6:
                status = "😊 Excellent - Person appears happy and engaged"
            elif current_score >= 0.2:
                status = "🙂 Good - Generally positive state"
            elif current_score >= -0.2:
                status = "😐 Neutral - No significant concerns"
            elif current_score >= -0.6:
                status = "😟 Poor - Some concerning signs detected"
            else:
                status = "😢 Critical - Multiple red flags present"
            
            self.ax2.text(0.05, y_pos, status, fontsize=12,
                         transform=self.ax2.transAxes)
            y_pos -= 0.10
        
        # Current alerts
        self.ax2.text(0.05, y_pos, "🚨 Current Alerts:", fontsize=12, fontweight='bold',
                     transform=self.ax2.transAxes)
        y_pos -= 0.08
        
        if self.monitor.current_alerts:
            for alert in self.monitor.current_alerts[:5]:  # Show max 5
                level_colors = {
                    'CRITICAL': '#ff0000',
                    'HIGH': '#ff6600',
                    'MEDIUM': '#ffaa00',
                    'POSITIVE': '#00cc00'
                }
                color = level_colors.get(alert['level'], '#000000')
                
                self.ax2.text(0.08, y_pos, f"[{alert['level']}] {alert['message']}", 
                             fontsize=10, color=color, transform=self.ax2.transAxes)
                y_pos -= 0.06
        else:
            self.ax2.text(0.08, y_pos, "No alerts - All systems normal ✓", 
                         fontsize=10, color='green', transform=self.ax2.transAxes)
            y_pos -= 0.06
        
        y_pos -= 0.04
        
        # Statistics
        self.ax2.text(0.05, y_pos, "📊 Statistics:", fontsize=12, fontweight='bold',
                     transform=self.ax2.transAxes)
        y_pos -= 0.08
        
        stats_text = f"Total Frames: {self.monitor.total_frames}\n"
        stats_text += f"Total Alerts: {self.monitor.alert_count}\n"
        
        if self.monitor.mh_score_history:
            avg_score = np.mean(self.monitor.mh_score_history)
            stats_text += f"Average Score: {avg_score:.3f}"
        
        self.ax2.text(0.08, y_pos, stats_text, fontsize=10,
                     transform=self.ax2.transAxes, verticalalignment='top')
        
    def run(self, interval=1000):
        """Run the live dashboard"""
        print("🧠 Mental Health Monitor Started")
        print(f"📁 Monitoring: {self.monitor.csv_path}")
        print("📊 Dashboard will update every second...")
        print("Press Ctrl+C to stop\n")
        
        anim = FuncAnimation(self.fig, self.update, init_func=self.init_plot,
                            interval=interval, blit=False, cache_frame_data=False)
        plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Mental Health Monitor')
    parser.add_argument('csv_file', help='Path to CSV file to monitor')
    parser.add_argument('--window', type=int, default=30, 
                       help='Window size for trend analysis (default: 30 frames)')
    parser.add_argument('--interval', type=int, default=1000,
                       help='Update interval in milliseconds (default: 1000)')
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = MentalHealthMonitor(args.csv_file, window_size=args.window)
    
    # Create and run dashboard
    dashboard = MentalHealthDashboard(monitor)
    
    try:
        dashboard.run(interval=args.interval)
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")

if __name__ == "__main__":
    main()