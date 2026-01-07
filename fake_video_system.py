import cv2
import dlib
import os
import numpy as np
from scipy.spatial import distance
from scipy import stats
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from collections import deque
import time
import sys
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnhancedDeepfakeDetector:
    def __init__(self):
        # Initialize face detector and predictor
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Check if cascade classifier loaded successfully
        if self.face_detector.empty():
            print("❌ Error: Could not load face cascade classifier")
            sys.exit(1)
            
        try:
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            self.use_dlib = True
            print("✅ Advanced facial landmark detection enabled")
        except Exception as e:
            print(f"⚠️  Basic detection mode (dlib landmarks not available: {e})")
            self.use_dlib = False
        
        # Facial landmark indices
        self.LEFT_EYE_POINTS = list(range(36, 42))
        self.RIGHT_EYE_POINTS = list(range(42, 48))
        self.MOUTH_POINTS = list(range(48, 68))
        self.NOSE_POINTS = list(range(27, 36))
        self.JAW_POINTS = list(range(0, 17))
        self.LEFT_BROW_POINTS = list(range(17, 22))
        self.RIGHT_BROW_POINTS = list(range(22, 27))
        
        # BALANCED detection parameters
        self.EYE_AR_THRESH = 0.25
        self.MOUTH_AR_THRESH = 0.7
        
        # ENHANCED THRESHOLDS - Better discrimination between real and fake
        self.THRESHOLDS = {
            'face_warping': {
                'suspicious_low': 0.05,   # Too stable = suspicious
                'normal_low': 0.15,       # Normal lower bound
                'normal_high': 0.45,      # Normal upper bound  
                'extreme': 0.70,          # Too much warping
            },
            'pulse_strength': {
                'missing': 0.002,         # Almost no pulse
                'very_weak': 0.006,       # Very weak pulse
                'weak': 0.012,            # Weak pulse
                'strong': 0.025           # Strong pulse
            },
            'blink_rate': {
                'no_blinks': 0,           # No blinking = major red flag
                'very_low': 5,            # Very few blinks
                'normal_low': 15,         # Normal lower bound
                'normal_high': 60,        # Normal upper bound
                'excessive': 120          # Too many blinks
            },
            'lighting_correlation': {
                'poor': 0.2,              # Poor lighting correlation
                'normal': 0.4,            # Normal correlation
                'good': 0.7               # Good correlation
            },
            'texture_consistency': {
                'poor': 0.4,              # Poor texture
                'normal': 0.6,            # Normal texture
                'suspicious_high': 0.9,   # Too perfect = suspicious
                'perfect': 0.95           # Suspiciously perfect
            },
            'pulse_bpm_range': {
                'too_low': 45,            # Abnormally low BPM
                'normal_low': 55,         # Normal lower bound
                'normal_high': 95,        # Normal upper bound
                'too_high': 120           # Abnormally high BPM
            },
            'pulse_snr_suspicious': 25.0,  # Suspiciously clean signal
            'face_size_variation': {
                'too_stable': 0.05,       # Too stable = suspicious
                'normal_high': 0.4,       # Normal variation
                'excessive': 0.8          # Too much variation
            }
        }
        
        self.reset_tracking()
        
    def reset_tracking(self):
        """Reset all tracking variables"""
        self.frame_data = []
        self.landmark_sequences = []
        self.optical_flow_data = []
        
    def detect_pulse_signals(self, video_path):
        """Enhanced pulse detection with better error handling"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {video_path}")
            return {'pulse_strength': 0.01, 'pulse_snr': 1.0, 'estimated_bpm': 70}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1000
            
        face_colors = {'red': [], 'green': [], 'blue': []}
        frame_count = 0
        max_frames = min(600, total_frames)
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                frame_count += 1
                faces = self.face_detector.detectMultiScale(frame, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    if x < 0 or y < 0 or w <= 0 or h <= 0:
                        continue
                    if x + w > frame.shape[1] or y + h > frame.shape[0]:
                        continue
                        
                    # Better ROI selection for pulse detection
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Focus on forehead and cheeks for pulse
                    forehead_h = max(1, h//4)
                    forehead_y_start = h//10
                    forehead_y_end = forehead_y_start + forehead_h
                    forehead_x_start = w//4
                    forehead_x_end = 3*w//4
                    
                    if (forehead_y_end <= face_roi.shape[0] and 
                        forehead_x_end <= face_roi.shape[1] and
                        forehead_y_start >= 0 and forehead_x_start >= 0):
                        
                        forehead = face_roi[forehead_y_start:forehead_y_end, 
                                          forehead_x_start:forehead_x_end]
                        
                        if forehead.size > 100:
                            avg_colors = np.mean(forehead.reshape(-1, 3), axis=0)
                            
                            face_colors['blue'].append(float(avg_colors[0]))
                            face_colors['green'].append(float(avg_colors[1]))
                            face_colors['red'].append(float(avg_colors[2]))
                            
            except Exception as e:
                continue
        
        cap.release()
        
        pulse_features = {'pulse_strength': 0.01, 'pulse_snr': 1.0, 'estimated_bpm': 70}
        
        if len(face_colors['green']) > 60:
            try:
                green_signal = np.array(face_colors['green'])
                
                # Remove outliers
                q75, q25 = np.percentile(green_signal, [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    green_signal = green_signal[(green_signal >= lower_bound) & (green_signal <= upper_bound)]
                
                if len(green_signal) < 30:
                    return pulse_features
                
                # Remove DC component and apply filtering
                green_signal = green_signal - np.mean(green_signal)
                
                # Bandpass filter for pulse range
                nyquist = fps / 2
                low_freq = max(0.8 / nyquist, 0.01)
                high_freq = min(3.0 / nyquist, 0.99)
                
                if low_freq < high_freq:
                    b, a = butter(3, [low_freq, high_freq], btype='band')
                    filtered_signal = filtfilt(b, a, green_signal)
                else:
                    filtered_signal = green_signal
                    
                # FFT analysis
                if len(filtered_signal) > 0:
                    freqs = np.fft.fftfreq(len(filtered_signal), 1/fps)
                    fft_values = np.abs(np.fft.fft(filtered_signal))
                    
                    pulse_range = (freqs >= 0.8) & (freqs <= 3.0)
                    if np.any(pulse_range):
                        pulse_power = np.max(fft_values[pulse_range])
                        total_power = np.sum(fft_values) + 1e-10
                        pulse_features['pulse_strength'] = pulse_power / total_power
                        
                        # Find dominant frequency
                        dominant_idx = np.argmax(fft_values[pulse_range])
                        dominant_freq = freqs[pulse_range][dominant_idx]
                        pulse_features['estimated_bpm'] = abs(dominant_freq) * 60
                        
                        # Signal-to-noise ratio
                        noise_power = np.mean(fft_values[~pulse_range]) if np.any(~pulse_range) else 1e-10
                        pulse_features['pulse_snr'] = pulse_power / (noise_power + 1e-10)
                        
            except Exception as e:
                print(f"⚠️  Pulse analysis error: {e}")
                
        return pulse_features
    
    def detect_advanced_face_warping(self, landmarks_sequence):
        """Enhanced face warping detection"""
        if len(landmarks_sequence) < 10:
            return 0.02
        
        warping_scores = []
        
        for i in range(1, len(landmarks_sequence)):
            try:
                if len(landmarks_sequence[i-1]) < 68 or len(landmarks_sequence[i]) < 68:
                    continue
                    
                prev_landmarks = landmarks_sequence[i-1]
                curr_landmarks = landmarks_sequence[i]
                
                # Multiple geometric relationships
                relationships = []
                
                # Eye-to-eye distance
                if all(idx < len(prev_landmarks) and idx < len(curr_landmarks) for idx in [36, 45]):
                    prev_eye_dist = distance.euclidean(prev_landmarks[36], prev_landmarks[45])
                    curr_eye_dist = distance.euclidean(curr_landmarks[36], curr_landmarks[45])
                    if prev_eye_dist > 0:
                        relationships.append(abs(curr_eye_dist - prev_eye_dist) / prev_eye_dist)
                
                # Nose-to-mouth distance
                if all(idx < len(prev_landmarks) and idx < len(curr_landmarks) for idx in [33, 51]):
                    prev_nose_mouth = distance.euclidean(prev_landmarks[33], prev_landmarks[51])
                    curr_nose_mouth = distance.euclidean(curr_landmarks[33], curr_landmarks[51])
                    if prev_nose_mouth > 0:
                        relationships.append(abs(curr_nose_mouth - prev_nose_mouth) / prev_nose_mouth)
                
                # Face width (jaw)
                if all(idx < len(prev_landmarks) and idx < len(curr_landmarks) for idx in [0, 16]):
                    prev_jaw_width = distance.euclidean(prev_landmarks[0], prev_landmarks[16])
                    curr_jaw_width = distance.euclidean(curr_landmarks[0], curr_landmarks[16])
                    if prev_jaw_width > 0:
                        relationships.append(abs(curr_jaw_width - prev_jaw_width) / prev_jaw_width)
                
                if relationships:
                    warping_scores.append(np.mean(relationships))
                    
            except (IndexError, ZeroDivisionError, Exception):
                continue
        
        return np.mean(warping_scores) if warping_scores else 0.02
    
    def analyze_eye_movements(self, video_path):
        """Enhanced eye movement analysis"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'blink_rate': 15, 'pupil_distance_std': 0.15}
        
        blink_patterns = []
        pupil_distances = []
        frame_count = 0
        max_frames = 300  # Increased for better analysis
        total_blinks = 0
        blink_state = False
        blink_start = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    if x < 0 or y < 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
                        continue
                        
                    if self.use_dlib:
                        try:
                            rect = dlib.rectangle(x, y, x + w, y + h)
                            landmarks = self.predictor(gray, rect)
                            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
                            
                            if len(landmarks) >= 68:
                                left_eye = landmarks[self.LEFT_EYE_POINTS]
                                right_eye = landmarks[self.RIGHT_EYE_POINTS]
                                
                                left_ear = self.eye_aspect_ratio(left_eye)
                                right_ear = self.eye_aspect_ratio(right_eye)
                                avg_ear = (left_ear + right_ear) / 2.0
                                blink_patterns.append(avg_ear)
                                
                                # Enhanced blink detection
                                if avg_ear < self.EYE_AR_THRESH:
                                    if not blink_state:
                                        blink_state = True
                                        blink_start = frame_count
                                else:
                                    if blink_state:
                                        blink_duration = frame_count - blink_start
                                        if 2 <= blink_duration <= 15:
                                            total_blinks += 1
                                        blink_state = False
                                
                                # Pupil distance consistency
                                left_center = np.mean(left_eye, axis=0)
                                right_center = np.mean(right_eye, axis=0)
                                pupil_dist = distance.euclidean(left_center, right_center)
                                pupil_distances.append(pupil_dist)
                                
                        except Exception:
                            continue
            except Exception:
                continue
        
        cap.release()
        
        results = {}
        
        # Calculate blink rate per minute
        if frame_count > 30:
            fps = 30.0
            video_duration_minutes = (frame_count / fps) / 60.0
            results['blink_rate'] = total_blinks / video_duration_minutes if video_duration_minutes > 0 else 15
        else:
            results['blink_rate'] = 15
        
        # Pupil distance analysis
        if len(pupil_distances) > 10:
            pupil_std = np.std(pupil_distances)
            pupil_mean = np.mean(pupil_distances)
            results['pupil_distance_std'] = pupil_std / pupil_mean if pupil_mean > 0 else 0.15
        else:
            results['pupil_distance_std'] = 0.15
        
        return results
    
    def detect_temporal_inconsistency(self, video_path):
        """Enhanced temporal analysis"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'face_size_variation': 0.25, 'face_warping': 0.02}
        
        self.landmark_sequences = []
        face_sizes = []
        frame_count = 0
        max_frames = 300
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    if x < 0 or y < 0 or w <= 0 or h <= 0:
                        continue
                    if x + w > gray.shape[1] or y + h > gray.shape[0]:
                        continue
                        
                    face_sizes.append(w * h)
                    
                    if self.use_dlib:
                        try:
                            rect = dlib.rectangle(x, y, x + w, y + h)
                            landmarks = self.predictor(gray, rect)
                            landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
                            self.landmark_sequences.append(landmarks_np)
                        except Exception:
                            continue
                            
            except Exception:
                continue
        
        cap.release()
        
        results = {}
        
        # Face size consistency
        if len(face_sizes) > 10:
            size_std = np.std(face_sizes)
            size_mean = np.mean(face_sizes)
            results['face_size_variation'] = size_std / size_mean if size_mean > 0 else 0.25
        else:
            results['face_size_variation'] = 0.25
        
        # Face warping detection
        results['face_warping'] = self.detect_advanced_face_warping(self.landmark_sequences)
        
        return results
    
    def detect_lighting_inconsistencies(self, video_path):
        """Enhanced lighting analysis"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'lighting_correlation': 0.2}
        
        face_brightness = []
        background_brightness = []
        frame_count = 0
        max_frames = 200  # Increased for better analysis
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    if x < 0 or y < 0 or w <= 0 or h <= 0:
                        continue
                    if x + w > gray.shape[1] or y + h > gray.shape[0]:
                        continue
                        
                    # Face brightness
                    face_roi = gray[y:y+h, x:x+w]
                    face_brightness.append(float(np.mean(face_roi)))
                    
                    # Background brightness
                    bg_y1, bg_y2 = max(0, y-h//4), min(gray.shape[0], y+h+h//4)
                    bg_x1, bg_x2 = max(0, x-w//4), min(gray.shape[1], x+w+w//4)
                    
                    if bg_y2 > bg_y1 and bg_x2 > bg_x1:
                        background_roi = gray[bg_y1:bg_y2, bg_x1:bg_x2].copy()
                        
                        face_y_start = y - bg_y1
                        face_x_start = x - bg_x1
                        
                        if (0 <= face_y_start < background_roi.shape[0] - h and 
                            0 <= face_x_start < background_roi.shape[1] - w and
                            face_y_start + h <= background_roi.shape[0] and
                            face_x_start + w <= background_roi.shape[1]):
                            
                            background_roi[face_y_start:face_y_start+h, 
                                         face_x_start:face_x_start+w] = 0
                        
                        bg_pixels = background_roi[background_roi > 0]
                        if len(bg_pixels) > 100:
                            background_brightness.append(float(np.mean(bg_pixels)))
                            
            except Exception:
                continue
        
        cap.release()
        
        results = {'lighting_correlation': 0.2}
        
        if len(face_brightness) > 10 and len(background_brightness) > 10:
            try:
                min_len = min(len(face_brightness), len(background_brightness))
                face_br = np.array(face_brightness[:min_len])
                bg_br = np.array(background_brightness[:min_len])
                
                if len(face_br) > 5 and len(bg_br) > 5:
                    correlation = np.corrcoef(face_br, bg_br)[0, 1]
                    results['lighting_correlation'] = correlation if not np.isnan(correlation) else 0.2
                    
            except Exception:
                pass
        
        return results
    
    def analyze_texture_consistency(self, video_path):
        """Enhanced texture analysis"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'texture_consistency': 0.5}
        
        texture_features = []
        frame_count = 0
        max_frames = 150  # Increased for better analysis
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    if x < 0 or y < 0 or w <= 0 or h <= 0:
                        continue
                    if x + w > gray.shape[1] or y + h > gray.shape[0]:
                        continue
                        
                    face_roi = gray[y:y+h, x:x+w]
                    if face_roi.size > 0:
                        face_roi = cv2.resize(face_roi, (64, 64))
                        
                        # Enhanced texture analysis
                        grad_x = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
                        grad_y = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
                        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                        
                        # Additional texture features
                        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
                        
                        texture_features.append([
                            float(np.mean(grad_magnitude)),
                            float(np.std(grad_magnitude)),
                            float(np.mean(face_roi)),
                            float(np.std(face_roi)),
                            float(np.var(laplacian))  # Additional feature
                        ])
                        
            except Exception:
                continue
        
        cap.release()
        
        if len(texture_features) > 10:
            try:
                texture_features = np.array(texture_features)
                
                # Calculate temporal consistency
                correlations = []
                for i in range(len(texture_features) - 1):
                    corr = np.corrcoef(texture_features[i], texture_features[i+1])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                
                consistency = np.mean(correlations) if correlations else 0.5
                return {'texture_consistency': max(0, consistency)}
            except Exception:
                return {'texture_consistency': 0.5}
        
        return {'texture_consistency': 0.5}
    
    def enhanced_ensemble_decision(self, temporal_data, eye_data, lighting_data, pulse_data, texture_data):
        """COMPLETELY REDESIGNED decision logic based on analysis"""
        
        feature_scores = {}
        
        # REVISED WEIGHTS - More balanced approach
        feature_weights = {
            'temporal': 0.20,     # Reduced - too variable
            'eye': 0.30,          # Increased - reliable indicator
            'pulse': 0.20,        # Maintained
            'lighting': 0.15,     # Increased slightly
            'texture': 0.15       # Maintained
        }
        
        # TEMPORAL ANALYSIS - Focus on extremes
        temporal_score = 0.0
        face_warping = temporal_data.get('face_warping', 0.02)
        face_size_var = temporal_data.get('face_size_variation', 0.25)
        
        # Key insight: Both too stable AND too unstable can be suspicious
        if face_warping < self.THRESHOLDS['face_warping']['suspicious_low']:  # Too stable
            temporal_score = 0.4
            print("   🔍 Face movement suspiciously stable")
        elif face_warping > self.THRESHOLDS['face_warping']['extreme']:  # Too chaotic
            temporal_score = 0.7
            print("   🔍 Excessive face warping detected")
        elif face_warping > self.THRESHOLDS['face_warping']['normal_high']:
            temporal_score = 0.3
        
        # Face size variation
        size_var = face_size_var
        if size_var < self.THRESHOLDS['face_size_variation']['too_stable']:
            temporal_score = max(temporal_score, 0.3)
            print("   🔍 Face size suspiciously stable")
        elif size_var > self.THRESHOLDS['face_size_variation']['excessive']:
            temporal_score = max(temporal_score, 0.4)
        
        feature_scores['temporal'] = temporal_score
        
        # EYE ANALYSIS - Enhanced logic
        eye_score = 0.0
        blink_rate = eye_data.get('blink_rate', 15)
        pupil_std = eye_data.get('pupil_distance_std', 0.15)
        
        # CRITICAL: No blinking is a major red flag for deepfakes
        if blink_rate <= self.THRESHOLDS['blink_rate']['no_blinks']:
            eye_score = 0.8
            print("   🚨 NO BLINKING DETECTED - Major deepfake indicator")
        elif blink_rate < self.THRESHOLDS['blink_rate']['very_low']:
            eye_score = 0.6
            print("   🔍 Very low blink rate detected")
        elif blink_rate < self.THRESHOLDS['blink_rate']['normal_low']:
            eye_score = 0.3
        elif blink_rate > self.THRESHOLDS['blink_rate']['excessive']:
            eye_score = 0.4
        
        # Pupil consistency
        if pupil_std > 0.5:  # Very inconsistent
            eye_score = max(eye_score, 0.3)
        
        feature_scores['eye'] = eye_score
        
        # PULSE ANALYSIS - Enhanced
        pulse_score = 0.0
        pulse_strength = pulse_data.get('pulse_strength', 0.01)
        pulse_snr = pulse_data.get('pulse_snr', 1.0)
        estimated_bpm = pulse_data.get('estimated_bpm', 70)  # Fixed variable name
        
        # Missing pulse
        if pulse_strength < self.THRESHOLDS['pulse_strength']['missing']:
            pulse_score = 0.7
            print("   🔍 No pulse signal detected")
        elif pulse_strength < self.THRESHOLDS['pulse_strength']['very_weak']:
            pulse_score = 0.4
        
        # BPM analysis
        if (estimated_bpm < self.THRESHOLDS['pulse_bpm_range']['too_low'] or 
            estimated_bpm > self.THRESHOLDS['pulse_bpm_range']['too_high']):
            pulse_score = max(pulse_score, 0.3)
        
        # Suspiciously clean signal
        if pulse_snr > self.THRESHOLDS['pulse_snr_suspicious']:
            pulse_score = max(pulse_score, 0.2)
        
        feature_scores['pulse'] = pulse_score
        
        # LIGHTING ANALYSIS
        lighting_score = 0.0
        lighting_corr = lighting_data.get('lighting_correlation', 0.2)
        
        if lighting_corr < self.THRESHOLDS['lighting_correlation']['poor']:
            lighting_score = 0.4
        elif lighting_corr > self.THRESHOLDS['lighting_correlation']['good']:
            # Perfect lighting can also be suspicious
            lighting_score = 0.1
        
        feature_scores['lighting'] = lighting_score
        
        # TEXTURE ANALYSIS - Focus on perfection
        texture_score = 0.0
        texture_consistency = texture_data.get('texture_consistency', 0.5)
        
        if texture_consistency > self.THRESHOLDS['texture_consistency']['perfect']:
            texture_score = 0.5
            print("   🔍 Suspiciously perfect texture detected")
        elif texture_consistency > self.THRESHOLDS['texture_consistency']['suspicious_high']:
            texture_score = 0.3
        elif texture_consistency < self.THRESHOLDS['texture_consistency']['poor']:
            texture_score = 0.3
        
        feature_scores['texture'] = texture_score
        
        # Calculate weighted score
        weighted_score = sum(feature_scores[key] * feature_weights[key] for key in feature_scores)
        
        # ENHANCED COMBINATION PATTERN DETECTION
        combination_bonus = 0.0
        
        # Pattern 1: Classic deepfake signature - No blinking + decent other metrics
        if blink_rate == 0 and pulse_strength > 0.005:
            combination_bonus += 0.3
            print("   🚨 CLASSIC DEEPFAKE PATTERN: No blinking with artificial pulse")
        
        # Pattern 2: "Too perfect" syndrome
        perfect_metrics = 0
        if texture_consistency > 0.9: perfect_metrics += 1
        if lighting_corr > 0.7: perfect_metrics += 1
        if pulse_snr > 20: perfect_metrics += 1
        if face_warping < 0.1: perfect_metrics += 1
        
        if perfect_metrics >= 3:
            combination_bonus += 0.25
            print(f"   🔍 'TOO PERFECT' PATTERN: {perfect_metrics}/4 metrics suspiciously good")
        
        # Pattern 3: Unnatural stability
        if face_warping < 0.05 and face_size_var < 0.1:
            combination_bonus += 0.2
            print("   🔍 UNNATURAL STABILITY: Face too stable across frames")
        
        # Pattern 4: Deepfake quality indicators
        if (texture_consistency > 0.85 and 
            lighting_corr > 0.5 and 
            blink_rate < 10):
            combination_bonus += 0.2
            print("   🔍 HIGH-QUALITY DEEPFAKE INDICATORS")
        
        # Apply combination bonus
        final_score = min(1.0, weighted_score + combination_bonus)
        
        # AUTHENTICITY BOOST for clearly natural patterns
        authenticity_boost = 0.0
        
        # Natural movement and blinking
        if (self.THRESHOLDS['blink_rate']['normal_low'] <= blink_rate <= self.THRESHOLDS['blink_rate']['normal_high'] and
            self.THRESHOLDS['face_warping']['normal_low'] <= face_warping <= self.THRESHOLDS['face_warping']['normal_high']):
            authenticity_boost += 0.1
            print("   ✅ Natural movement and blinking patterns detected")
        
        # Natural pulse
        if (self.THRESHOLDS['pulse_bpm_range']['normal_low'] <= estimated_bpm <= self.THRESHOLDS['pulse_bpm_range']['normal_high'] and
            pulse_strength > self.THRESHOLDS['pulse_strength']['weak']):
            authenticity_boost += 0.05
        
        # Apply authenticity boost
        final_score = max(0.0, final_score - authenticity_boost)
        
        # Calculate confidence
        evidence_count = sum(1 for score in feature_scores.values() if score > 0.25)
        confidence = 0.7 + (evidence_count * 0.06) + (combination_bonus * 0.4)
        confidence = min(0.98, max(0.65, confidence))
        
        return final_score, confidence, {
            'feature_scores': feature_scores,
            'feature_weights': feature_weights,
            'evidence_count': evidence_count,
            'combination_bonus': combination_bonus,
            'authenticity_boost': authenticity_boost,
            'weighted_score': weighted_score,
            'key_indicators': self.get_key_indicators(blink_rate, face_warping, pulse_strength, texture_consistency)
        }
    
    def get_key_indicators(self, blink_rate, face_warping, pulse_strength, texture_consistency):
        """Get key indicators for interpretation"""
        indicators = []
        
        if blink_rate == 0:
            indicators.append("NO_BLINKING")
        elif blink_rate < 5:
            indicators.append("VERY_LOW_BLINKING")
        
        if face_warping < 0.05:
            indicators.append("SUSPICIOUSLY_STABLE")
        elif face_warping > 0.7:
            indicators.append("EXCESSIVE_WARPING")
        
        if pulse_strength < 0.002:
            indicators.append("NO_PULSE")
        elif pulse_strength < 0.006:
            indicators.append("WEAK_PULSE")
        
        if texture_consistency > 0.95:
            indicators.append("PERFECT_TEXTURE")
        elif texture_consistency > 0.9:
            indicators.append("VERY_SMOOTH_TEXTURE")
        
        return indicators
    
    def eye_aspect_ratio(self, eye_points):
        """Calculate eye aspect ratio with error handling"""
        try:
            if len(eye_points) < 6:
                return 0.25
            
            A = distance.euclidean(eye_points[1], eye_points[5])
            B = distance.euclidean(eye_points[2], eye_points[4])
            C = distance.euclidean(eye_points[0], eye_points[3])
            
            if C == 0:
                return 0.25
                
            return (A + B) / (2.0 * C)
        except Exception:
            return 0.25
    
    def comprehensive_analysis(self, video_path):
        """Run comprehensive analysis with enhanced logic"""
        print("🚀 ENHANCED DEEPFAKE DETECTION ANALYSIS")
        print("="*60)
        
        # Validate video file
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file '{video_path}' not found!")
            return None
        
        # Test if video can be opened
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video file '{video_path}'!")
            return None
        cap.release()
        
        self.reset_tracking()
        
        try:
            print("🔍 Phase 1: Temporal analysis...")
            temporal_data = self.detect_temporal_inconsistency(video_path)
            
            print("👁️ Phase 2: Eye movement analysis...")
            eye_data = self.analyze_eye_movements(video_path)
            
            print("💓 Phase 3: Pulse signal detection...")
            pulse_data = self.detect_pulse_signals(video_path)
            
            print("💡 Phase 4: Lighting analysis...")
            lighting_data = self.detect_lighting_inconsistencies(video_path)
            
            print("🎨 Phase 5: Texture analysis...")
            texture_data = self.analyze_texture_consistency(video_path)
            
            # Make decision with enhanced logic
            final_score, confidence, breakdown = self.enhanced_ensemble_decision(
                temporal_data, eye_data, lighting_data, pulse_data, texture_data
            )
            
            self.print_enhanced_results(temporal_data, eye_data, lighting_data, pulse_data, 
                                      texture_data, final_score, confidence, breakdown)
            
            return {
                'final_score': final_score,
                'confidence': confidence,
                'breakdown': breakdown,
                'temporal_data': temporal_data,
                'eye_data': eye_data,
                'pulse_data': pulse_data,
                'lighting_data': lighting_data,
                'texture_data': texture_data
            }
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_enhanced_results(self, temporal_data, eye_data, lighting_data, pulse_data, 
                             texture_data, final_score, confidence, breakdown):
        """Print enhanced analysis results"""
        print("\n" + "="*60)
        print("🎯 ENHANCED DEEPFAKE DETECTION RESULTS")
        print("="*60)
        
        print(f"\n🚨 FINAL VERDICT: {final_score:.1%} FAKE PROBABILITY")
        print(f"🎯 CONFIDENCE LEVEL: {confidence:.1%}")
        
        # Enhanced classification
        if final_score > 0.75 and confidence > 0.85:
            print("🚨🚨🚨 VERY HIGH PROBABILITY OF DEEPFAKE")
            verdict_emoji = "🚨🚨🚨"
        elif final_score > 0.60:
            print("🚨🚨 HIGH PROBABILITY OF DEEPFAKE")
            verdict_emoji = "🚨🚨"
        elif final_score > 0.40:
            print("🚨 MODERATE PROBABILITY OF DEEPFAKE")
            verdict_emoji = "🚨"
        elif final_score > 0.25:
            print("⚠️  LOW SUSPICION - REVIEW RECOMMENDED")
            verdict_emoji = "⚠️"
        elif final_score > 0.15:
            print("⚡ MINIMAL SUSPICION")
            verdict_emoji = "⚡"
        else:
            print("✅ LIKELY AUTHENTIC")
            verdict_emoji = "✅"
        
        # Key indicators
        key_indicators = breakdown.get('key_indicators', [])
        if key_indicators:
            print(f"\n🔍 KEY DEEPFAKE INDICATORS:")
            for indicator in key_indicators:
                print(f"   • {indicator.replace('_', ' ')}")
        
        print(f"\n📊 FEATURE ANALYSIS:")
        for feature, score in breakdown['feature_scores'].items():
            weight = breakdown['feature_weights'][feature]
            contribution = score * weight
            if score > 0.5:
                status = "🚨 HIGH RISK"
            elif score > 0.25:
                status = "⚠️  SUSPICIOUS"
            else:
                status = "✅ NORMAL"
            print(f"   {feature.upper()}: {score:.1%} (weight: {weight:.1%}) → {status}")
            print(f"      Contribution to final score: {contribution:.3f}")
        
        print(f"\n📈 DETAILED METRICS:")
        print(f"   Face Warping: {temporal_data.get('face_warping', 0):.3f}")
        print(f"   Face Size Variation: {temporal_data.get('face_size_variation', 0):.3f}")
        print(f"   Blink Rate: {eye_data.get('blink_rate', 0):.1f} blinks/min")
        print(f"   Pupil Distance Std: {eye_data.get('pupil_distance_std', 0):.3f}")
        print(f"   Pulse Strength: {pulse_data.get('pulse_strength', 0):.4f}")
        print(f"   Pulse SNR: {pulse_data.get('pulse_snr', 0):.2f}")
        print(f"   Estimated BPM: {pulse_data.get('estimated_bpm', 0):.1f}")
        print(f"   Lighting Correlation: {lighting_data.get('lighting_correlation', 0):.3f}")
        print(f"   Texture Consistency: {texture_data.get('texture_consistency', 0):.3f}")
        
        print(f"\n🔍 ANALYSIS BREAKDOWN:")
        print(f"   Base Weighted Score: {breakdown['weighted_score']:.3f}")
        print(f"   Combination Bonus: +{breakdown.get('combination_bonus', 0):.3f}")
        print(f"   Authenticity Boost: -{breakdown.get('authenticity_boost', 0):.3f}")
        print(f"   Evidence Count: {breakdown['evidence_count']}/5 features flagged")
        
        print(f"\n💡 ENHANCED INTERPRETATION:")
        
        if final_score < 0.15:
            print("   ✅ AUTHENTIC VIDEO INDICATORS:")
            print("     • Natural blinking patterns detected")
            print("     • Realistic biological signals present")
            print("     • Normal facial movement variation")
            print("     • No significant deepfake artifacts")
            
        elif final_score < 0.30:
            print("   ⚡ MOSTLY AUTHENTIC with minor concerns:")
            print("     • Some metrics slightly outside normal ranges")
            print("     • Likely due to video quality or compression")
            print("     • No strong deepfake indicators present")
            
        elif final_score < 0.50:
            print("   ⚠️  MODERATE SUSPICION:")
            print("     • Some concerning patterns detected")
            print("     • Could indicate low-quality deepfake or unusual conditions")
            print("     • Manual expert review strongly recommended")
            
        elif final_score < 0.70:
            print("   🚨 HIGH PROBABILITY OF DEEPFAKE:")
            print("     • Multiple suspicious indicators present")
            print("     • Strong evidence of synthetic content")
            print("     • Likely AI-generated or heavily manipulated")
            
        else:
            print("   🚨🚨 VERY HIGH PROBABILITY OF DEEPFAKE:")
            print("     • Classic deepfake signatures detected")
            print("     • Multiple red flags across different features")
            print("     • Almost certainly synthetic content")
        
        # Specific pattern interpretations
        if breakdown.get('combination_bonus', 0) > 0.2:
            print(f"\n🔍 SPECIFIC DEEPFAKE PATTERNS IDENTIFIED:")
            if eye_data.get('blink_rate', 0) == 0:
                print("     • NO BLINKING: Classic deepfake signature")
            if temporal_data.get('face_warping', 0) < 0.05:
                print("     • UNNATURAL STABILITY: Face movement too consistent")
            if texture_data.get('texture_consistency', 0) > 0.9:
                print("     • PERFECT TEXTURE: Suspiciously smooth rendering")
        
        print("="*60)

def main():
    """Main function to run the enhanced deepfake detector"""
    if len(sys.argv) != 2:
        print("Usage: python enhanced_deepfake_detector.py <video_path>")
        print("Example: python enhanced_deepfake_detector.py sample_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file '{video_path}' not found!")
        sys.exit(1)
    
    print("🎬 ENHANCED DEEPFAKE DETECTOR v4.0")
    print("="*60)
    print(f"🎥 Analyzing video: {video_path}")
    print("⏱️  This may take a few minutes...")
    
    start_time = time.time()
    
    try:
        detector = EnhancedDeepfakeDetector()
        results = detector.comprehensive_analysis(video_path)
        
        if results is None:
            print("❌ Analysis failed!")
            sys.exit(1)
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        print(f"\n⏱️  Analysis completed in {analysis_time:.1f} seconds")
        
        # Save results to file
        import json
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        results_file = f"{base_name}_enhanced_deepfake_analysis.json"
        
        output_data = {
            'video_path': video_path,
            'analysis_time': analysis_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'detector_version': 'v4.0_enhanced',
            'results': results
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        output_data = convert_numpy_types(output_data)
        
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"📄 Detailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()