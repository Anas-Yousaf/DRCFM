import cv2
import numpy as np
import mtcnn
from architecture import *
from train_v2 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle
from collections import deque, defaultdict
import time

confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)
smoothing_window = 7
max_track_age = 10

# Liveness detection parameters
blink_threshold = 0.3
eye_ar_consec_frames = 3
eye_ratio_threshold = 0.2
motion_threshold = 3.0
texture_threshold = 30.0

class LivenessDetector:
    def __init__(self):
        # For eye blinking detection
        self.eye_counter = 0
        self.total_blinks = 0
        self.last_blink_time = time.time()
        
        # For texture analysis
        self.texture_scores = deque(maxlen=20)
        
        # For motion analysis
        self.prev_face_centers = deque(maxlen=10)
        self.motion_scores = deque(maxlen=20)
    
    def get_eye_aspect_ratio(self, face_landmarks):
        """Calculate eye aspect ratio to detect blinks"""
        # Simplified for demonstration - in a real system, use facial landmarks
        # This is a placeholder - would use actual landmark points from a detector
        # Format would be: ratio = distance(eye_vertical) / distance(eye_horizontal)
        return np.random.uniform(0.2, 0.35)  # Simulated eye ratio
    
    def analyze_texture(self, face):
        """Analyze face texture for spoof detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        
        # Apply Laplacian for texture analysis
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        score = np.var(laplacian)
        self.texture_scores.append(score)
        
        # Real faces typically have higher texture variance
        return np.mean(self.texture_scores) if self.texture_scores else 0
    
    def analyze_motion(self, face_center):
        """Analyze natural face motion"""
        if face_center is not None:
            self.prev_face_centers.append(face_center)
            
            if len(self.prev_face_centers) > 1:
                # Calculate motion between consecutive frames
                motions = []
                for i in range(1, len(self.prev_face_centers)):
                    prev = np.array(self.prev_face_centers[i-1])
                    curr = np.array(self.prev_face_centers[i])
                    motion = np.linalg.norm(curr - prev)
                    motions.append(motion)
                
                avg_motion = np.mean(motions) if motions else 0
                self.motion_scores.append(avg_motion)
                
                # Natural face has small but non-zero motion
                return np.mean(self.motion_scores) if self.motion_scores else 0
        return 0
    
    def check_liveness(self, face, face_center, face_landmarks=None):
        """Combine multiple liveness checks"""
        # 1. Eye blinking (simulated)
        eye_ratio = self.get_eye_aspect_ratio(face_landmarks)
        
        if eye_ratio < blink_threshold:
            self.eye_counter += 1
        else:
            if self.eye_counter >= eye_ar_consec_frames:
                self.total_blinks += 1
                current_time = time.time()
                self.last_blink_time = current_time
            self.eye_counter = 0
            
        time_since_last_blink = time.time() - self.last_blink_time
        blink_score = 1.0 if time_since_last_blink < 5.0 and self.total_blinks > 0 else 0.0
        
        # 2. Texture analysis
        texture_score = self.analyze_texture(face)
        texture_normalized = min(1.0, texture_score / texture_threshold)
        
        # 3. Motion analysis
        motion_score = self.analyze_motion(face_center)
        motion_normalized = min(1.0, motion_score / motion_threshold)
        
        # Combine scores with weights
        combined_score = (0.5 * blink_score) + (0.3 * texture_normalized) + (0.2 * motion_normalized)
        
        is_live = combined_score > 0.6
        return is_live, combined_score

class Tracker:
    def __init__(self):
        self.tracks = {}
        self.track_id = 0

    def update(self, box):
        x1, y1, w, h = box
        center = (x1 + w//2, y1 + h//2)
        matched_id = None
        min_dist = float("inf")

        for tid, data in self.tracks.items():
            prev_center = data['center']
            dist = np.linalg.norm(np.array(center) - np.array(prev_center))
            if dist < 50 and dist < min_dist:
                matched_id = tid
                min_dist = dist

        if matched_id is not None:
            self.tracks[matched_id]['center'] = center
            self.tracks[matched_id]['age'] = 0
            return matched_id
        else:
            self.tracks[self.track_id] = {'center': center, 'age': 0}
            self.track_id += 1
            return self.track_id - 1

    def cleanup(self):
        to_delete = [tid for tid, data in self.tracks.items() if data['age'] > max_track_age]
        for tid in to_delete:
            del self.tracks[tid]

    def age_all(self):
        for tid in self.tracks:
            self.tracks[tid]['age'] += 1

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(img, detector, encoder, encoding_dict, tracker, history, liveness_detectors):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    tracker.age_all()

    # Phase label
    cv2.putText(img, "PHASE 2: DRCF (TRACKING + LIVENESS)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        
        # Skip processing if face is too small
        if face.shape[0] < 20 or face.shape[1] < 20:
            continue
            
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'
        distance = float("inf")

        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        track_id = tracker.update(res['box'])
        history[track_id]['names'].append(name)
        
        if track_id not in liveness_detectors:
            liveness_detectors[track_id] = LivenessDetector()
            
        # Calculate face center for liveness detection
        x1, y1, width, height = res['box']
        face_center = (x1 + width//2, y1 + height//2)
        
        # Check liveness
        is_live, liveness_score = liveness_detectors[track_id].check_liveness(
            face, face_center, res.get('keypoints')
        )
        
        # Store liveness result in history
        history[track_id]['liveness'].append(is_live)

        if len(history[track_id]['names']) > smoothing_window:
            history[track_id]['names'].popleft()
            history[track_id]['liveness'].popleft()

        common_name = max(set(history[track_id]['names']), key=history[track_id]['names'].count)
        confidence_score = 1 - distance  # Convert distance to confidence
        
        # Determine if the face is considered live based on recent history
        recent_liveness = list(history[track_id]['liveness'])
        is_considered_live = sum(recent_liveness) / max(1, len(recent_liveness)) > 0.5

        # Set color based on both recognition and liveness
        if common_name != 'unknown' and is_considered_live:
            # Known face, live - green
            color = (0, 255, 0)
            status = "LIVE"
        elif common_name != 'unknown' and not is_considered_live:
            # Known face, spoof - yellow
            color = (0, 255, 255)
            status = "SPOOF"
        elif common_name == 'unknown' and is_considered_live:
            # Unknown face, live - blue
            color = (255, 0, 0)
            status = "LIVE"
        else:
            # Unknown face, spoof - red
            color = (0, 0, 255)
            status = "SPOOF"

        cv2.rectangle(img, pt_1, pt_2, color, 2)
        cv2.putText(img, f"{common_name} (ID: {track_id})", 
                    (pt_1[0], pt_1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, f"{status} ({liveness_score:.2f})", 
                    (pt_1[0], pt_1[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display liveness detection info
    cv2.putText(img, "Liveness Detection Active", (10, img.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
    tracker.cleanup()
    return img

if __name__ == "__main__":
    face_encoder = InceptionResNetV2()
    face_encoder.load_weights("facenet_keras_weights.h5")
    encoding_dict = load_pickle('encodings/encodings.pkl')
    face_detector = mtcnn.MTCNN()

    tracker = Tracker()
    # Modified history to track both names and liveness
    history = defaultdict(lambda: {'names': deque(maxlen=smoothing_window), 
                                  'liveness': deque(maxlen=smoothing_window)})
    liveness_detectors = {}

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("CAM NOT OPENED")
            break

        frame = detect(frame, face_detector, face_encoder, encoding_dict, tracker, history, liveness_detectors)
        cv2.imshow('Phase 2: DRCF with Tracking and Liveness', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()