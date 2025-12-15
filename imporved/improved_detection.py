import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import json
import os
from typing import List, Tuple, Optional, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedISLClassifier(nn.Module):
    def __init__(self, input_size: int = 63, hidden_sizes: List[int] = [256, 128, 64], 
                 num_classes: int = 26, dropout_rate: float = 0.3, use_residual: bool = False):
        super(ImprovedISLClassifier, self).__init__()
        self.use_residual = use_residual
        layers = []
        prev_size = input_size

        # Input layer with batch normalization
        layers.extend([
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])

        # Hidden layers with progressive dropout reduction
        for i, hidden_size in enumerate(hidden_sizes[1:], 1):
            current_dropout = dropout_rate * (0.8 ** i)
            layers.extend([
                nn.Linear(hidden_sizes[i-1], hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(current_dropout)
            ])

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)

class HandLandmarkProcessor:
    """Robust hand landmark processing with consistent preprocessing"""
    
    def __init__(self, use_3d=True, normalize_method='minmax', num_hands=2):
        self.use_3d = use_3d
        self.normalize_method = normalize_method
        self.num_hands = num_hands
        self.single_hand_size = 63 if use_3d else 42  # 21 landmarks * 3 or 2 coordinates
        self.feature_size = self.single_hand_size * num_hands
        
    def extract_landmarks(self, hand_landmarks) -> np.ndarray:
        """Extract landmark coordinates with consistent preprocessing"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            if self.use_3d:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                landmarks.extend([landmark.x, landmark.y])
        return np.array(landmarks, dtype=np.float32)
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks with consistent method"""
        if len(landmarks) == 0:
            return np.zeros(self.single_hand_size, dtype=np.float32)
        landmarks_reshaped = landmarks.reshape(-1, 3 if self.use_3d else 2)
        base_point = landmarks_reshaped[0]  # Wrist landmark
        relative_landmarks = landmarks_reshaped - base_point
        relative_landmarks = relative_landmarks.flatten()
        if self.normalize_method == 'minmax':
            max_val = np.max(np.abs(relative_landmarks))
            if max_val > 0:
                relative_landmarks = relative_landmarks / max_val
        elif self.normalize_method == 'std':
            std_val = np.std(relative_landmarks)
            if std_val > 0:
                relative_landmarks = (relative_landmarks - np.mean(relative_landmarks)) / std_val
        return relative_landmarks.astype(np.float32)
    
    def process_frame_landmarks(self, results) -> Tuple[np.ndarray, bool]:
        """Process landmarks from MediaPipe results for both hands"""
        if not results.multi_hand_landmarks:
            return np.zeros(self.feature_size, dtype=np.float32), False
        hand_landmarks_list = results.multi_hand_landmarks
        all_hand_features = []
        for i in range(self.num_hands):
            if i < len(hand_landmarks_list):
                raw_landmarks = self.extract_landmarks(hand_landmarks_list[i])
                normalized_landmarks = self.normalize_landmarks(raw_landmarks)
                all_hand_features.append(normalized_landmarks)
            else:
                # Pad missing hand with zeros
                all_hand_features.append(np.zeros(self.single_hand_size, dtype=np.float32))
        combined = np.concatenate(all_hand_features)
        return combined, True

class PredictionStabilizer:
    """Stabilize predictions using temporal smoothing"""
    
    def __init__(self, window_size=10, confidence_threshold=0.7):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.prediction_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        
    def add_prediction(self, prediction: int, confidence: float):
        """Add new prediction to history"""
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
    
    def get_stable_prediction(self) -> Tuple[Optional[int], float]:
        """Get stabilized prediction using majority voting and confidence"""
        if len(self.prediction_history) < 3:
            return None, 0.0
        
        # Check if recent predictions have high confidence
        recent_confidences = list(self.confidence_history)[-3:]
        if np.mean(recent_confidences) < self.confidence_threshold:
            return None, 0.0
        
        # Majority voting on recent predictions
        recent_predictions = list(self.prediction_history)[-5:]
        unique, counts = np.unique(recent_predictions, return_counts=True)
        
        if len(unique) > 0:
            majority_pred = unique[np.argmax(counts)]
            avg_confidence = np.mean([conf for pred, conf in 
                                    zip(self.prediction_history, self.confidence_history) 
                                    if pred == majority_pred])
            return int(majority_pred), float(avg_confidence)
        
        return None, 0.0
    
    def reset(self):
        """Reset prediction history"""
        self.prediction_history.clear()
        self.confidence_history.clear()

class ImprovedISLDetector:
    """Main ISL detection class with improved robustness"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize components
        self.landmark_processor = HandLandmarkProcessor(
            use_3d=self.config.get('use_3d', True),
            normalize_method=self.config.get('normalize_method', 'minmax'),
            num_hands=2
        )
        
        self.stabilizer = PredictionStabilizer(
            window_size=self.config.get('stabilizer_window', 10),
            confidence_threshold=self.config.get('confidence_threshold', 0.7)
        )
        
        # Load model
        self.model = self.load_model(model_path)
        self.words = self.config.get('words', list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Detection parameters
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=2,  # Now process up to two hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        
    def load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'use_3d': True,
            'normalize_method': 'minmax',
            'stabilizer_window': 10,
            'confidence_threshold': 0.7,
            'input_size': 63,
            'hidden_sizes': [256, 128, 64],
            'dropout_rate': 0.3,
            'words': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using defaults.")
        
        return default_config
    
    def load_model(self, model_path: str) -> ImprovedISLClassifier:
        """Load trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            model = ImprovedISLClassifier(
                input_size=self.config['input_size'],
                hidden_sizes=self.config['hidden_sizes'],
                num_classes=len(self.config['words']),
                dropout_rate=self.config['dropout_rate']
            ).to(self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'words' in checkpoint:
                    self.words = checkpoint['words']
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """Make prediction from landmarks"""
        try:
            # Convert to tensor
            input_tensor = torch.tensor(landmarks).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = predicted.item()
                confidence_score = confidence.item()
                
                return self.words[predicted_class], confidence_score
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "ERROR", 0.0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str], float]:
        """Process single frame and return annotated frame with prediction"""
        self.frame_count += 1
        
        # Flip frame for selfie view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(frame_rgb)
        
        # Process landmarks
        landmarks, has_hand = self.landmark_processor.process_frame_landmarks(results)
        
        prediction_text = None
        confidence = 0.0
        
        if has_hand:
            self.detection_count += 1
            
            # Make prediction
            predicted_label, pred_confidence = self.predict(landmarks)
            
            # Add to stabilizer
            pred_index = self.words.index(predicted_label) if predicted_label in self.words else 0
            self.stabilizer.add_prediction(pred_index, pred_confidence)
            
            # Get stable prediction
            stable_pred, stable_conf = self.stabilizer.get_stable_prediction()
            if stable_pred is not None:
                prediction_text = self.words[stable_pred]
                confidence = stable_conf
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
        else:
            # Reset stabilizer if no hand detected for several frames
            if self.frame_count % 30 == 0:  # Reset every 30 frames
                self.stabilizer.reset()
        
        # Add prediction text to frame
        if prediction_text:
            # Main prediction
            cv2.putText(frame, f"Sign: {prediction_text}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Confidence
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Add statistics
        detection_rate = (self.detection_count / self.frame_count) * 100 if self.frame_count > 0 else 0
        cv2.putText(frame, f"Detection Rate: {detection_rate:.1f}%", (50, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame, prediction_text, confidence
    
    def run_realtime_detection(self, camera_index: int = 0):
        """Run real-time ISL detection"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera {camera_index}")
            return
        
        logger.info("Starting real-time ISL detection. Press 'q' to quit, 'r' to reset stabilizer")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
                
                # Process frame
                annotated_frame, prediction, confidence = self.process_frame(frame)
                
                # Display
                cv2.imshow('Improved ISL Detector', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.stabilizer.reset()
                    logger.info("Stabilizer reset")
                
        except KeyboardInterrupt:
            logger.info("Detection stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            logger.info(f"Final statistics:")
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Frames with hand detected: {self.detection_count}")
            logger.info(f"Detection rate: {(self.detection_count/self.frame_count)*100:.1f}%")

# Example usage
if __name__ == "__main__":
    # For real-time detection (update model path as needed)
    try:
        detector = ImprovedISLDetector(
            model_path='best_improved_model.pth',  # Update this path
            config_path='training_config.json'  # Optional config file
        )
        detector.run_realtime_detection()
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        print("Make sure your model file path is correct and the model was trained with the new format!")