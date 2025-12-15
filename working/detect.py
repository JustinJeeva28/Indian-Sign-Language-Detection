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
import asyncio
import websockets
import base64
import pyaudio
from pydub import AudioSegment
import io
import threading
import time
import queue

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ARCHITECTURE OPTIONS - Choose one based on your trained model
# ============================================================================

class ImprovedISLClassifier(nn.Module):
    """
    NEW ARCHITECTURE (Enhanced with Residual Connections)
    - Use this for models trained with: [512, 256, 128, 64] hidden sizes
    - Has residual connections for better learning
    - Better for 20+ classes with different orientations
    - To use: Train with improved_trainer.py (default config)
    """
    
    def __init__(self, input_size: int = 126, hidden_sizes: List[int] = [512, 256, 128, 64], 
                 num_classes: int = 56, dropout_rate: float = 0.3, use_residual: bool = True):
        super(ImprovedISLClassifier, self).__init__()
        
        self.use_residual = use_residual
        self.input_size = input_size
        
        # Feature extraction block - wider first layer for better feature learning
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)  # Less dropout in feature extraction
        )
        
        # Residual blocks for deeper learning
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            block = self._make_residual_block(
                hidden_sizes[i], 
                hidden_sizes[i+1], 
                dropout_rate * (0.85 ** i)  # Progressive dropout reduction
            )
            self.residual_blocks.append(block)
        
        # Output layer with better initialization
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_sizes[-1]),
            nn.Dropout(dropout_rate * 0.3),  # Less dropout before output
            nn.Linear(hidden_sizes[-1], num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _make_residual_block(self, in_size: int, out_size: int, dropout: float):
        """Create a residual block with skip connection if dimensions match"""
        block = nn.ModuleDict({
            'main': nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_size, out_size),
                nn.BatchNorm1d(out_size),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            ),
            'skip': nn.Linear(in_size, out_size) if in_size != out_size else nn.Identity()
        })
        return block
    
    def _init_weights(self, module):
        """Initialize network weights with better strategies"""
        if isinstance(module, nn.Linear):
            # Use He initialization for ReLU activations
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            if self.use_residual:
                residual = block['skip'](x)
                main_out = block['main'](x)
                # Add residual connection
                x = main_out + residual
            else:
                x = block['main'](x)
        
        # Classification
        x = self.classifier(x)
        return x


class ImprovedISLClassifierOld(nn.Module):
    """
    OLD ARCHITECTURE (Simple Feedforward)
    - Use this for models trained with: [256, 128, 64] hidden sizes
    - Simple feedforward network without residual connections
    - Good for basic classification tasks
    - To use: Train with old config (hidden_sizes: [256, 128, 64])
    """
    
    def __init__(self, input_size: int = 126, hidden_sizes: List[int] = [256, 128, 64], 
                 num_classes: int = 56, dropout_rate: float = 0.3, use_residual: bool = False):
        super(ImprovedISLClassifierOld, self).__init__()
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
        """Initialize network weights"""
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
        
        # KNN-like voting buffer
        self.prediction_buffer = []
        self.buffer_size = 20  # Number of frames to collect before voting
        self.last_final_prediction = "Waiting..."
        self.last_confidence = 0.0
        # ElevenLabs TTS integration
        self.speaker = ElevenLabsSpeaker(
            api_key="sk_989f85658385d98264758b44568d4cfc670f1690d4f66570",
            voice_id="Xb7hH8MSUJpSbSDYk0k2",
            model_id="eleven_flash_v2_5"
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
            'input_size': 126,  # 2 hands * 63 features (3D)
            'hidden_sizes': [512, 256, 128, 64],  # Updated architecture
            'dropout_rate': 0.3,
            'words': [
                "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "A", "afraid", "agree", "assistance", "B", "bad", "become", "C", "college", "D", "doctor",
                "E", "F", "from", "G", "H", "I", "J", "K", "L", "M", "N", "none",
                "O", "P", "pain", "pray", "Q", "R", "S", "secondary", "skin", "small", "specific", "stand",
                "T", "today", "U", "V", "W", "warn", "which", "work", "X", "Y", "you", "Z"
            ]
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
    
    def load_model(self, model_path: str) -> nn.Module:
        """
        Load trained model - automatically detects architecture based on hidden_sizes
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract configuration from checkpoint if available
            if 'config' in checkpoint:
                checkpoint_config = checkpoint['config']
                # Update config with values from checkpoint
                if 'input_size' in checkpoint_config:
                    self.config['input_size'] = checkpoint_config['input_size']
                if 'hidden_sizes' in checkpoint_config:
                    self.config['hidden_sizes'] = checkpoint_config['hidden_sizes']
                if 'dropout_rate' in checkpoint_config:
                    self.config['dropout_rate'] = checkpoint_config['dropout_rate']
                if 'words' in checkpoint_config:
                    self.words = checkpoint_config['words']
                    self.config['words'] = checkpoint_config['words']
            
            # Extract words from checkpoint (could be in 'words' key or 'config' key)
            if 'words' in checkpoint:
                self.words = checkpoint['words']
                self.config['words'] = checkpoint['words']
            elif 'label_encoder' in checkpoint:
                # If label encoder is saved, extract classes from it
                try:
                    from sklearn.preprocessing import LabelEncoder
                    label_encoder = checkpoint['label_encoder']
                    self.words = list(label_encoder.classes_)
                    self.config['words'] = self.words
                except:
                    pass
            
            # Determine number of classes
            num_classes = len(self.words) if self.words else len(self.config['words'])
            
            # Detect architecture based on hidden_sizes
            hidden_sizes = self.config['hidden_sizes']
            use_new_architecture = len(hidden_sizes) == 4 and hidden_sizes[0] == 512
            
            if use_new_architecture:
                # NEW ARCHITECTURE: [512, 256, 128, 64] with residuals
                logger.info("Using NEW architecture (with residual connections)")
                model = ImprovedISLClassifier(
                    input_size=self.config['input_size'],
                    hidden_sizes=self.config['hidden_sizes'],
                    num_classes=num_classes,
                    dropout_rate=self.config['dropout_rate'],
                    use_residual=True
                ).to(self.device)
            else:
                # OLD ARCHITECTURE: [256, 128, 64] simple feedforward
                logger.info("Using OLD architecture (simple feedforward)")
                model = ImprovedISLClassifierOld(
                    input_size=self.config['input_size'],
                    hidden_sizes=self.config['hidden_sizes'],
                    num_classes=num_classes,
                    dropout_rate=self.config['dropout_rate'],
                    use_residual=False
                ).to(self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Number of classes: {num_classes}")
            logger.info(f"Words: {self.words[:10]}..." if len(self.words) > 10 else f"Words: {self.words}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
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
            
            # Make prediction for the current frame
            predicted_label, pred_confidence = self.predict(landmarks)
            
            # Add the prediction to our buffer
            if pred_confidence > 0.5: # Only add confident predictions
                self.prediction_buffer.append(predicted_label)

            # If the buffer is full, determine the most common sign and display it
            if len(self.prediction_buffer) >= self.buffer_size:
                if self.prediction_buffer:
                    final_prediction = max(set(self.prediction_buffer), key=self.prediction_buffer.count)
                    self.last_final_prediction = final_prediction
                    self.last_confidence = 1.0
                    print(f"Final Detected Sign: {self.last_final_prediction}")
                    # Speak only if new
                    self.speaker.speak(str(self.last_final_prediction))

                # Clear the buffer to start collecting for the next sign
                self.prediction_buffer.clear()

            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
        else:
            # If no hand is detected, you might want to clear the buffer
            # so it doesn't carry over old predictions.
            if len(self.prediction_buffer) > 0:
                 self.prediction_buffer.clear()
                 print("Buffer cleared due to no hand detection.")
        
        # FPS calculation
        if not hasattr(self, 'prev_time'):
            self.prev_time = time.time()
            self.fps = 0.0
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0.0
        self.prev_time = current_time

        # Add the final prediction text to the frame
        cv2.putText(frame, f"Sign: {self.last_final_prediction}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        # Show FPS on the camera window
        cv2.putText(frame, f"FPS: {self.fps:.2f}", (50, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # Add statistics
        detection_rate = (self.detection_count / self.frame_count) * 100 if self.frame_count > 0 else 0
        cv2.putText(frame, f"Detection Rate: {detection_rate:.1f}%", (50, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame, self.last_final_prediction, self.last_confidence
    
    def run_realtime_detection(self, camera_source = 0):
        """
        Run real-time ISL detection
        
        Args:
            camera_source: Can be:
                - int: Camera index (0, 1, 2, etc.) for local webcam
                - str: IP camera URL (e.g., "https://10.153.9.148:8080/video")
        """
        # Check if camera_source is a URL (string) or index (int)
        if isinstance(camera_source, str):
            cap = cv2.VideoCapture(camera_source)
            logger.info(f"Connecting to IP camera: {camera_source}")
        else:
            cap = cv2.VideoCapture(camera_source)
            logger.info(f"Using local camera index: {camera_source}")
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera source: {camera_source}")
            return
        
        logger.info("Starting real-time ISL detection. Press 'q' to quit, 'r' to reset prediction buffer")
        
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
                    self.prediction_buffer.clear()
                    self.last_final_prediction = "Waiting..."
                    logger.info("Prediction buffer reset")
                
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

class ElevenLabsSpeaker:
    def __init__(self, api_key, voice_id, model_id):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id={self.model_id}"
        self.player = LiveAudioPlayer()
        self.last_spoken = None
        self.lock = threading.Lock()
        self.loop = asyncio.get_event_loop()
        self.ws_task = None
        self.queue = asyncio.Queue()
        self.running = True
        threading.Thread(target=self._start_loop, daemon=True).start()

    def _start_loop(self):
        self.loop.run_until_complete(self._run())

    async def _run(self):
        async with websockets.connect(self.uri) as websocket:
            await websocket.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "use_speaker_boost": False},
                "generation_config": {"chunk_length_schedule": [50, 120, 160, 290]},
                "xi_api_key": self.api_key,
            }))
            listen_task = asyncio.create_task(self._listen(websocket))
            last_sent = time.time()
            while self.running:
                try:
                    text = await asyncio.wait_for(self.queue.get(), timeout=17)
                    if text:
                        await websocket.send(json.dumps({"text": text, "flush": True}))
                    last_sent = time.time()
                except asyncio.TimeoutError:
                    # Send keep-alive space if no new text for 17 seconds
                    await websocket.send(json.dumps({"text": " "}))
            await websocket.send(json.dumps({"text": ""}))
            await listen_task
        self.player.close()

    async def _listen(self, websocket):
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                if data.get("audio"):
                    chunk = base64.b64decode(data["audio"])
                    self.player.play_mp3_chunk(chunk)
                elif data.get('isFinal'):
                    break
            except websockets.exceptions.ConnectionClosed:
                break

    def speak(self, text):
        with self.lock:
            if text and text != self.last_spoken:
                self.last_spoken = text
                self.loop.call_soon_threadsafe(self.queue.put_nowait, text)

class LiveAudioPlayer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.running = True
        self.playback_thread.start()

    def play_mp3_chunk(self, chunk):
        if chunk:
            # Clear any currently queued audio to play the new sign immediately
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            self.audio_queue.put(chunk)

    def _playback_worker(self):
        while self.running:
            chunk = self.audio_queue.get()
            if chunk is None:
                break
            audio = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
            if self.stream is None:
                self.stream = self.p.open(format=self.p.get_format_from_width(audio.sample_width),
                                          channels=audio.channels,
                                          rate=audio.frame_rate,
                                          output=True)
            self.stream.write(audio.raw_data)
            self.audio_queue.task_done()

    def close(self):
        self.running = False
        self.audio_queue.put(None)
        self.playback_thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

# Example usage
if __name__ == "__main__":
    """
    ============================================================================
    USAGE INSTRUCTIONS - Two Commands Available
    ============================================================================
    
    COMMAND 1 - NEW MODEL (56 classes: numbers, letters, none + 20 words):
        python detect.py new
        - Uses: cv_model_fold_2.pth (or latest CV model)
        - Architecture: [512, 256, 128, 64] with residual connections
        - Classes: 1-9, A-Z, none, + 20 words (afraid, agree, etc.)
    
    COMMAND 2 - OLD MODEL (36 classes: numbers, letters, none only):
        python detect.py old
        - Uses: best_improved_model.pth
        - Architecture: [256, 128, 64] simple feedforward
        - Classes: 1-9, A-Z, none (no words)
    ============================================================================
    """
    import sys
    import glob
    
    # Determine which model to use
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
        
        if model_type == 'new':
            # NEW MODEL: 56 classes with words
            cv_models = glob.glob('cv_model_fold_*.pth')
            if cv_models:
                model_path = max(cv_models, key=os.path.getctime)
                logger.info(f"Using NEW model (56 classes): {model_path}")
            else:
                logger.error("No CV model found! Please train with improved_trainer.py first.")
                sys.exit(1)
                
        elif model_type == 'old':
            # OLD MODEL: 36 classes (numbers, letters, none only)
            model_path = 'best_improved_model.pth'
            if not os.path.exists(model_path):
                logger.error(f"OLD model not found: {model_path}")
                sys.exit(1)
            logger.info(f"Using OLD model (36 classes): {model_path}")
            
        else:
            # Custom model path
            model_path = sys.argv[1]
            logger.info(f"Using custom model: {model_path}")
    else:
        # Default: try NEW model first, fallback to OLD
        cv_models = glob.glob('cv_model_fold_*.pth')
        if cv_models:
            model_path = max(cv_models, key=os.path.getctime)
            logger.info(f"Auto-selected NEW model (56 classes): {model_path}")
        elif os.path.exists('best_improved_model.pth'):
            model_path = 'best_improved_model.pth'
            logger.info(f"Auto-selected OLD model (36 classes): {model_path}")
        else:
            logger.error("No model found! Use 'python detect.py new' or 'python detect.py old'")
            sys.exit(1)
    
    # Parse camera source (default: local webcam index 0)
    camera_source = 0
    if len(sys.argv) > 2:
        # Check if second argument is a URL or camera index
        camera_arg = sys.argv[2]
        if camera_arg.startswith('http://') or camera_arg.startswith('https://'):
            camera_source = camera_arg
            logger.info(f"Using IP camera: {camera_source}")
        else:
            try:
                camera_source = int(camera_arg)
                logger.info(f"Using local camera index: {camera_source}")
            except ValueError:
                logger.warning(f"Invalid camera source: {camera_arg}, using default (0)")
    
    try:
        detector = ImprovedISLDetector(
            model_path=model_path,
            config_path='training_config.json'  # Optional config file
        )
        detector.run_realtime_detection(camera_source=camera_source)
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        print("Make sure your model file path is correct and the model was trained with the new format!")
        print(f"Tried to load: {model_path}")
        print("\nUsage:")
        print("  python detect.py new  - Use NEW model (56 classes with words)")
        print("  python detect.py old  - Use OLD model (36 classes: numbers, letters, none)")
        print("  python detect.py new https://10.153.9.148:8080/video  - Use IP camera with NEW model")
        print("  python detect.py old https://10.153.9.148:8080/video  - Use IP camera with OLD model")