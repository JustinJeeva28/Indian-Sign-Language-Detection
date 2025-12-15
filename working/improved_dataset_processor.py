import cv2
import mediapipe as mp
import csv
import pandas as pd
import numpy as np
import os
import logging
from typing import List, Tuple, Optional
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HandLandmarkProcessor:
    """Robust hand landmark processing with consistent preprocessing"""
    
    def __init__(self, use_3d=True, normalize_method='minmax', num_hands=2):
        self.use_3d = use_3d
        self.normalize_method = normalize_method
        self.num_hands = num_hands
        self.single_hand_size = 63 if use_3d else 42
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

class ImprovedDatasetProcessor:
    """Improved dataset processor for consistent feature extraction"""
    
    def __init__(self, use_3d=True, normalize_method='minmax', augment_data=False):
        self.use_3d = use_3d
        self.normalize_method = normalize_method
        self.augment_data = augment_data
        self.landmark_processor = HandLandmarkProcessor(use_3d=use_3d, normalize_method=normalize_method, num_hands=2)
        self.mp_hands = mp.solutions.hands
        
        # Statistics
        self.processed_count = 0
        self.failed_count = 0
        self.label_counts = {}
        
    def augment_landmarks(self, landmarks: np.ndarray, num_augmentations=2) -> List[np.ndarray]:
        """Apply data augmentation to landmarks"""
        augmented = [landmarks]  # Original
        
        if not self.augment_data:
            return augmented
        
        landmarks_reshaped = landmarks.reshape(-1, 3 if self.use_3d else 2)
        
        for _ in range(num_augmentations):
            # Add small random noise
            noise_factor = 0.02
            noise = np.random.normal(0, noise_factor, landmarks_reshaped.shape)
            noisy_landmarks = landmarks_reshaped + noise
            
            # Slight rotation (only for 2D case)
            if not self.use_3d:
                angle = np.random.uniform(-5, 5) * np.pi / 180  # ±5 degrees
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                
                # Apply rotation around center
                center = np.mean(noisy_landmarks, axis=0)
                centered = noisy_landmarks - center
                rotated = np.dot(centered, rotation_matrix.T)
                noisy_landmarks = rotated + center
            
            # Slight scaling
            scale_factor = np.random.uniform(0.95, 1.05)
            scaled_landmarks = noisy_landmarks * scale_factor
            
            # Flatten and add to augmented list
            augmented.append(scaled_landmarks.flatten().astype(np.float32))
        
        return augmented
    
    def process_single_image(self, img_path: str, label: str, hands_detector) -> List[Tuple[str, np.ndarray]]:
        """Process single image and return landmarks with label"""
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                self.failed_count += 1
                return []
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands_detector.process(image_rgb)
            
            # Extract landmarks
            landmarks, has_hand = self.landmark_processor.process_frame_landmarks(results)
            
            if not has_hand:
                self.failed_count += 1
                return []
            
            # Apply augmentation if enabled
            augmented_landmarks = self.augment_landmarks(landmarks)
            
            # Return list of (label, landmarks) tuples
            processed_samples = [(label, aug_landmarks) for aug_landmarks in augmented_landmarks]
            self.processed_count += len(processed_samples)
            
            return processed_samples
            
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
            self.failed_count += 1
            return []
    
    def process_dataset(self, dataset_dir: str, output_csv: str, 
                       valid_extensions=('.jpg', '.jpeg', '.png', '.bmp'),
                       max_samples_per_class=None):
        """Process entire dataset with improved preprocessing"""
        
        logger.info(f"Starting dataset processing...")
        logger.info(f"Dataset directory: {dataset_dir}")
        logger.info(f"Output file: {output_csv}")
        logger.info(f"Using 3D coordinates: {self.use_3d}")
        logger.info(f"Normalization method: {self.normalize_method}")
        logger.info(f"Data augmentation: {self.augment_data}")
        
        processed_data = []
        
        # Initialize MediaPipe
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5) as hands:
            
            # Get all labels (subdirectories)
            labels = sorted([d for d in os.listdir(dataset_dir) 
                           if os.path.isdir(os.path.join(dataset_dir, d))])
            
            logger.info(f"Found {len(labels)} classes: {labels}")
            
            # Process each label
            for label in labels:
                label_dir = os.path.join(dataset_dir, label)
                
                # Get all image files
                image_files = [f for f in os.listdir(label_dir) 
                             if f.lower().endswith(valid_extensions)]
                
                if max_samples_per_class:
                    image_files = image_files[:max_samples_per_class]
                
                logger.info(f"Processing label '{label}': {len(image_files)} images")
                
                label_processed = 0
                label_failed = 0
                
                # Process images with progress bar
                for img_name in tqdm(image_files, desc=f"Processing {label}"):
                    img_path = os.path.join(label_dir, img_name)
                    
                    # Process single image
                    samples = self.process_single_image(img_path, label, hands)
                    
                    if samples:
                        processed_data.extend(samples)
                        label_processed += len(samples)
                    else:
                        label_failed += 1
                
                # Update statistics
                self.label_counts[label] = label_processed
                
                logger.info(f"Label '{label}' completed: {label_processed} samples processed, {label_failed} failed")
        
        # Save to CSV
        if processed_data:
            logger.info(f"Saving {len(processed_data)} samples to {output_csv}")
            
            # Create DataFrame
            feature_columns = [f'feature_{i}' for i in range(len(processed_data[0][1]))]
            columns = ['label'] + feature_columns
            
            # Prepare data for DataFrame
            df_data = []
            for label, landmarks in processed_data:
                row = [label] + landmarks.tolist()
                df_data.append(row)
            
            # Create and save DataFrame
            df = pd.DataFrame(df_data, columns=columns)
            df.to_csv(output_csv, index=False)
            
            # Print final statistics
            logger.info("Dataset processing completed!")
            logger.info(f"Total samples processed: {len(processed_data)}")
            logger.info(f"Total images failed: {self.failed_count}")
            logger.info(f"Success rate: {(len(processed_data)/(len(processed_data)+self.failed_count))*100:.1f}%")
            
            logger.info("Samples per class:")
            for label, count in self.label_counts.items():
                logger.info(f"  {label}: {count}")
            
            # Save statistics
            stats_file = output_csv.replace('.csv', '_stats.txt')
            with open(stats_file, 'w') as f:
                f.write(f"Dataset Processing Statistics\n")
                f.write(f"============================\n\n")
                f.write(f"Total samples: {len(processed_data)}\n")
                f.write(f"Failed images: {self.failed_count}\n")
                f.write(f"Success rate: {(len(processed_data)/(len(processed_data)+self.failed_count))*100:.1f}%\n")
                f.write(f"Features per sample: {len(processed_data[0][1])}\n")
                f.write(f"Using 3D coordinates: {self.use_3d}\n")
                f.write(f"Normalization: {self.normalize_method}\n")
                f.write(f"Data augmentation: {self.augment_data}\n\n")
                f.write("Samples per class:\n")
                for label, count in self.label_counts.items():
                    f.write(f"  {label}: {count}\n")
            
            logger.info(f"Statistics saved to {stats_file}")
            
        else:
            logger.error("No data was processed successfully!")
    
    def validate_dataset(self, csv_path: str):
        """Validate the processed dataset"""
        try:
            df = pd.read_csv(csv_path)
            
            logger.info("Dataset validation:")
            logger.info(f"Total samples: {len(df)}")
            logger.info(f"Features per sample: {len(df.columns) - 1}")
            logger.info(f"Classes: {sorted(df['label'].astype(str).unique())}")
            logger.info("Class distribution:")
            
            for label, count in df['label'].value_counts().sort_index().items():
                logger.info(f"  {label}: {count}")
            
            # Check for any NaN values
            nan_count = df.isnull().sum().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in dataset!")
            else:
                logger.info("No NaN values found - dataset is clean!")
                
        except Exception as e:
            logger.error(f"Error validating dataset: {e}")

def main():
    """Main function to process dataset"""
    
    # Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from working/ to project root, then into data/
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    DATASET_DIR = os.path.join(PROJECT_ROOT, 'data')  # Path to reorganized dataset directory (parent folder)
    DATASET_DIR = os.path.abspath(DATASET_DIR)  # Get absolute path
    OUTPUT_CSV = 'improved_keypoint.csv'
    USE_3D = True  # Set to False if you want 2D coordinates only
    # NOTE: Now using both hands, so feature size is 126 (3D) or 84 (2D)
    MAX_SAMPLES_PER_CLASS = None  # Set to limit samples per class (None for all)
    AUGMENT_DATA = True  # Set to True to enable data augmentation
    
    # Check if dataset directory exists
    logger.info(f"Looking for dataset at: {DATASET_DIR}")
    if not os.path.exists(DATASET_DIR):
        logger.error(f"Dataset directory '{DATASET_DIR}' not found!")
        logger.info("Please create a dataset directory with the following structure:")
        logger.info("data/")
        logger.info("  ├── A/")
        logger.info("  │   ├── image1.jpg")
        logger.info("  │   ├── image2.jpg")
        logger.info("  │   └── ...")
        logger.info("  ├── B/")
        logger.info("  │   ├── image1.jpg")
        logger.info("  │   └── ...")
        logger.info("  └── ...")
        return
    
    # Initialize processor
    processor = ImprovedDatasetProcessor(
        use_3d=USE_3D,
        normalize_method='minmax',
        augment_data=AUGMENT_DATA
    )
    
    # Process dataset
    processor.process_dataset(
        dataset_dir=DATASET_DIR,
        output_csv=OUTPUT_CSV,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS
    )
    
    # Validate the processed dataset
    if os.path.exists(OUTPUT_CSV):
        processor.validate_dataset(OUTPUT_CSV)
    
    logger.info("Dataset processing completed!")

if __name__ == "__main__":
    main()