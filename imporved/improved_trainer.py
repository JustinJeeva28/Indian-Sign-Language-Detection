"""
Improved PyTorch ISL Classifier
A comprehensive Indian Sign Language classifier using PyTorch with enhanced preprocessing and training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
import logging
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ISLDataset(Dataset):
    """Enhanced Dataset for ISL classification with data augmentation"""
    
    def __init__(self, keypoints: np.ndarray, labels: np.ndarray, augment=False):
        self.keypoints = torch.FloatTensor(keypoints)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
    
    def __len__(self):
        return len(self.keypoints)
    
    def __getitem__(self, idx):
        keypoints = self.keypoints[idx]
        label = self.labels[idx]
        
        # Apply augmentation during training
        if self.augment:
            keypoints = self.augment_keypoints(keypoints)
        
        return keypoints, label
    
    def augment_keypoints(self, keypoints):
        """Apply random augmentation to keypoints"""
        # Add small random noise
        noise = torch.randn_like(keypoints) * 0.01
        keypoints = keypoints + noise
        
        # Random scaling
        scale = torch.rand(1) * 0.1 + 0.95  # Scale between 0.95 and 1.05
        keypoints = keypoints * scale
        
        return keypoints

class ImprovedISLClassifier(nn.Module):
    """Enhanced ISL Classifier with better architecture and regularization"""
    
    def __init__(self, input_size: int = 126, hidden_sizes: List[int] = [256, 128, 64], 
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
            current_dropout = dropout_rate * (0.8 ** i)  # Reduce dropout in deeper layers
            
            layers.extend([
                nn.Linear(hidden_sizes[i-1], hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(current_dropout)
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()

class ImprovedISLTrainer:
    """Enhanced trainer class with comprehensive features"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize components
        self.model = None
        self.label_encoder = LabelEncoder()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Cross-validation results
        self.cv_scores = []
        
    def load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'input_size': 126,
            'hidden_sizes': [256, 128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 150,
            'patience': 15,
            'weight_decay': 1e-4,
            'use_scheduler': True,
            'scheduler_patience': 7,
            'scheduler_factor': 0.5,
            'min_lr': 1e-6,
            'use_class_weights': False,
            'augment_training': True,
            'cross_validation_folds': 5,
            'test_size': 0.2,
            'val_size': 0.15
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using defaults.")
        
        return default_config
    
    def save_config(self, config_path: str):
        """Save current configuration"""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        logger.info(f"Configuration saved to {config_path}")
    
    def load_data_from_improved_csv(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from improved CSV format"""
        try:
            df = pd.read_csv(csv_path)
            
            # First column is label, rest are features
            labels = df.iloc[:, 0].astype(str).values
            keypoints = df.iloc[:, 1:].values.astype(np.float32)
            
            # Encode labels
            encoded_labels = self.label_encoder.fit_transform(labels)
            
            logger.info(f"Loaded {len(keypoints)} samples with {keypoints.shape[1]} features")
            logger.info(f"Classes ({len(self.label_encoder.classes_)}): {list(self.label_encoder.classes_)}")
            logger.info(f"Class distribution:")
            
            unique, counts = np.unique(encoded_labels, return_counts=True)
            for i, (class_idx, count) in enumerate(zip(unique, counts)):
                class_name = self.label_encoder.classes_[class_idx]
                logger.info(f"  {class_name}: {count} samples")
            
            return keypoints, encoded_labels
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return None, None
    
    def prepare_data(self, keypoints: np.ndarray, labels: np.ndarray) -> Tuple:
        """Prepare data with stratified splitting"""
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            keypoints, labels, 
            test_size=self.config['test_size'], 
            random_state=42, 
            stratify=labels
        )
        
        # Second split: separate train and validation
        val_size_adjusted = self.config['val_size'] / (1 - self.config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size_adjusted, 
            random_state=42, 
            stratify=y_temp
        )
        
        # Create datasets
        train_dataset = ISLDataset(X_train, y_train, augment=self.config['augment_training'])
        val_dataset = ISLDataset(X_val, y_val, augment=False)
        test_dataset = ISLDataset(X_test, y_test, augment=False)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Data split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, (X_temp, y_temp)
    
    def calculate_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets"""
        unique, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        num_classes = len(unique)
        
        # Calculate weights (inverse frequency)
        weights = total_samples / (num_classes * counts)
        
        # Convert to tensor
        class_weights = torch.FloatTensor(weights).to(self.device)
        
        logger.info("Class weights calculated:")
        for i, (class_idx, weight) in enumerate(zip(unique, weights)):
            class_name = self.label_encoder.classes_[class_idx]
            logger.info(f"  {class_name}: {weight:.3f}")
        
        return class_weights
    
    def initialize_model(self, num_classes: int):
        """Initialize the model with current configuration"""
        self.config['num_classes'] = num_classes
        
        self.model = ImprovedISLClassifier(
            input_size=self.config['input_size'],
            hidden_sizes=self.config['hidden_sizes'],
            num_classes=num_classes,
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model initialized:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch with detailed logging"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def train(self, train_loader, val_loader, save_path: str = 'best_improved_model.pth'):
        """Main training loop with comprehensive monitoring"""
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        # Setup loss function
        if self.config['use_class_weights']:
            class_weights = self.calculate_class_weights(
                np.concatenate([y.numpy() for _, y in train_loader])
            )
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup scheduler
        if self.config['use_scheduler']:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=self.config['scheduler_factor'],
                patience=self.config['scheduler_patience'],
                min_lr=self.config['min_lr'],
            )
        
        # Setup early stopping
        early_stopping = EarlyStopping(patience=self.config['patience'])
        
        logger.info("Starting training...")
        logger.info(f"Configuration: {self.config}")
        
        best_val_accuracy = 0
        
        for epoch in range(self.config['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            if self.config['use_scheduler']:
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Logging
            logger.info(f'Epoch {epoch+1}/{self.config["epochs"]}:')
            logger.info(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            logger.info(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}')
            if self.config['use_scheduler']:
                logger.info(f'  Learning Rate: {current_lr:.6f}')
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                self.save_model(save_path, epoch, val_acc)
                logger.info(f'  → New best model saved (Val Acc: {val_acc:.2f}%)')
            
            # Early stopping
            if early_stopping(val_acc, self.model):
                logger.info(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")
        
        return save_path
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Perform k-fold cross-validation"""
        logger.info(f"Starting {self.config['cross_validation_folds']}-fold cross-validation...")
        
        kfold = StratifiedKFold(
            n_splits=self.config['cross_validation_folds'], 
            shuffle=True, 
            random_state=42
        )
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            logger.info(f"Training fold {fold + 1}...")
            
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Create datasets
            train_dataset = ISLDataset(X_train_fold, y_train_fold, augment=self.config['augment_training'])
            val_dataset = ISLDataset(X_val_fold, y_val_fold, augment=False)
            
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
            
            # Initialize new model for this fold
            self.initialize_model(len(self.label_encoder.classes_))
            
            # Train
            model_path = f'cv_model_fold_{fold+1}.pth'
            self.train(train_loader, val_loader, model_path)
            
            # Evaluate
            self.load_model(model_path)
            _, val_acc, _ = self.validate_epoch(val_loader, nn.CrossEntropyLoss())
            cv_scores.append(val_acc)
            
            logger.info(f"Fold {fold + 1} validation accuracy: {val_acc:.2f}%")
            
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)
        
        self.cv_scores = cv_scores
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        logger.info(f"Cross-validation completed!")
        logger.info(f"CV Scores: {[f'{score:.2f}%' for score in cv_scores]}")
        logger.info(f"Mean CV Score: {mean_cv_score:.2f}% ± {std_cv_score:.2f}%")
        
        return cv_scores
    
    def evaluate(self, test_loader, model_path: Optional[str] = None):
        """Comprehensive model evaluation"""
        if model_path:
            self.load_model(model_path)
        
        if self.model is None:
            raise ValueError("No model to evaluate!")
        
        logger.info("Starting model evaluation...")
        
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        logger.info(f"Test Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Weighted F1-Score: {f1:.4f}")
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(all_targets, all_preds, target_names=class_names)
        logger.info(f"Classification Report:\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        self.plot_confusion_matrix(cm, class_names)
        
        return accuracy, f1, all_preds, all_targets, all_probs
    
    def plot_training_history(self):
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Training Loss', alpha=0.8)
        axes[0, 0].plot(self.val_losses, label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(self.train_accuracies, label='Training Accuracy', alpha=0.8)
        axes[0, 1].plot(self.val_accuracies, label='Validation Accuracy', alpha=0.8)
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        if self.learning_rates:
            axes[1, 0].plot(self.learning_rates, alpha=0.8, color='orange')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Cross-validation scores
        if self.cv_scores:
            axes[1, 1].bar(range(1, len(self.cv_scores) + 1), self.cv_scores, alpha=0.7)
            axes[1, 1].axhline(y=np.mean(self.cv_scores), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(self.cv_scores):.2f}%')
            axes[1, 1].set_title('Cross-Validation Scores')
            axes[1, 1].set_xlabel('Fold')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_improved.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Training history plots saved as 'training_history_improved.png'")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """Plot confusion matrix with improved visualization"""
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Count'})
        
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Confusion matrix saved as 'confusion_matrix_improved.png'")
    
    def save_model(self, filepath: str, epoch: Optional[int] = None, accuracy: Optional[float] = None):
        """Save the trained model with metadata"""
        if self.model is None:
            logger.error("No model to save!")
            return
        
        # Update config with actual class mapping
        self.config['words'] = list(self.label_encoder.classes_)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'cv_scores': self.cv_scores
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if accuracy is not None:
            checkpoint['accuracy'] = accuracy
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # Load configuration and label encoder
            if 'config' in checkpoint:
                self.config.update(checkpoint['config'])
            if 'label_encoder' in checkpoint:
                self.label_encoder = checkpoint['label_encoder']
            
            # Initialize and load model
            num_classes = len(self.label_encoder.classes_)
            self.initialize_model(num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load training history if available
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
            if 'val_losses' in checkpoint:
                self.val_losses = checkpoint['val_losses']
            if 'train_accuracies' in checkpoint:
                self.train_accuracies = checkpoint['train_accuracies']
            if 'val_accuracies' in checkpoint:
                self.val_accuracies = checkpoint['val_accuracies']
            if 'cv_scores' in checkpoint:
                self.cv_scores = checkpoint['cv_scores']
            
            logger.info(f"Model loaded successfully from {filepath}")
            if 'accuracy' in checkpoint:
                logger.info(f"Model accuracy: {checkpoint['accuracy']:.2f}%")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

def main():
    """Main execution function with comprehensive workflow"""
    
    # Configuration
    CSV_PATH = 'improved_keypoint.csv'  # Path to your processed dataset
    MODEL_PATH = 'best_improved_model.pth'
    CONFIG_PATH = 'training_config.json'
    
    # Check if dataset exists
    if not os.path.exists(CSV_PATH):
        logger.error(f"Dataset file '{CSV_PATH}' not found!")
        logger.info("Please run the improved_dataset_processor.py first to generate the dataset.")
        return
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = ImprovedISLTrainer()
    
    # Load data
    logger.info("Loading dataset...")
    keypoints, labels = trainer.load_data_from_improved_csv(CSV_PATH)
    
    if keypoints is None:
        logger.error("Failed to load dataset!")
        return
    
    # Update input size based on actual data
    trainer.config['input_size'] = keypoints.shape[1]
    logger.info(f"Updated input size to {trainer.config['input_size']} features")
    
    # Save configuration after all updates
    trainer.save_config(CONFIG_PATH)
    
    # Prepare data
    train_loader, val_loader, test_loader, (X_temp, y_temp) = trainer.prepare_data(keypoints, labels)
    
    # Initialize model
    num_classes = len(trainer.label_encoder.classes_)
    trainer.initialize_model(num_classes)
    
    # Perform cross-validation (optional but recommended for robust evaluation)
    logger.info("Starting cross-validation...")
    trainer.cross_validate(X_temp, y_temp)
    
    # Train the model on the full training and validation sets
    logger.info("Starting final model training...")
    final_model_save_path = trainer.train(train_loader, val_loader, MODEL_PATH)
    
    # Evaluate the best model on the test set
    logger.info("Evaluating the best model on the test set...")
    trainer.evaluate(test_loader, model_path=final_model_save_path)
    
    # Plot training history
    logger.info("Generating training history plots...")
    trainer.plot_training_history()
    
    # Save final configuration after all processing
    logger.info("Saving final configuration...")
    trainer.save_config(CONFIG_PATH)
    
    logger.info("Process completed successfully!")

if __name__ == '__main__':
    main()