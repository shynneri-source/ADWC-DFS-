"""
Stage 2: Stratified Cascade Training
Trains three specialist models on different data distributions:
- Easy model: Trained on easy + medium samples
- Medium model: Trained on all samples with different weights
- Hard model: Trained on hard + medium samples
"""

import numpy as np
from lightgbm import LGBMClassifier
from tqdm import tqdm


class CascadeTrainer:
    """
    Trains cascade of specialist models for different difficulty levels
    """
    
    def __init__(self, cascade_params, scale_pos_weights, sample_weights_config, verbose=1):
        """
        Initialize CascadeTrainer
        
        Args:
            cascade_params: Parameters for LightGBM models
            scale_pos_weights: Dictionary with scale_pos_weight for each specialist
            sample_weights_config: Dictionary with sample weight ratios
            verbose: Verbosity level
        """
        self.cascade_params = cascade_params
        self.scale_pos_weights = scale_pos_weights
        self.sample_weights_config = sample_weights_config
        self.verbose = verbose
        
        self.model_easy_ = None
        self.model_medium_ = None
        self.model_hard_ = None
    
    def train_easy_specialist(self, X, y, masks):
        """
        Train specialist for easy samples
        Trained on: Easy + Medium (weighted)
        
        Args:
            X: Feature matrix
            y: Labels
            masks: Dictionary with stratification masks
            
        Returns:
            Trained model
        """
        easy_mask = masks['easy']
        medium_mask = masks['medium']
        
        # Combine easy and medium samples
        combined_mask = easy_mask | medium_mask
        X_train = X[combined_mask]
        y_train = y[combined_mask]
        
        # Create sample weights
        weights = np.ones(len(y_train))
        n_easy = np.sum(easy_mask)
        weights[n_easy:] = self.sample_weights_config['easy_medium_weight']
        
        if self.verbose:
            print(f"\nTraining Easy Specialist:")
            print(f"  Samples: {len(y_train)} (Easy: {n_easy}, Medium: {len(y_train)-n_easy})")
            print(f"  Fraud rate: {np.mean(y_train):.4f}")
        
        # Train model
        model = LGBMClassifier(
            **self.cascade_params,
            scale_pos_weight=self.scale_pos_weights['easy']
        )
        model.fit(X_train, y_train, sample_weight=weights)
        
        return model
    
    def train_medium_specialist(self, X, y, masks):
        """
        Train specialist for medium difficulty
        Trained on: ALL samples with different weights
        
        Args:
            X: Feature matrix
            y: Labels
            masks: Dictionary with stratification masks
            
        Returns:
            Trained model
        """
        easy_mask = masks['easy']
        medium_mask = masks['medium']
        hard_mask = masks['hard']
        
        # Use all samples
        X_train = X
        y_train = y
        
        # Create sample weights
        weights = np.zeros(len(y_train))
        weights[easy_mask] = self.sample_weights_config['medium_easy_weight']
        weights[medium_mask] = 1.0
        weights[hard_mask] = self.sample_weights_config['medium_hard_weight']
        
        if self.verbose:
            print(f"\nTraining Medium Specialist:")
            print(f"  Samples: {len(y_train)} (All data with varied weights)")
            print(f"  Fraud rate: {np.mean(y_train):.4f}")
        
        # Train model
        model = LGBMClassifier(
            **self.cascade_params,
            scale_pos_weight=self.scale_pos_weights['medium']
        )
        model.fit(X_train, y_train, sample_weight=weights)
        
        return model
    
    def train_hard_specialist(self, X, y, masks):
        """
        Train specialist for hard samples
        Trained on: Hard + Medium (weighted)
        
        Args:
            X: Feature matrix
            y: Labels
            masks: Dictionary with stratification masks
            
        Returns:
            Trained model
        """
        medium_mask = masks['medium']
        hard_mask = masks['hard']
        
        # Combine hard and medium samples
        combined_mask = hard_mask | medium_mask
        X_train = X[combined_mask]
        y_train = y[combined_mask]
        
        # Create sample weights (hard first, then medium)
        weights = np.ones(len(y_train))
        # Hard samples come first in combined data
        n_hard = np.sum(hard_mask)
        n_medium = np.sum(medium_mask)
        
        # Reorder: hard samples first
        hard_indices = np.where(hard_mask)[0]
        medium_indices = np.where(medium_mask)[0]
        reorder_indices = np.concatenate([hard_indices, medium_indices])
        
        X_train = X[reorder_indices]
        y_train = y[reorder_indices]
        
        weights[:n_hard] = 1.0
        weights[n_hard:] = self.sample_weights_config['hard_medium_weight']
        
        if self.verbose:
            print(f"\nTraining Hard Specialist:")
            print(f"  Samples: {len(y_train)} (Hard: {n_hard}, Medium: {n_medium})")
            print(f"  Fraud rate: {np.mean(y_train):.4f}")
        
        # Train model
        model = LGBMClassifier(
            **self.cascade_params,
            scale_pos_weight=self.scale_pos_weights['hard']
        )
        model.fit(X_train, y_train, sample_weight=weights)
        
        return model
    
    def train_cascade(self, X, y, masks):
        """
        Train all three specialist models
        
        Args:
            X: Feature matrix
            y: Labels
            masks: Dictionary with stratification masks
            
        Returns:
            Dictionary with trained models
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"{'Stage 2: Cascade Training':^60}")
            print(f"{'='*60}")
        
        # Train specialists
        self.model_easy_ = self.train_easy_specialist(X, y, masks)
        self.model_medium_ = self.train_medium_specialist(X, y, masks)
        self.model_hard_ = self.train_hard_specialist(X, y, masks)
        
        if self.verbose:
            print(f"\n{'='*60}\n")
        
        return {
            'easy': self.model_easy_,
            'medium': self.model_medium_,
            'hard': self.model_hard_
        }
    
    def predict_cascade(self, X, models=None):
        """
        Get predictions from all cascade models
        
        Args:
            X: Feature matrix
            models: Dictionary of models (uses self.models if None)
            
        Returns:
            Dictionary with predictions from each model
        """
        if models is None:
            models = {
                'easy': self.model_easy_,
                'medium': self.model_medium_,
                'hard': self.model_hard_
            }
        
        predictions = {}
        for name, model in models.items():
            if model is not None:
                predictions[name] = model.predict_proba(X)[:, 1]
            else:
                raise ValueError(f"Model '{name}' not trained yet")
        
        return predictions
