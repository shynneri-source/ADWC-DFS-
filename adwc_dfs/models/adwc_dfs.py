"""
Main ADWC-DFS Model
Integrates all four stages into a complete pipeline
"""

import numpy as np
import pickle
import time
from ..config import ADWCDFSConfig
from ..stages import (
    DensityProfiler,
    CascadeTrainer,
    FeatureSynthesizer,
    MetaClassifier
)


class ADWCDFS:
    """
    Adaptive Density-Weighted Cascade with Dynamic Feature Synthesis
    
    A meta-learning framework for imbalanced classification that combines:
    1. Local density estimation for difficulty profiling
    2. Cascade of specialist models for different difficulty levels
    3. Dynamic feature synthesis from prediction disagreement
    4. Adaptive meta-classifier with uncertainty-weighted learning
    """
    
    def __init__(self, config=None, verbose=1):
        """
        Initialize ADWC-DFS model
        
        Args:
            config: Configuration object (uses default if None)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.config = config if config is not None else ADWCDFSConfig()
        self.verbose = verbose
        
        # Initialize stages
        self.stage1_profiler = None
        self.stage2_cascade = None
        self.stage3_synthesizer = None
        self.stage4_meta = None
        
        # Stored data from training
        self.density_features_ = None
        self.cascade_models_ = None
        self.meta_features_ = None
        self.y_train_ = None  # Store training labels
        self.cascade_predictions_train_ = None  # Store training predictions
        
        # Training metadata
        self.is_fitted_ = False
        self.training_time_ = None
    
    def fit(self, X, y):
        """
        Fit ADWC-DFS model to training data
        
        Args:
            X: Training features (numpy array or pandas DataFrame)
            y: Training labels (numpy array or pandas Series)
            
        Returns:
            self
        """
        start_time = time.time()
        
        # Convert to numpy arrays
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"{'ADWC-DFS Training Pipeline':^60}")
            print(f"{'#'*60}")
            print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Class distribution: {np.sum(y==0)} legit, {np.sum(y==1)} fraud")
            print(f"Fraud rate: {np.mean(y):.4%}")
        
        # Stage 1: Local Density Profiling
        self.stage1_profiler = DensityProfiler(
            k=self.config.K_NEIGHBORS,
            alpha=self.config.ALPHA,
            beta=self.config.BETA,
            gamma=self.config.GAMMA,
            n_jobs=self.config.N_JOBS,
            verbose=self.verbose
        )
        
        self.density_features_ = self.stage1_profiler.compute_difficulty_score(X, y)
        masks = self.stage1_profiler.get_stratification_masks(
            easy_percentile=self.config.EASY_PERCENTILE,
            hard_percentile=self.config.HARD_PERCENTILE
        )
        
        # Stage 2: Cascade Training
        self.stage2_cascade = CascadeTrainer(
            cascade_params=self.config.CASCADE_PARAMS,
            scale_pos_weights={
                'easy': self.config.SCALE_POS_WEIGHT_EASY,
                'medium': self.config.SCALE_POS_WEIGHT_MEDIUM,
                'hard': self.config.SCALE_POS_WEIGHT_HARD
            },
            sample_weights_config={
                'easy_medium_weight': self.config.EASY_MEDIUM_WEIGHT,
                'medium_easy_weight': self.config.MEDIUM_EASY_WEIGHT,
                'medium_hard_weight': self.config.MEDIUM_HARD_WEIGHT,
                'hard_medium_weight': self.config.HARD_MEDIUM_WEIGHT
            },
            verbose=self.verbose
        )
        
        self.cascade_models_ = self.stage2_cascade.train_cascade(X, y, masks)
        
        # Get cascade predictions
        cascade_predictions = self.stage2_cascade.predict_cascade(X)
        
        # Store training labels and predictions for later use
        self.y_train_ = y
        self.cascade_predictions_train_ = cascade_predictions
        
        # Stage 3: Feature Synthesis
        self.stage3_synthesizer = FeatureSynthesizer(verbose=self.verbose)
        
        X_meta = self.stage3_synthesizer.synthesize_features(
            predictions=cascade_predictions,
            DS=self.density_features_['DS'],
            LID=self.density_features_['LID'],
            CCDR=self.density_features_['CCDR'],
            indices=self.density_features_['indices'],
            X_original=X,
            top_features=self.config.TOP_FEATURES
        )
        
        self.meta_features_ = X_meta
        
        # Stage 4: Meta-Classifier
        self.stage4_meta = MetaClassifier(
            meta_params=self.config.META_PARAMS,
            alpha_base=self.config.ALPHA_BASE,
            verbose=self.verbose
        )
        
        self.stage4_meta.train(
            X_meta=X_meta,
            y=y,
            DS=self.density_features_['DS'],
            feature_names=self.stage3_synthesizer.get_feature_names()
        )
        
        # Mark as fitted
        self.is_fitted_ = True
        self.training_time_ = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"Training completed in {self.training_time_:.2f} seconds")
            print(f"{'#'*60}\n")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Features (numpy array or pandas DataFrame)
            
        Returns:
            Array of probabilities for fraud class
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Convert to numpy array
        if hasattr(X, 'values'):
            X = X.values
        
        # Stage 1: Compute density features for test data
        # Use training data's k-NN model to find neighbors in training set
        distances, indices = self.stage1_profiler.knn_model_.kneighbors(X)
        
        # Compute LID
        LID = self.stage1_profiler.compute_lid(distances)
        
        # For CCDR, use training labels from neighbors
        neighbor_indices = indices[:, :]  # All neighbors are from training
        neighbor_labels = self.y_train_[neighbor_indices]
        
        # Compute density of each class in neighborhood
        fraud_density = np.mean(neighbor_labels == 1, axis=1)
        legit_density = np.mean(neighbor_labels == 0, axis=1)
        
        # Compute CCDR
        epsilon = 1e-6
        CCDR = np.log(fraud_density + epsilon) - np.log(legit_density + epsilon)
        
        # Compute max similarity - simplified version
        from sklearn.metrics.pairwise import cosine_similarity
        max_sim = np.zeros(len(X))
        
        # Get training data features from kNN model
        X_train = self.stage1_profiler.knn_model_._fit_X
        
        # For each test sample, compute similarity to its nearest training neighbor
        batch_size = 1000
        for i in range(0, len(X), batch_size):
            end_idx = min(i + batch_size, len(X))
            batch_X = X[i:end_idx]
            batch_indices = indices[i:end_idx, 0]  # Closest training sample
            
            # Get nearest neighbors
            nearest_neighbors = X_train[batch_indices]
            
            # Compute cosine similarity
            similarities = np.sum(batch_X * nearest_neighbors, axis=1) / (
                np.linalg.norm(batch_X, axis=1) * np.linalg.norm(nearest_neighbors, axis=1) + 1e-10
            )
            max_sim[i:end_idx] = similarities
        
        # Compute DS
        DS = (
            self.config.ALPHA * np.abs(CCDR) + 
            self.config.BETA * LID + 
            self.config.GAMMA * (1 - max_sim)
        )
        
        # Stage 2: Get cascade predictions
        cascade_predictions = self.stage2_cascade.predict_cascade(X)
        
        # Stage 3: Synthesize features
        # Pass training predictions for neighbor consensus
        X_meta = self.stage3_synthesizer.synthesize_features(
            predictions=cascade_predictions,
            DS=DS,
            LID=LID,
            CCDR=CCDR,
            indices=indices,
            X_original=X,
            top_features=self.config.TOP_FEATURES
        )
        
        # Stage 4: Meta-classifier prediction
        y_pred_proba = self.stage4_meta.predict_proba(X_meta)
        
        return y_pred_proba
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels
        
        Args:
            X: Features (numpy array or pandas DataFrame)
            threshold: Classification threshold
            
        Returns:
            Array of predicted labels
        """
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba >= threshold).astype(int)
    
    def get_feature_importance(self):
        """Get feature importance from meta-classifier"""
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.stage4_meta.get_feature_importance()
    
    def get_cascade_predictions(self, X):
        """Get predictions from individual cascade models"""
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.stage2_cascade.predict_cascade(X)
    
    def save(self, filepath):
        """
        Save model to file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        if self.verbose:
            print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """
        Load model from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded ADWCDFS model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        return model
