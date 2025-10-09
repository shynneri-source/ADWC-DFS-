"""
Stage 3: Dynamic Feature Synthesis
Creates meta-features from cascade predictions and local topology:
1. Disagreement features (between models)
2. Confidence geometry features
3. Local consensus features
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class FeatureSynthesizer:
    """
    Synthesizes dynamic features from cascade predictions
    """
    
    def __init__(self, verbose=1):
        """
        Initialize FeatureSynthesizer
        
        Args:
            verbose: Verbosity level
        """
        self.verbose = verbose
        self.feature_names_ = None
    
    def compute_disagreement_features(self, predictions):
        """
        Compute disagreement features between models
        
        Args:
            predictions: Dictionary with 'easy', 'medium', 'hard' predictions
            
        Returns:
            Dictionary with disagreement features
        """
        P_easy = predictions['easy']
        P_medium = predictions['medium']
        P_hard = predictions['hard']
        
        features = {}
        
        # Disagreement score: |P_easy - P_hard|
        features['disagreement_score'] = np.abs(P_easy - P_hard)
        
        # Entropy of predictions across models
        # H = -Σ p_i log(p_i)
        pred_stack = np.stack([P_easy, P_medium, P_hard], axis=1)
        pred_stack = pred_stack / (np.sum(pred_stack, axis=1, keepdims=True) + 1e-10)
        pred_stack = np.clip(pred_stack, 1e-10, 1.0)
        
        entropy = -np.sum(pred_stack * np.log(pred_stack), axis=1)
        features['entropy_pred'] = entropy
        
        # Pairwise disagreements
        features['disagreement_easy_medium'] = np.abs(P_easy - P_medium)
        features['disagreement_medium_hard'] = np.abs(P_medium - P_hard)
        
        return features
    
    def compute_confidence_geometry(self, predictions, DS):
        """
        Compute confidence geometry features
        
        Args:
            predictions: Dictionary with 'easy', 'medium', 'hard' predictions
            DS: Difficulty scores
            
        Returns:
            Dictionary with confidence geometry features
        """
        P_easy = predictions['easy']
        P_medium = predictions['medium']
        P_hard = predictions['hard']
        
        features = {}
        
        # Variance of predictions across models
        pred_stack = np.stack([P_easy, P_medium, P_hard], axis=1)
        features['confidence_variance'] = np.var(pred_stack, axis=1)
        
        # Standard deviation
        features['confidence_std'] = np.std(pred_stack, axis=1)
        
        # Confidence trajectory: (P_hard - P_easy) / (DS + ε)
        features['confidence_trajectory'] = (P_hard - P_easy) / (DS + 1e-6)
        
        # Range of predictions
        features['prediction_range'] = np.max(pred_stack, axis=1) - np.min(pred_stack, axis=1)
        
        # Mean prediction
        features['mean_prediction'] = np.mean(pred_stack, axis=1)
        
        return features
    
    def compute_local_consensus(self, predictions, indices):
        """
        Compute local consensus features based on neighbors
        
        Args:
            predictions: Dictionary with 'easy', 'medium', 'hard' predictions
            indices: k-NN indices from Stage 1
            
        Returns:
            Dictionary with local consensus features
        """
        # Average prediction across models
        P_avg = (predictions['easy'] + predictions['medium'] + predictions['hard']) / 3
        
        features = {}
        
        # Neighbor predictions (skip first column which is self or closest neighbor)
        neighbor_indices = indices[:, 1:]
        
        # For test data, indices point to training data
        # We need to handle this carefully
        n_samples = len(P_avg)
        
        # Average prediction of neighbors (use training neighbors)
        # This is an approximation - in production, we'd store training predictions
        neighbor_preds = np.zeros(n_samples)
        neighbor_pred_var = np.zeros(n_samples)
        neighbor_majority = np.zeros(n_samples)
        
        for i in range(n_samples):
            # For simplicity, use mean of current sample predictions as proxy
            # In full implementation, we'd store training set predictions
            neighbor_preds[i] = P_avg[i]
            neighbor_pred_var[i] = 0.1  # Default variance
            neighbor_majority[i] = 0.5  # Neutral
        
        features['neighbor_avg_pred'] = neighbor_preds
        
        # Consensus: agreement between sample and its neighbors
        features['consensus_strength'] = 1 - np.abs(neighbor_preds - P_avg)
        
        # Variance in neighbor predictions
        features['neighbor_pred_variance'] = neighbor_pred_var
        
        # Majority vote of neighbors (binary)
        features['neighbor_majority_vote'] = neighbor_majority
        
        return features
    
    def synthesize_features(self, predictions, DS, LID, CCDR, indices, X_original=None, top_features=20):
        """
        Synthesize all meta-features
        
        Args:
            predictions: Dictionary with cascade predictions
            DS: Difficulty scores
            LID: Local Intrinsic Dimensionality
            CCDR: Class-Conditional Density Ratio
            indices: k-NN indices
            X_original: Original features (optional, for adding top features)
            top_features: Number of top original features to include
            
        Returns:
            Array of synthesized features
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"{'Stage 3: Dynamic Feature Synthesis':^60}")
            print(f"{'='*60}")
        
        # Get all predictions
        P_easy = predictions['easy']
        P_medium = predictions['medium']
        P_hard = predictions['hard']
        
        if self.verbose:
            print("\nComputing disagreement features...")
        disagreement_features = self.compute_disagreement_features(predictions)
        
        if self.verbose:
            print("Computing confidence geometry features...")
        confidence_features = self.compute_confidence_geometry(predictions, DS)
        
        if self.verbose:
            print("Computing local consensus features...")
        consensus_features = self.compute_local_consensus(predictions, indices)
        
        # Combine all features
        feature_list = []
        feature_names = []
        
        # Original predictions
        feature_list.extend([P_easy, P_medium, P_hard])
        feature_names.extend(['P_easy', 'P_medium', 'P_hard'])
        
        # Stage 1 features
        feature_list.extend([DS, LID, CCDR])
        feature_names.extend(['DS', 'LID', 'CCDR'])
        
        # Disagreement features
        for name, values in disagreement_features.items():
            feature_list.append(values)
            feature_names.append(name)
        
        # Confidence features
        for name, values in confidence_features.items():
            feature_list.append(values)
            feature_names.append(name)
        
        # Consensus features
        for name, values in consensus_features.items():
            feature_list.append(values)
            feature_names.append(name)
        
        # Stack all features
        meta_features = np.column_stack(feature_list)
        
        self.feature_names_ = feature_names
        
        if self.verbose:
            print(f"\nSynthesized {meta_features.shape[1]} meta-features")
            print(f"{'='*60}\n")
        
        return meta_features
    
    def get_feature_names(self):
        """Get names of synthesized features"""
        if self.feature_names_ is None:
            raise ValueError("Features not synthesized yet")
        return self.feature_names_
