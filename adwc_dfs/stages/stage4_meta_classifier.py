"""
Stage 4: Adaptive Meta-Classifier
Trains a lightweight meta-classifier with uncertainty-weighted focal loss
"""

import numpy as np
from lightgbm import LGBMClassifier
import pandas as pd


class MetaClassifier:
    """
    Meta-classifier with adaptive sample weighting
    """
    
    def __init__(self, meta_params, alpha_base=5.0, verbose=1):
        """
        Initialize MetaClassifier
        
        Args:
            meta_params: Parameters for meta LightGBM model
            alpha_base: Base alpha for adaptive weighting
            verbose: Verbosity level
        """
        self.meta_params = meta_params
        self.alpha_base = alpha_base
        self.verbose = verbose
        
        self.model_ = None
        self.feature_importance_ = None
    
    def compute_adaptive_weights(self, DS, entropy, confidence_variance):
        """
        Compute adaptive sample weights for focal loss
        
        α_i = α_base · (1 + DS_i / max(DS))
        uncertainty_i = entropy_i · confidence_variance_i
        
        Args:
            DS: Difficulty scores
            entropy: Prediction entropy from cascade
            confidence_variance: Confidence variance from cascade
            
        Returns:
            Adaptive weights for each sample
        """
        # Normalize difficulty score
        DS_normalized = DS / (np.max(DS) + 1e-10)
        
        # Adaptive alpha
        alpha_i = self.alpha_base * (1 + DS_normalized)
        
        # Uncertainty component
        uncertainty = entropy * confidence_variance
        
        # Final weight combines both
        # Higher weight for difficult and uncertain samples
        weights = alpha_i * (1 + uncertainty)
        
        return weights
    
    def train(self, X_meta, y, DS, feature_names=None):
        """
        Train meta-classifier
        
        Args:
            X_meta: Meta-features
            y: Labels
            DS: Difficulty scores
            feature_names: Names of features (optional)
            
        Returns:
            Trained model
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"{'Stage 4: Meta-Classifier Training':^60}")
            print(f"{'='*60}")
        
        # Find entropy and confidence variance columns
        # They should be in the meta features
        if feature_names is not None:
            try:
                entropy_idx = feature_names.index('entropy_pred')
                conf_var_idx = feature_names.index('confidence_variance')
                entropy = X_meta[:, entropy_idx]
                confidence_variance = X_meta[:, conf_var_idx]
            except ValueError:
                if self.verbose:
                    print("Warning: Could not find entropy/confidence_variance, using uniform weights")
                entropy = np.ones(len(X_meta))
                confidence_variance = np.ones(len(X_meta))
        else:
            entropy = np.ones(len(X_meta))
            confidence_variance = np.ones(len(X_meta))
        
        # Compute adaptive weights
        if self.verbose:
            print("\nComputing adaptive sample weights...")
        
        weights = self.compute_adaptive_weights(DS, entropy, confidence_variance)
        
        if self.verbose:
            print(f"Weight statistics:")
            print(f"  Mean: {np.mean(weights):.4f}")
            print(f"  Std:  {np.std(weights):.4f}")
            print(f"  Min:  {np.min(weights):.4f}")
            print(f"  Max:  {np.max(weights):.4f}")
            print(f"\nTraining meta-classifier...")
        
        # Train meta-model
        self.model_ = LGBMClassifier(**self.meta_params)
        self.model_.fit(X_meta, y, sample_weight=weights)
        
        # Extract feature importance
        if feature_names is not None:
            importance = self.model_.feature_importances_
            self.feature_importance_ = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            if self.verbose:
                print(f"\nTop 10 Most Important Features:")
                print(f"{'-'*60}")
                for idx, row in self.feature_importance_.head(10).iterrows():
                    print(f"  {row['feature']:<40} {row['importance']:>8.2f}")
        
        if self.verbose:
            print(f"\n{'='*60}\n")
        
        return self.model_
    
    def predict(self, X_meta):
        """
        Predict using meta-classifier
        
        Args:
            X_meta: Meta-features
            
        Returns:
            Predicted labels
        """
        if self.model_ is None:
            raise ValueError("Model not trained yet")
        
        return self.model_.predict(X_meta)
    
    def predict_proba(self, X_meta):
        """
        Predict probabilities using meta-classifier
        
        Args:
            X_meta: Meta-features
            
        Returns:
            Predicted probabilities
        """
        if self.model_ is None:
            raise ValueError("Model not trained yet")
        
        return self.model_.predict_proba(X_meta)[:, 1]
    
    def get_feature_importance(self):
        """Get feature importance DataFrame"""
        if self.feature_importance_ is None:
            raise ValueError("Model not trained yet or feature names not provided")
        
        return self.feature_importance_
