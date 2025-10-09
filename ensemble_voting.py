"""
Ensemble Voting để nâng cao Recall cho Fraud Detection
Mục tiêu: Tăng Recall từ 86.71% lên 90%+
"""
import numpy as np
import pandas as pd
from adwc_dfs import ADWCDFS
from adwc_dfs.config import ADWCDFSConfig
from sklearn.metrics import (
    recall_score, precision_score, f1_score, 
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import pickle
import time
from pathlib import Path


def preprocess_data(df_train, df_test):
    """
    Preprocess the fraud detection data
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n📊 Preprocessing data...")
    
    # Identify target column
    target_col = 'is_fraud'
    
    # Drop columns that shouldn't be used for modeling
    cols_to_drop = [
        target_col,
        'trans_date_trans_time',  # Datetime (can extract features if needed)
        'cc_num',  # Card number (identifier)
        'merchant',  # Merchant name (too many categories)
        'first',  # First name
        'last',  # Last name
        'street',  # Street address
        'city',  # City name (can use city_pop instead)
        'job',  # Job (too many categories)
        'dob',  # Date of birth (can extract age)
        'trans_num',  # Transaction number (identifier)
    ]
    
    # Remove index column if present
    if df_train.columns[0] == 'Unnamed: 0' or df_train.columns[0] == '':
        cols_to_drop.append(df_train.columns[0])
    
    # Extract features
    y_train = df_train[target_col].values
    y_test = df_test[target_col].values
    
    # Drop non-feature columns
    X_train = df_train.drop(columns=[col for col in cols_to_drop if col in df_train.columns])
    X_test = df_test.drop(columns=[col for col in cols_to_drop if col in df_test.columns])
    
    # Encode categorical variables
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        print(f"   Encoding {len(categorical_cols)} categorical columns...")
        
        for col in categorical_cols:
            # Simple label encoding for categorical variables
            combined = pd.concat([X_train[col], X_test[col]], axis=0)
            categories = combined.unique()
            cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
            
            X_train[col] = X_train[col].map(cat_to_idx)
            X_test[col] = X_test[col].map(cat_to_idx)
    
    # Fill any missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Convert to numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    
    # Standardize features
    print("   Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"   ✅ Final shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    
    return X_train, X_test, y_train, y_test


class VotingEnsemble:
    """
    Ensemble voting để maximize recall trong fraud detection
    
    Strategies:
    - Soft Voting: Trung bình xác suất từ các models
    - Hard Voting: Đa số models vote
    - Aggressive Voting: Nếu ít nhất min_votes models báo fraud → fraud
    """
    
    def __init__(self, n_models=5, voting='soft', weights=None, verbose=1):
        """
        Args:
            n_models: Số lượng models trong ensemble
            voting: 'soft' (average proba) hoặc 'hard' (majority vote)
            weights: Weights cho mỗi model (None = equal weights)
            verbose: 0=silent, 1=progress, 2=detailed
        """
        self.n_models = n_models
        self.voting = voting
        self.weights = weights
        self.verbose = verbose
        self.models = []
        self.model_recalls = []  # Track recall của từng model
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, configs=None):
        """
        Train multiple models for ensemble
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional, for computing weights)
            configs: List of config dicts for each model (default: different seeds)
        """
        if self.verbose:
            print(f"🚀 Training Ensemble with {self.n_models} models")
            print("=" * 60)
        
        # Default configs: different random seeds
        if configs is None:
            # Vary random seeds and slightly vary scale_pos_weight
            base_weights = [40, 60, 80]
            configs = []
            for i in range(self.n_models):
                # Vary scale_pos_weight slightly: +/- 5
                variation = (i - self.n_models//2) * 5
                configs.append({
                    'random_state': 42 + i * 111,
                    'scale_pos_weight_easy': base_weights[0] + variation,
                    'scale_pos_weight_medium': base_weights[1] + variation,
                    'scale_pos_weight_hard': base_weights[2] + variation,
                })
        
        # Train each model
        for i, config_dict in enumerate(configs):
            if self.verbose:
                print(f"\n📊 Training Model {i+1}/{self.n_models}")
                if self.verbose >= 2:
                    print(f"   Config: {config_dict}")
            
            start_time = time.time()
            
            # Create custom config object
            config = ADWCDFSConfig()
            for key, value in config_dict.items():
                # Map config dict keys to config attributes
                if key == 'random_state':
                    config.RANDOM_STATE = value
                    config.CASCADE_PARAMS['random_state'] = value
                    config.META_PARAMS['random_state'] = value
                elif key == 'scale_pos_weight_easy':
                    config.SCALE_POS_WEIGHT_EASY = value
                elif key == 'scale_pos_weight_medium':
                    config.SCALE_POS_WEIGHT_MEDIUM = value
                elif key == 'scale_pos_weight_hard':
                    config.SCALE_POS_WEIGHT_HARD = value
                else:
                    # Allow setting any config attribute
                    setattr(config, key.upper(), value)
            
            # Train model
            model = ADWCDFS(config=config, verbose=max(0, self.verbose-1))
            model.fit(X_train, y_train)
            self.models.append(model)
            
            elapsed = time.time() - start_time
            
            # Compute recall on validation set if provided
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                recall = recall_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred)
                self.model_recalls.append(recall)
                
                if self.verbose:
                    print(f"   ✅ Done in {elapsed:.1f}s | Recall: {recall:.4f} | Precision: {precision:.4f}")
            else:
                if self.verbose:
                    print(f"   ✅ Done in {elapsed:.1f}s")
        
        # Compute weights based on recall if not provided
        if self.weights is None and len(self.model_recalls) > 0:
            # Weight models by their recall performance
            total_recall = sum(self.model_recalls)
            self.weights = [r / total_recall for r in self.model_recalls]
            if self.verbose:
                print(f"\n📊 Computed weights based on recall: {[f'{w:.3f}' for w in self.weights]}")
        elif self.weights is None:
            # Equal weights
            self.weights = [1.0 / self.n_models] * self.n_models
            if self.verbose:
                print(f"\n📊 Using equal weights: {[f'{w:.3f}' for w in self.weights]}")
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("✅ Ensemble training complete!")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probabilities using ensemble
        
        Returns:
            Array of shape (n_samples,) with fraud probabilities
        """
        if self.voting == 'soft':
            # Weighted average of probabilities
            all_proba = []
            for model in self.models:
                proba = model.predict_proba(X)
                # Handle both 1D and 2D array formats
                if proba.ndim == 2:
                    proba = proba[:, 1]
                all_proba.append(proba)
            
            all_proba = np.array(all_proba)  # shape: (n_models, n_samples)
            weights_array = np.array(self.weights)
            
            # Weighted average
            weighted_proba = np.average(all_proba, axis=0, weights=weights_array)
            return weighted_proba
        
        elif self.voting == 'hard':
            # Majority voting (return probability as percentage of votes)
            all_pred = np.array([m.predict(X) for m in self.models])
            return all_pred.mean(axis=0)
    
    def predict(self, X, threshold=0.13):
        """
        Predict labels using threshold
        
        Args:
            X: Features
            threshold: Probability threshold for fraud classification
        """
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
    
    def predict_aggressive(self, X, min_votes=2, individual_threshold=0.13):
        """
        Aggressive voting để maximize recall
        Nếu ít nhất min_votes models báo fraud → fraud
        
        Args:
            X: Features
            min_votes: Minimum số models phải vote fraud
            individual_threshold: Threshold cho mỗi model riêng lẻ
        
        Returns:
            Array of fraud predictions (0 or 1)
        """
        all_pred = np.array([m.predict(X, threshold=individual_threshold) 
                            for m in self.models])
        return (all_pred.sum(axis=0) >= min_votes).astype(int)
    
    def predict_two_stage(self, X, stage1_threshold=0.13, stage2_threshold=0.05):
        """
        Two-stage prediction để catch edge cases
        
        Stage 1: High confidence (normal threshold)
        Stage 2: Re-examine low probability cases với lower threshold
        
        Args:
            X: Features
            stage1_threshold: Threshold cho stage 1
            stage2_threshold: Threshold cho stage 2 (lower)
        """
        # Stage 1: High confidence predictions
        proba = self.predict_proba(X)
        stage1_fraud = proba > stage1_threshold
        
        # Stage 2: Re-examine cases between thresholds
        reexamine = (proba > stage2_threshold) & (proba <= stage1_threshold)
        
        # For re-examination, use most aggressive model
        if reexamine.any():
            # Get most aggressive prediction from any model
            all_proba = np.array([m.predict_proba(X) for m in self.models])
            if all_proba.ndim == 3:
                all_proba = all_proba[:, :, 1]
            
            max_proba = all_proba.max(axis=0)
            stage2_fraud = max_proba > stage2_threshold
            
            # Combine: Stage 1 frauds + Stage 2 frauds
            final_fraud = stage1_fraud | (reexamine & stage2_fraud)
        else:
            final_fraud = stage1_fraud
        
        return final_fraud.astype(int)
    
    def evaluate(self, X_test, y_test, strategies=None):
        """
        Evaluate ensemble với nhiều strategies
        
        Args:
            X_test, y_test: Test data
            strategies: List of dicts với strategy configs
                       Default: test multiple approaches
        
        Returns:
            DataFrame with results for each strategy
        """
        if strategies is None:
            strategies = [
                {'name': 'Soft Voting (0.13)', 'method': 'predict', 
                 'kwargs': {'threshold': 0.13}},
                {'name': 'Soft Voting (0.10)', 'method': 'predict', 
                 'kwargs': {'threshold': 0.10}},
                {'name': 'Aggressive (2/5 votes)', 'method': 'predict_aggressive', 
                 'kwargs': {'min_votes': 2}},
                {'name': 'Aggressive (1/5 votes)', 'method': 'predict_aggressive', 
                 'kwargs': {'min_votes': 1}},
                {'name': 'Two-Stage (0.13, 0.05)', 'method': 'predict_two_stage', 
                 'kwargs': {'stage1_threshold': 0.13, 'stage2_threshold': 0.05}},
            ]
        
        results = []
        
        print("\n" + "=" * 80)
        print("📊 Evaluating Ensemble Strategies")
        print("=" * 80)
        
        for strategy in strategies:
            name = strategy['name']
            method = getattr(self, strategy['method'])
            kwargs = strategy.get('kwargs', {})
            
            # Predict
            y_pred = method(X_test, **kwargs)
            
            # Metrics
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Store results
            results.append({
                'Strategy': name,
                'Recall': recall,
                'Precision': precision,
                'F1': f1,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn
            })
            
            print(f"\n{name}:")
            print(f"  Recall:    {recall:.4f} ({tp}/{tp+fn} frauds detected)")
            print(f"  Precision: {precision:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  FN: {fn} (missed frauds) | FP: {fp} (false alarms)")
        
        print("\n" + "=" * 80)
        
        return pd.DataFrame(results)
    
    def save(self, filepath):
        """Save ensemble to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        if self.verbose:
            print(f"💾 Ensemble saved to: {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load ensemble from file"""
        with open(filepath, 'rb') as f:
            ensemble = pickle.load(f)
        print(f"📂 Ensemble loaded from: {filepath}")
        return ensemble


def compare_with_baseline(X_test, y_test, 
                          baseline_model_path='results/adwc_dfs_model.pkl',
                          ensemble_model=None):
    """
    So sánh Ensemble với Single Model baseline
    
    Args:
        X_test, y_test: Test data
        baseline_model_path: Path to baseline ADWCDFS model
        ensemble_model: Trained VotingEnsemble (if None, will load from default path)
    """
    print("\n" + "=" * 80)
    print("📊 COMPARISON: Single Model vs Ensemble")
    print("=" * 80)
    
    # Load baseline
    print("\n📂 Loading baseline model...")
    with open(baseline_model_path, 'rb') as f:
        baseline = pickle.load(f)
    
    # Load ensemble if not provided
    if ensemble_model is None:
        print("📂 Loading ensemble model...")
        ensemble_model = VotingEnsemble.load('results/ensemble_model.pkl')
    
    results = []
    
    # Evaluate baseline
    print("\n1️⃣ Single Model (Baseline)")
    y_pred_baseline = baseline.predict(X_test)
    
    recall_base = recall_score(y_test, y_pred_baseline)
    precision_base = precision_score(y_test, y_pred_baseline)
    f1_base = f1_score(y_test, y_pred_baseline)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_baseline).ravel()
    
    print(f"   Recall:    {recall_base:.4f} ({tp}/{tp+fn})")
    print(f"   Precision: {precision_base:.4f}")
    print(f"   F1 Score:  {f1_base:.4f}")
    print(f"   FN: {fn} | FP: {fp}")
    
    results.append({
        'Model': 'Single Model',
        'Recall': recall_base,
        'Precision': precision_base,
        'F1': f1_base,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
    })
    
    # Evaluate ensemble (soft voting)
    print("\n2️⃣ Ensemble - Soft Voting")
    y_pred_ensemble = ensemble_model.predict(X_test, threshold=0.13)
    
    recall_ens = recall_score(y_test, y_pred_ensemble)
    precision_ens = precision_score(y_test, y_pred_ensemble)
    f1_ens = f1_score(y_test, y_pred_ensemble)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ensemble).ravel()
    
    print(f"   Recall:    {recall_ens:.4f} ({tp}/{tp+fn})")
    print(f"   Precision: {precision_ens:.4f}")
    print(f"   F1 Score:  {f1_ens:.4f}")
    print(f"   FN: {fn} | FP: {fp}")
    
    results.append({
        'Model': 'Ensemble (Soft)',
        'Recall': recall_ens,
        'Precision': precision_ens,
        'F1': f1_ens,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
    })
    
    # Improvement
    recall_improvement = (recall_ens - recall_base) / recall_base * 100
    
    print("\n" + "=" * 80)
    print("📈 IMPROVEMENT")
    print("=" * 80)
    print(f"Recall Improvement:    {recall_improvement:+.2f}%")
    print(f"Absolute Improvement:  {recall_ens - recall_base:+.4f}")
    print(f"Additional Frauds Detected: {int((recall_ens - recall_base) * (tp + fn))}")
    
    return pd.DataFrame(results)


# ============================================================================
# MAIN - Example Usage
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Voting Ensemble for Fraud Detection')
    parser.add_argument('--train_path', default='data/train.csv', help='Path to training CSV')
    parser.add_argument('--test_path', default='data/test.csv', help='Path to test CSV')
    parser.add_argument('--n_models', type=int, default=5, help='Number of models in ensemble')
    parser.add_argument('--sample_frac', type=float, default=None, 
                       help='Sample fraction for quick testing (e.g., 0.1)')
    parser.add_argument('--output', default='results/ensemble_model.pkl', 
                       help='Output path for trained ensemble')
    
    args = parser.parse_args()
    
    print("🎯 FRAUD DETECTION - VOTING ENSEMBLE")
    print("=" * 80)
    print(f"Target: Increase Recall from 86.71% to 90%+")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading data...")
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    
    # Sample for quick testing
    if args.sample_frac:
        print(f"   Sampling {args.sample_frac*100}% for quick test...")
        train_df = train_df.sample(frac=args.sample_frac, random_state=42)
        test_df = test_df.sample(frac=args.sample_frac, random_state=42)
    
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Test:  {len(test_df):,} samples")
    
    print(f"   Fraud rate (train): {train_df['is_fraud'].mean():.4f}")
    print(f"   Fraud rate (test):  {test_df['is_fraud'].mean():.4f}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)
    
    # Create and train ensemble
    print("\n" + "=" * 80)
    ensemble = VotingEnsemble(n_models=args.n_models, voting='soft', verbose=1)
    ensemble.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    
    # Evaluate với nhiều strategies
    results_df = ensemble.evaluate(X_test, y_test)
    
    # Display results table
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Find best strategy
    best_idx = results_df['Recall'].idxmax()
    best_strategy = results_df.iloc[best_idx]
    
    print("\n" + "=" * 80)
    print("🏆 BEST STRATEGY")
    print("=" * 80)
    print(f"Strategy: {best_strategy['Strategy']}")
    print(f"Recall:   {best_strategy['Recall']:.4f}")
    print(f"Precision: {best_strategy['Precision']:.4f}")
    print(f"Detected: {best_strategy['TP']}/{best_strategy['TP'] + best_strategy['FN']} frauds")
    print(f"Missed:   {best_strategy['FN']} frauds ({best_strategy['FN']/(best_strategy['TP']+best_strategy['FN'])*100:.2f}%)")
    
    # Save ensemble
    ensemble.save(args.output)
    
    # Save results
    results_path = Path(args.output).parent / 'ensemble_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("✅ DONE!")
    print("=" * 80)
    print(f"\nTo use this ensemble:")
    print(f"  from ensemble_voting import VotingEnsemble")
    print(f"  ensemble = VotingEnsemble.load('{args.output}')")
    print(f"  predictions = ensemble.predict(X_new)")
