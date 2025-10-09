"""
Evaluation and comparison script for ADWC-DFS
Compares against baseline methods
"""

import numpy as np
import pandas as pd
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

from adwc_dfs import ADWCDFS
from adwc_dfs.config import ADWCDFSConfig
from adwc_dfs.utils import (
    calculate_metrics, print_metrics, 
    setup_logger, close_logger
)


def load_and_preprocess(train_path, test_path=None, sample_frac=None):
    """Load and preprocess data (simplified version)"""
    print(f"\nLoading data from {train_path}...")
    df = pd.read_csv(train_path)
    
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=42)
    
    # Simple preprocessing
    target_col = 'is_fraud'
    
    # Drop non-numeric and identifier columns
    cols_to_drop = [
        target_col, 'trans_date_trans_time', 'cc_num', 'merchant',
        'first', 'last', 'street', 'city', 'job', 'dob', 'trans_num'
    ]
    
    if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
        cols_to_drop.append(df.columns[0])
    
    y = df[target_col].values
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Encode categoricals
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    
    X = X.fillna(0).values
    
    # Split if no test data
    if test_path is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, y_train = X, y
        # Load test data with same preprocessing
        df_test = pd.read_csv(test_path)
        y_test = df_test[target_col].values
        X_test = df_test.drop(columns=[col for col in cols_to_drop if col in df_test.columns])
        for col in X_test.select_dtypes(include=['object']).columns:
            X_test[col] = pd.Categorical(X_test[col]).codes
        X_test = X_test.fillna(0).values
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Fraud rate - Train: {np.mean(y_train):.4%}, Test: {np.mean(y_test):.4%}")
    
    return X_train, X_test, y_train, y_test


def train_baseline_xgboost(X_train, y_train):
    """Train baseline XGBoost with class weights"""
    print("\n>>> Training Baseline: XGBoost with Class Weights <<<")
    start = time.time()
    
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start
    print(f"Training time: {training_time:.2f}s")
    
    return model, training_time


def train_smote_xgboost(X_train, y_train):
    """Train XGBoost with SMOTE oversampling"""
    print("\n>>> Training Baseline: SMOTE + XGBoost <<<")
    start = time.time()
    
    # Apply SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_resampled.shape}, fraud rate: {np.mean(y_resampled):.4%}")
    
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    model.fit(X_resampled, y_resampled)
    
    training_time = time.time() - start
    print(f"Training time: {training_time:.2f}s")
    
    return model, training_time


def train_random_forest(X_train, y_train):
    """Train Random Forest with balanced class weights"""
    print("\n>>> Training Baseline: Random Forest <<<")
    start = time.time()
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start
    print(f"Training time: {training_time:.2f}s")
    
    return model, training_time


def train_adwcdfs(X_train, y_train, k_neighbors=30):
    """Train ADWC-DFS model"""
    print("\n>>> Training ADWC-DFS <<<")
    start = time.time()
    
    config = ADWCDFSConfig()
    config.K_NEIGHBORS = k_neighbors
    
    model = ADWCDFS(config=config, verbose=1)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start
    print(f"Training time: {training_time:.2f}s")
    
    return model, training_time


def evaluate_all_models(models_dict, X_test, y_test):
    """Evaluate all models and compare"""
    results = {}
    
    print("\n" + "="*80)
    print(f"{'Model Comparison on Test Set':^80}")
    print("="*80)
    
    for name, model in models_dict.items():
        print(f"\n>>> {name} <<<")
        
        # Predict
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        print_metrics(metrics, name)
        
        results[name] = metrics
    
    # Create comparison table
    print("\n" + "="*80)
    print(f"{'Summary Comparison':^80}")
    print("="*80)
    
    comparison_df = pd.DataFrame(results).T
    key_metrics = ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    
    print("\n" + comparison_df[key_metrics].to_string())
    
    return results, comparison_df


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare ADWC-DFS with baselines')
    parser.add_argument('--train_path', type=str, default='data/train.csv')
    parser.add_argument('--test_path', type=str, default='data/test.csv')
    parser.add_argument('--sample_frac', type=float, default=0.1,
                       help='Fraction of data to use (default: 0.1 for quick test)')
    parser.add_argument('--k_neighbors', type=int, default=30)
    parser.add_argument('--output_csv', type=str, default='results/comparison.csv')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save evaluation logs')
    parser.add_argument('--no_log', action='store_true',
                       help='Disable logging to file')
    
    args = parser.parse_args()
    
    # Setup logging
    tee_stdout = None
    log_file_path = None
    if not args.no_log:
        log_file_path, tee_stdout = setup_logger(
            log_dir=args.log_dir,
            prefix='evaluation',
            capture_stdout=True
        )
        print(f"\n📝 Log file: {log_file_path}")
    
    try:
        print("\n" + "#"*80)
        print(f"{'ADWC-DFS Evaluation & Comparison':^80}")
        print("#"*80)
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess(
        args.train_path, args.test_path, args.sample_frac
    )
    
    # Train all models
    models = {}
    timings = {}
    
    # Baseline 1: XGBoost with class weights
    model, train_time = train_baseline_xgboost(X_train, y_train)
    models['XGBoost (Class Weights)'] = model
    timings['XGBoost (Class Weights)'] = train_time
    
    # Baseline 2: SMOTE + XGBoost
    model, train_time = train_smote_xgboost(X_train, y_train)
    models['SMOTE + XGBoost'] = model
    timings['SMOTE + XGBoost'] = train_time
    
    # Baseline 3: Random Forest
    model, train_time = train_random_forest(X_train, y_train)
    models['Random Forest'] = model
    timings['Random Forest'] = train_time
    
    # ADWC-DFS
    model, train_time = train_adwcdfs(X_train, y_train, args.k_neighbors)
    models['ADWC-DFS'] = model
    timings['ADWC-DFS'] = train_time
    
    # Evaluate all models
    results, comparison_df = evaluate_all_models(models, X_test, y_test)
    
    # Add training times
    comparison_df['training_time'] = pd.Series(timings)
    
        # Save comparison
        comparison_df.to_csv(args.output_csv)
        print(f"\n\nComparison saved to {args.output_csv}")
        
        # Print final summary
        print("\n" + "#"*80)
        print(f"{'Evaluation Complete!':^80}")
        print("#"*80)
        print("\nKey Findings:")
        print(f"  Best F1 Score: {comparison_df['f1'].idxmax()} ({comparison_df['f1'].max():.4f})")
        print(f"  Best PR AUC: {comparison_df['pr_auc'].idxmax()} ({comparison_df['pr_auc'].max():.4f})")
        print(f"  Fastest Training: {comparison_df['training_time'].idxmin()} ({comparison_df['training_time'].min():.2f}s)")
        
        if log_file_path:
            print(f"\n📝 Full evaluation log saved to: {log_file_path}")
        
        print("\n" + "#"*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Close logger
        if tee_stdout is not None:
            close_logger(tee_stdout)


if __name__ == '__main__':
    main()
