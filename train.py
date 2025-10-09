"""
Training script for ADWC-DFS model on fraud detection data
"""

import numpy as np
import pandas as pd
import argparse
import os
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adwc_dfs import ADWCDFS
from adwc_dfs.config import ADWCDFSConfig
from adwc_dfs.utils import (
    calculate_metrics, print_metrics, plot_metrics, 
    plot_difficulty_distribution, setup_logger, close_logger
)


def load_data(train_path, test_path=None, sample_frac=None):
    """
    Load training and test data
    
    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV (optional)
        sample_frac: Fraction of data to sample (for quick testing)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"\nLoading training data from {train_path}...")
    df_train = pd.read_csv(train_path)
    
    # Sample data if requested
    if sample_frac is not None and sample_frac < 1.0:
        print(f"Sampling {sample_frac*100}% of data...")
        df_train = df_train.sample(frac=sample_frac, random_state=42)
    
    print(f"Training data shape: {df_train.shape}")
    print(f"Columns: {list(df_train.columns)}")
    
    # Load test data if provided
    if test_path and os.path.exists(test_path):
        print(f"\nLoading test data from {test_path}...")
        df_test = pd.read_csv(test_path)
        print(f"Test data shape: {df_test.shape}")
    else:
        print("\nNo test data provided, splitting training data...")
        df_train, df_test = train_test_split(
            df_train, test_size=0.2, random_state=42, stratify=df_train['is_fraud']
        )
        print(f"Train split: {df_train.shape}")
        print(f"Test split: {df_test.shape}")
    
    return df_train, df_test


def preprocess_data(df_train, df_test):
    """
    Preprocess the fraud detection data
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print("\nPreprocessing data...")
    
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
    
    print(f"Categorical columns: {list(categorical_cols)}")
    
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
    
    # Get feature names
    feature_names = X_train.columns.tolist()
    
    # Convert to numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape} (fraud rate: {np.mean(y_train):.4%})")
    print(f"  y_test: {y_test.shape} (fraud rate: {np.mean(y_test):.4%})")
    
    return X_train, X_test, y_train, y_test, feature_names


def train_model(X_train, y_train, config=None):
    """
    Train ADWC-DFS model
    
    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration object
        
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("Starting ADWC-DFS Training")
    print("="*60)
    
    # Initialize model
    model = ADWCDFS(config=config, verbose=1)
    
    # Train model
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, output_dir='results'):
    """
    Evaluate model on training and test data
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save results
    """
    print("\n" + "="*60)
    print("Evaluating Model")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # Evaluate on training data
    print("\n>>> Training Set Performance <<<")
    y_train_pred_proba = model.predict_proba(X_train)
    
    # Apply probability calibration using isotonic regression
    from sklearn.isotonic import IsotonicRegression
    print("\nApplying probability calibration...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_train_pred_proba, y_train)
    y_train_pred_proba_cal = calibrator.predict(y_train_pred_proba)
    
    # Find optimal threshold on calibrated probabilities
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_train, y_train_pred_proba_cal)
    
    # Strategy 1: Maximize F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1_optimal_idx = np.argmax(f1_scores)
    f1_optimal_threshold = thresholds[f1_optimal_idx] if f1_optimal_idx < len(thresholds) else 0.5
    
    # Strategy 2: Maximize recall with minimum precision >= 0.46 (slightly lower than 0.50)
    min_precision = 0.46
    valid_indices = np.where(precision >= min_precision)[0]
    
    # Strategy 3: F-beta=2.7 for strong recall focus
    min_precision_fbeta = 0.45  # Very low threshold
    valid_indices_fbeta = np.where(precision >= min_precision_fbeta)[0]
    if len(valid_indices_fbeta) > 0:
        # F-beta with beta=2.7
        beta = 2.7
        fbeta_scores = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-10)
        fbeta_at_valid = fbeta_scores[valid_indices_fbeta]
        best_fbeta_idx = valid_indices_fbeta[np.argmax(fbeta_at_valid)]
        fbeta_optimal_threshold = thresholds[best_fbeta_idx] if best_fbeta_idx < len(thresholds) else 0.5
    else:
        fbeta_optimal_threshold = f1_optimal_threshold
        best_fbeta_idx = f1_optimal_idx
    
    if len(valid_indices) > 0:
        # Among those with acceptable precision, find max recall
        recall_at_valid = recall[valid_indices]
        best_valid_idx = valid_indices[np.argmax(recall_at_valid)]
        recall_optimal_threshold = thresholds[best_valid_idx] if best_valid_idx < len(thresholds) else 0.5
        
        print(f"\n>>> Threshold optimization strategies (on calibrated probabilities):")
        print(f"   Strategy 1 (Max F1): threshold={f1_optimal_threshold:.4f}")
        print(f"     -> Precision={precision[f1_optimal_idx]:.4f}, Recall={recall[f1_optimal_idx]:.4f}, F1={f1_scores[f1_optimal_idx]:.4f}")
        print(f"   Strategy 2 (Max Recall @ Precision>={min_precision}): threshold={recall_optimal_threshold:.4f}")
        print(f"     -> Precision={precision[best_valid_idx]:.4f}, Recall={recall[best_valid_idx]:.4f}")
        print(f"   Strategy 3 (Max F-beta @ Precision>={min_precision_fbeta}): threshold={fbeta_optimal_threshold:.4f}")
        print(f"     -> Precision={precision[best_fbeta_idx]:.4f}, Recall={recall[best_fbeta_idx]:.4f}, F-beta={fbeta_scores[best_fbeta_idx]:.4f}")
        
        # Use strategy 3 (F-beta=2.7) for high recall
        optimal_threshold = fbeta_optimal_threshold
        print(f"\n>>> Selected: Strategy 3 (F-beta=2.7, High recall with reasonable precision)")
    else:
        # Fallback to F1 if no threshold meets precision constraint
        optimal_threshold = f1_optimal_threshold
        print(f"\n>>> Optimal threshold (Max F1): {optimal_threshold:.4f}")
        print(f"   At this threshold: Precision={precision[f1_optimal_idx]:.4f}, Recall={recall[f1_optimal_idx]:.4f}, F1={f1_scores[f1_optimal_idx]:.4f}")
    
    # Use optimal threshold for predictions
    y_train_pred = (y_train_pred_proba_cal >= optimal_threshold).astype(int)
    
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_pred_proba_cal)
    print_metrics(train_metrics, "Training Set Metrics")
    
    # Plot training metrics
    plot_metrics(y_train, y_train_pred_proba_cal, 
                save_path=os.path.join(output_dir, 'plots', 'train_metrics.png'))
    
    # Evaluate on test data with calibration and optimal threshold
    print("\n>>> Test Set Performance <<<")
    y_test_pred_proba = model.predict_proba(X_test)
    
    # Apply same calibration
    y_test_pred_proba_cal = calibrator.predict(y_test_pred_proba)
    y_test_pred = (y_test_pred_proba_cal >= optimal_threshold).astype(int)
    
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba_cal)
    test_metrics['optimal_threshold'] = optimal_threshold  # Store threshold
    print_metrics(test_metrics, "Test Set Metrics")
    
    # Plot test metrics
    plot_metrics(y_test, y_test_pred_proba_cal,
                save_path=os.path.join(output_dir, 'plots', 'test_metrics.png'))
    
    # Plot difficulty distribution (training data)
    if hasattr(model, 'density_features_') and model.density_features_ is not None:
        DS = model.density_features_['DS']
        plot_difficulty_distribution(DS, y_train,
                                    save_path=os.path.join(output_dir, 'plots', 'difficulty_distribution.png'))
    
    # Save metrics to file
    metrics_df = pd.DataFrame({
        'Metric': list(test_metrics.keys()),
        'Training': [train_metrics.get(k, 'N/A') for k in test_metrics.keys()],
        'Test': [test_metrics[k] for k in test_metrics.keys()]
    })
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    print(f"\nMetrics saved to {os.path.join(output_dir, 'metrics.csv')}")
    
    # Save feature importance
    try:
        feature_importance = model.get_feature_importance()
        feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        print(f"Feature importance saved to {os.path.join(output_dir, 'feature_importance.csv')}")
    except:
        print("Could not extract feature importance")
    
    return train_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train ADWC-DFS model for fraud detection')
    parser.add_argument('--train_path', type=str, default='data/train.csv',
                       help='Path to training data CSV')
    parser.add_argument('--test_path', type=str, default='data/test.csv',
                       help='Path to test data CSV')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--model_path', type=str, default='results/adwc_dfs_model.pkl',
                       help='Path to save trained model')
    parser.add_argument('--sample_frac', type=float, default=None,
                       help='Fraction of data to sample (for quick testing)')
    parser.add_argument('--k_neighbors', type=int, default=30,
                       help='Number of neighbors for k-NN')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save training logs')
    parser.add_argument('--no_log', action='store_true',
                       help='Disable logging to file')
    
    args = parser.parse_args()
    
    # Setup logging
    tee_stdout = None
    log_file_path = None
    if not args.no_log:
        log_file_path, tee_stdout = setup_logger(
            log_dir=args.log_dir,
            prefix='training',
            capture_stdout=True
        )
        print(f"\n📝 Log file: {log_file_path}")
    
    try:
        # Start timing
        start_time = time.time()
        
        print("\n" + "#"*60)
        print(f"{'ADWC-DFS Training Pipeline':^60}")
        print("#"*60)
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration:")
        print(f"  Train path: {args.train_path}")
        print(f"  Test path: {args.test_path}")
        print(f"  Output dir: {args.output_dir}")
        print(f"  Model path: {args.model_path}")
        print(f"  Sample fraction: {args.sample_frac}")
        print(f"  K neighbors: {args.k_neighbors}")
        if log_file_path:
            print(f"  Log file: {log_file_path}")
        
        # Load data
        df_train, df_test = load_data(args.train_path, args.test_path, args.sample_frac)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df_train, df_test)
        
        # Create config
        config = ADWCDFSConfig()
        config.K_NEIGHBORS = args.k_neighbors
        
        # Train model
        model = train_model(X_train, y_train, config)
        
        # Evaluate model
        train_metrics, test_metrics = evaluate_model(
            model, X_train, y_train, X_test, y_test, args.output_dir
        )
        
        # Save model
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model.save(args.model_path)
        print(f"\nModel saved to {args.model_path}")
        
        # Print summary
        total_time = time.time() - start_time
        print("\n" + "#"*60)
        print(f"{'Training Complete!':^60}")
        print("#"*60)
        print(f"\nTotal time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"\nKey Test Metrics:")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1 Score: {test_metrics['f1']:.4f}")
        print(f"  ROC AUC: {test_metrics.get('roc_auc', 0):.4f}")
        print(f"  PR AUC: {test_metrics.get('pr_auc', 0):.4f}")
        
        if log_file_path:
            print(f"\n📝 Full training log saved to: {log_file_path}")
        
        print("\n" + "#"*60 + "\n")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Close logger and restore stdout
        if tee_stdout is not None:
            close_logger(tee_stdout)


if __name__ == '__main__':
    main()
