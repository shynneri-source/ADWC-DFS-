"""
Example usage script showing different ways to use ADWC-DFS
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adwc_dfs import ADWCDFS
from adwc_dfs.config import ADWCDFSConfig
from adwc_dfs.utils import calculate_metrics, print_metrics


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration"""
    print("\n" + "="*80)
    print("Example 1: Basic Usage")
    print("="*80)
    
    # Load and prepare data
    df = pd.read_csv('data/train.csv')
    df = df.sample(frac=0.05, random_state=42)  # 5% sample
    
    # Preprocess (simplified)
    target_col = 'is_fraud'
    cols_to_drop = [
        target_col, 'trans_date_trans_time', 'cc_num', 'merchant',
        'first', 'last', 'street', 'city', 'job', 'dob', 'trans_num'
    ]
    if df.columns[0] in ['Unnamed: 0', '']:
        cols_to_drop.append(df.columns[0])
    
    y = df[target_col].values
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    X = X.fillna(0).values
    
    # Split and standardize
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model with default config
    model = ADWCDFS(verbose=1)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test, threshold=0.5)
    
    # Evaluate
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_metrics(metrics, "Basic ADWC-DFS")
    
    return model


def example_2_custom_config():
    """Example 2: Custom configuration"""
    print("\n" + "="*80)
    print("Example 2: Custom Configuration")
    print("="*80)
    
    # Load and prepare data (same as example 1)
    df = pd.read_csv('data/train.csv')
    df = df.sample(frac=0.05, random_state=42)
    
    target_col = 'is_fraud'
    cols_to_drop = [
        target_col, 'trans_date_trans_time', 'cc_num', 'merchant',
        'first', 'last', 'street', 'city', 'job', 'dob', 'trans_num'
    ]
    if df.columns[0] in ['Unnamed: 0', '']:
        cols_to_drop.append(df.columns[0])
    
    y = df[target_col].values
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    X = X.fillna(0).values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create custom configuration
    config = ADWCDFSConfig()
    config.K_NEIGHBORS = 25  # Reduce neighbors for speed
    config.ALPHA = 0.5  # Increase CCDR weight
    config.BETA = 0.3
    config.GAMMA = 0.2
    config.SCALE_POS_WEIGHT_HARD = 20.0  # More focus on hard frauds
    
    print("\nCustom Configuration:")
    print(f"  K_NEIGHBORS: {config.K_NEIGHBORS}")
    print(f"  ALPHA: {config.ALPHA}")
    print(f"  SCALE_POS_WEIGHT_HARD: {config.SCALE_POS_WEIGHT_HARD}")
    
    # Train with custom config
    model = ADWCDFS(config=config, verbose=1)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test, threshold=0.5)
    
    # Evaluate
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_metrics(metrics, "Custom Config ADWC-DFS")
    
    return model


def example_3_save_load():
    """Example 3: Save and load model"""
    print("\n" + "="*80)
    print("Example 3: Save and Load Model")
    print("="*80)
    
    # Train a simple model
    df = pd.read_csv('data/train.csv')
    df = df.sample(frac=0.03, random_state=42)
    
    target_col = 'is_fraud'
    cols_to_drop = [
        target_col, 'trans_date_trans_time', 'cc_num', 'merchant',
        'first', 'last', 'street', 'city', 'job', 'dob', 'trans_num'
    ]
    if df.columns[0] in ['Unnamed: 0', '']:
        cols_to_drop.append(df.columns[0])
    
    y = df[target_col].values
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    X = X.fillna(0).values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train and save
    print("\nTraining model...")
    model = ADWCDFS(verbose=0)
    model.fit(X_train, y_train)
    
    model_path = 'results/example_model.pkl'
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Load and predict
    print(f"\nLoading model from {model_path}...")
    loaded_model = ADWCDFS.load(model_path)
    
    y_pred_proba = loaded_model.predict_proba(X_test)
    y_pred = loaded_model.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_metrics(metrics, "Loaded Model")
    
    return loaded_model


def example_4_feature_importance():
    """Example 4: Analyzing feature importance"""
    print("\n" + "="*80)
    print("Example 4: Feature Importance Analysis")
    print("="*80)
    
    # Train a model
    df = pd.read_csv('data/train.csv')
    df = df.sample(frac=0.03, random_state=42)
    
    target_col = 'is_fraud'
    cols_to_drop = [
        target_col, 'trans_date_trans_time', 'cc_num', 'merchant',
        'first', 'last', 'street', 'city', 'job', 'dob', 'trans_num'
    ]
    if df.columns[0] in ['Unnamed: 0', '']:
        cols_to_drop.append(df.columns[0])
    
    y = df[target_col].values
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    X = X.fillna(0).values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print("\nTraining model...")
    model = ADWCDFS(verbose=0)
    model.fit(X, y)
    
    # Get feature importance
    importance = model.get_feature_importance()
    
    print("\nTop 15 Most Important Meta-Features:")
    print("-" * 80)
    print(f"{'Feature':<40} {'Importance':>15} {'Cumulative %':>15}")
    print("-" * 80)
    
    total_importance = importance['importance'].sum()
    cumulative = 0
    
    for idx, row in importance.head(15).iterrows():
        cumulative += row['importance']
        cumulative_pct = (cumulative / total_importance) * 100
        print(f"{row['feature']:<40} {row['importance']:>15.2f} {cumulative_pct:>14.1f}%")
    
    # Analyze feature categories
    print("\n\nFeature Category Analysis:")
    print("-" * 80)
    
    categories = {
        'Cascade Predictions': ['P_easy', 'P_medium', 'P_hard'],
        'Density Features': ['DS', 'LID', 'CCDR'],
        'Disagreement': ['disagreement_score', 'disagreement_easy_medium', 
                        'disagreement_medium_hard', 'entropy_pred'],
        'Confidence': ['confidence_variance', 'confidence_std', 
                      'confidence_trajectory', 'prediction_range', 'mean_prediction'],
        'Consensus': ['neighbor_avg_pred', 'consensus_strength', 
                     'neighbor_pred_variance', 'neighbor_majority_vote']
    }
    
    for category, features in categories.items():
        category_importance = importance[
            importance['feature'].isin(features)
        ]['importance'].sum()
        category_pct = (category_importance / total_importance) * 100
        print(f"{category:<25} {category_importance:>10.2f} ({category_pct:>5.1f}%)")
    
    return model, importance


def example_5_threshold_tuning():
    """Example 5: Tuning classification threshold"""
    print("\n" + "="*80)
    print("Example 5: Threshold Tuning")
    print("="*80)
    
    # Prepare data
    df = pd.read_csv('data/train.csv')
    df = df.sample(frac=0.05, random_state=42)
    
    target_col = 'is_fraud'
    cols_to_drop = [
        target_col, 'trans_date_trans_time', 'cc_num', 'merchant',
        'first', 'last', 'street', 'city', 'job', 'dob', 'trans_num'
    ]
    if df.columns[0] in ['Unnamed: 0', '']:
        cols_to_drop.append(df.columns[0])
    
    y = df[target_col].values
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    X = X.fillna(0).values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    print("\nTraining model...")
    model = ADWCDFS(verbose=0)
    model.fit(X_train, y_train)
    
    # Get probabilities
    y_pred_proba = model.predict_proba(X_test)
    
    # Try different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\nPerformance at Different Thresholds:")
    print("-" * 80)
    print(f"{'Threshold':>10} {'Precision':>12} {'Recall':>12} {'F1':>12} {'FP':>8} {'FN':>8}")
    print("-" * 80)
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        print(f"{threshold:>10.1f} {metrics['precision']:>12.4f} "
              f"{metrics['recall']:>12.4f} {metrics['f1']:>12.4f} "
              f"{metrics['false_positives']:>8d} {metrics['false_negatives']:>8d}")
    
    print("\nRecommendation:")
    print("  - Lower threshold (0.3-0.4): Higher recall, more false positives")
    print("  - Higher threshold (0.6-0.7): Higher precision, more false negatives")
    print("  - Default (0.5): Balance between precision and recall")
    
    return model


def main():
    """Run all examples"""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "ADWC-DFS Example Usage Scripts".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    try:
        # Example 1: Basic usage
        model1 = example_1_basic_usage()
        
        # Example 2: Custom config
        model2 = example_2_custom_config()
        
        # Example 3: Save/load
        model3 = example_3_save_load()
        
        # Example 4: Feature importance
        model4, importance = example_4_feature_importance()
        
        # Example 5: Threshold tuning
        model5 = example_5_threshold_tuning()
        
        print("\n" + "#"*80)
        print("#" + " "*78 + "#")
        print("#" + "All Examples Completed Successfully!".center(78) + "#")
        print("#" + " "*78 + "#")
        print("#"*80 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
