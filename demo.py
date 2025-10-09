"""
Quick demo of ADWC-DFS on a small sample of data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adwc_dfs import ADWCDFS
from adwc_dfs.config import ADWCDFSConfig
from adwc_dfs.utils import (
    calculate_metrics, print_metrics,
    setup_logger, close_logger
)


def main():
    # Setup logging
    log_file_path, tee_stdout = setup_logger(
        log_dir='logs',
        prefix='demo',
        capture_stdout=True
    )
    
    try:
        print("\n" + "="*60)
        print("ADWC-DFS Demo - Quick Test")
        print("="*60)
        print(f"\n📝 Log file: {log_file_path}")
        
        # Load small sample of data
        print("\nLoading data...")
        df = pd.read_csv('data/train.csv')
        
        # Use only 10% for quick demo
        df = df.sample(frac=0.1, random_state=42)
        print(f"Using {len(df)} samples for demo")
        
        # Simple preprocessing
        target_col = 'is_fraud'
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
        print(f"Fraud rate - Train: {np.mean(y_train):.4%}, Test: {np.mean(y_test):.4%}")
        
        # Initialize ADWC-DFS with reduced neighbors for speed
        print("\nInitializing ADWC-DFS...")
        config = ADWCDFSConfig()
        config.K_NEIGHBORS = 20  # Reduced for demo speed
        
        model = ADWCDFS(config=config, verbose=1)
        
        # Train
        print("\nTraining model...")
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        y_pred_proba = model.predict_proba(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        print_metrics(metrics, "ADWC-DFS Performance")
        
        # Show feature importance
        try:
            print("\nTop 10 Most Important Meta-Features:")
            print("-" * 60)
            importance = model.get_feature_importance()
            for idx, row in importance.head(10).iterrows():
                print(f"  {row['feature']:<40} {row['importance']:>8.2f}")
        except:
            print("Could not extract feature importance")
        
        # Example predictions
        print("\n" + "="*60)
        print("Sample Predictions")
        print("="*60)
        
        # Show a few fraud predictions
        fraud_indices = np.where(y_test == 1)[0][:5]
        print("\nActual Fraud Cases:")
        for idx in fraud_indices:
            print(f"  Sample {idx}: Predicted={y_pred_proba[idx]:.4f}, "
                  f"Label={'FRAUD' if y_pred[idx] == 1 else 'LEGIT'} "
                  f"({'✓' if y_pred[idx] == y_test[idx] else '✗'})")
        
        # Show a few legit predictions
        legit_indices = np.where(y_test == 0)[0][:5]
        print("\nActual Legit Cases:")
        for idx in legit_indices:
            print(f"  Sample {idx}: Predicted={y_pred_proba[idx]:.4f}, "
                  f"Label={'FRAUD' if y_pred[idx] == 1 else 'LEGIT'} "
                  f"({'✓' if y_pred[idx] == y_test[idx] else '✗'})")
        
        print("\n" + "="*60)
        print("Demo Complete!")
        print("="*60)
        print("\nTo run full training:")
        print("  uv run train.py --train_path data/train.csv --test_path data/test.csv")
        print("\nTo compare with baselines:")
        print("  uv run evaluate.py --sample_frac 0.1")
        print(f"\n📝 Demo log saved to: {log_file_path}")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Close logger
        close_logger(tee_stdout)


if __name__ == '__main__':
    main()
