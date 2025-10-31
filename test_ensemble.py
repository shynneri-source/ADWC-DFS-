"""
Quick test của Voting Ensemble với 10% data
Chạy nhanh để verify implementation trước khi train full
"""
import pandas as pd
import numpy as np
from ensemble_voting import VotingEnsemble
from sklearn.metrics import recall_score, precision_score, confusion_matrix
import time

print("🎯 QUICK TEST - Voting Ensemble (10% data)")
print("=" * 80)

# Load 10% data
print("\n📂 Loading 10% data...")
train_df = pd.read_csv('data/train.csv').sample(frac=0.1, random_state=42)
test_df = pd.read_csv('data/test.csv').sample(frac=0.1, random_state=42)

print(f"Train: {len(train_df):,} samples ({train_df['is_fraud'].sum()} frauds)")
print(f"Test:  {len(test_df):,} samples ({test_df['is_fraud'].sum()} frauds)")

# Preprocess data using ensemble_voting function
from ensemble_voting import preprocess_data
X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)

# Train ensemble với 3 models (nhanh hơn)
print("\n🚀 Training 3-model ensemble...")
start = time.time()

ensemble = VotingEnsemble(n_models=3, voting='soft', verbose=1)
ensemble.fit(X_train, y_train, X_val=X_test, y_val=y_test)

elapsed = time.time() - start
print(f"\n⏱️ Training time: {elapsed:.1f}s")

# Evaluate với visualization
results_df = ensemble.evaluate(X_test, y_test, save_plots=True, output_dir='results/test_ensemble')

print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)
print(results_df.to_string(index=False))

print("\n✅ Test complete! Ensemble implementation is working.")
print(f"\n📊 Plots saved to: results/test_ensemble/plots/")
print("\nTo train full ensemble:")
print("  python ensemble_voting.py --n_models 5")
print("  bash train_ensemble.sh -n 5")  # Background option
