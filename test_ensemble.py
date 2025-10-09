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

X_train = train_df.drop(['fraud'], axis=1)
y_train = train_df['fraud']
X_test = test_df.drop(['fraud'], axis=1)
y_test = test_df['fraud']

print(f"Train: {len(train_df):,} samples ({y_train.sum()} frauds)")
print(f"Test:  {len(test_df):,} samples ({y_test.sum()} frauds)")

# Train ensemble với 3 models (nhanh hơn)
print("\n🚀 Training 3-model ensemble...")
start = time.time()

ensemble = VotingEnsemble(n_models=3, voting='soft', verbose=1)
ensemble.fit(X_train, y_train, X_val=X_test, y_val=y_test)

elapsed = time.time() - start
print(f"\n⏱️ Training time: {elapsed:.1f}s")

# Test strategies
print("\n" + "=" * 80)
print("📊 Testing Different Strategies")
print("=" * 80)

strategies = [
    ('Soft Voting (0.13)', lambda: ensemble.predict(X_test, threshold=0.13)),
    ('Soft Voting (0.10)', lambda: ensemble.predict(X_test, threshold=0.10)),
    ('Aggressive (2/3)', lambda: ensemble.predict_aggressive(X_test, min_votes=2)),
    ('Aggressive (1/3)', lambda: ensemble.predict_aggressive(X_test, min_votes=1)),
]

results = []
for name, predict_fn in strategies:
    y_pred = predict_fn()
    
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    results.append({
        'Strategy': name,
        'Recall': f"{recall:.4f}",
        'Precision': f"{precision:.4f}",
        'TP': tp,
        'FN': fn,
        'FP': fp
    })
    
    print(f"\n{name}:")
    print(f"  Recall: {recall:.4f} | Precision: {precision:.4f}")
    print(f"  Detected: {tp}/{tp+fn} frauds | Missed: {fn} | FP: {fp}")

# Summary
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

print("\n✅ Test complete! Ensemble implementation is working.")
print("\nTo train full ensemble:")
print("  python ensemble_voting.py --n_models 5")
print("\nTo train with full data:")
print("  python ensemble_voting.py --n_models 5 --train_path data/train.csv --test_path data/test.csv")
