"""
Analyze false negatives to understand what frauds we're missing
"""
import numpy as np
import pandas as pd
import pickle
from adwc_dfs import ADWCDFS
from sklearn.preprocessing import StandardScaler

# Load data
print("Loading data...")
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# Preprocess (same as train.py)
def preprocess(df):
    target_col = 'is_fraud'
    cols_to_drop = [
        target_col, 'trans_date_trans_time', 'cc_num', 'merchant',
        'first', 'last', 'street', 'city', 'job', 'dob', 'trans_num'
    ]
    if df.columns[0] == 'Unnamed: 0':
        cols_to_drop.append('Unnamed: 0')
    
    y = df[target_col].values
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Encode categoricals
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = pd.Categorical(X[col]).codes
    
    X = X.fillna(0).values
    return X, y

print("Preprocessing...")
X_test, y_test = preprocess(df_test)

# Standardize
X_train_raw, y_train = preprocess(df_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test)

# Load model
print("\nLoading model...")
model = ADWCDFS.load('results/adwc_dfs_model.pkl')

# Predict
print("Making predictions...")
y_pred_proba = model.predict_proba(X_test)

# Apply calibration and threshold
from sklearn.isotonic import IsotonicRegression
calibrator = IsotonicRegression(out_of_bounds='clip')
y_train_pred = model.predict_proba(X_train)
calibrator.fit(y_train_pred, y_train)
y_pred_proba_cal = calibrator.predict(y_pred_proba)

threshold = 0.1379
y_pred = (y_pred_proba_cal >= threshold).astype(int)

# Analyze false negatives
fn_mask = (y_test == 1) & (y_pred == 0)
fn_indices = np.where(fn_mask)[0]
fn_probas = y_pred_proba_cal[fn_mask]

tp_mask = (y_test == 1) & (y_pred == 1)
tp_probas = y_pred_proba_cal[tp_mask]

print(f"\n{'='*60}")
print("FALSE NEGATIVE ANALYSIS")
print(f"{'='*60}")
print(f"Total frauds in test: {np.sum(y_test)}")
print(f"Detected (TP): {np.sum(tp_mask)}")
print(f"Missed (FN): {np.sum(fn_mask)}")
print(f"Recall: {np.sum(tp_mask)/np.sum(y_test)*100:.2f}%")

print(f"\n{'='*60}")
print("PROBABILITY DISTRIBUTION")
print(f"{'='*60}")
print("\nMissed Frauds (FN) - Probability Stats:")
print(f"  Mean: {np.mean(fn_probas):.4f}")
print(f"  Median: {np.median(fn_probas):.4f}")
print(f"  Std: {np.std(fn_probas):.4f}")
print(f"  Min: {np.min(fn_probas):.4f}")
print(f"  Max: {np.max(fn_probas):.4f}")
print(f"  25th percentile: {np.percentile(fn_probas, 25):.4f}")
print(f"  75th percentile: {np.percentile(fn_probas, 75):.4f}")

print("\nDetected Frauds (TP) - Probability Stats:")
print(f"  Mean: {np.mean(tp_probas):.4f}")
print(f"  Median: {np.median(tp_probas):.4f}")
print(f"  Std: {np.std(tp_probas):.4f}")
print(f"  Min: {np.min(tp_probas):.4f}")
print(f"  Max: {np.max(tp_probas):.4f}")

# How many FN would we catch with lower threshold?
print(f"\n{'='*60}")
print("THRESHOLD SENSITIVITY ANALYSIS")
print(f"{'='*60}")
print(f"\nCurrent threshold: {threshold:.4f}")

for new_thresh in [0.12, 0.10, 0.08, 0.06, 0.04]:
    additional_tp = np.sum(fn_probas >= new_thresh)
    additional_fp = np.sum((y_test == 0) & (y_pred_proba_cal >= new_thresh) & (y_pred_proba_cal < threshold))
    new_recall = (np.sum(tp_mask) + additional_tp) / np.sum(y_test)
    print(f"\nThreshold {new_thresh:.4f}:")
    print(f"  Additional frauds caught: {additional_tp}")
    print(f"  Additional false positives: {additional_fp}")
    print(f"  New recall: {new_recall*100:.2f}%")
    print(f"  Total alerts: {np.sum(y_pred_proba_cal >= new_thresh)}")

# Feature analysis of FN
print(f"\n{'='*60}")
print("FEATURE CHARACTERISTICS OF MISSED FRAUDS")
print(f"{'='*60}")

X_test_df = pd.read_csv('data/test.csv')
feature_cols = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 
                'category', 'gender', 'state', 'zip', 'unix_time']

print("\nMissed Frauds vs All Frauds comparison:")
all_frauds_mask = (y_test == 1)

for col in ['amt', 'lat', 'long', 'city_pop']:
    if col in X_test_df.columns:
        fn_values = X_test_df.loc[fn_indices, col].values
        all_fraud_values = X_test_df.loc[all_frauds_mask, col].values
        print(f"\n{col}:")
        print(f"  Missed frauds - Mean: {np.mean(fn_values):.2f}, Std: {np.std(fn_values):.2f}")
        print(f"  All frauds - Mean: {np.mean(all_fraud_values):.2f}, Std: {np.std(all_fraud_values):.2f}")

print(f"\n{'='*60}")
