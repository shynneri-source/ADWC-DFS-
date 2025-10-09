"""
BEST Configuration for Maximum Recall (86.71% on test set)
Use this configuration to reproduce the best fraud detection results
"""

class BestRecallConfig:
    """Best configuration achieving 86.71% recall on fraud detection"""
    
    # Stage 1: Local Density Profiling
    K_NEIGHBORS = 30
    ALPHA = 0.4  
    BETA = 0.3   
    GAMMA = 0.3  
    
    # Stage 2: Cascade Training
    EASY_PERCENTILE = 33
    HARD_PERCENTILE = 67
    
    # Sample weights
    EASY_MEDIUM_WEIGHT = 0.4
    MEDIUM_EASY_WEIGHT = 0.2
    MEDIUM_HARD_WEIGHT = 0.6
    HARD_MEDIUM_WEIGHT = 0.5
    
    # OPTIMAL scale_pos_weight for 86.71% recall
    SCALE_POS_WEIGHT_EASY = 40.0
    SCALE_POS_WEIGHT_MEDIUM = 60.0
    SCALE_POS_WEIGHT_HARD = 80.0
    
    # OPTIMAL LightGBM parameters for cascade
    CASCADE_PARAMS = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.02,
        'num_leaves': 31,
        'min_child_samples': 15,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 0.001,  
        'reg_alpha': 0.3,
        'reg_lambda': 1.0,
        'min_split_gain': 0.05,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    # Stage 4: Meta-Classifier  
    ALPHA_BASE = 10.0
    
    # OPTIMAL LightGBM parameters for meta-classifier
    META_PARAMS = {
        'n_estimators': 120,
        'max_depth': 5,
        'learning_rate': 0.02,
        'num_leaves': 20,
        'min_child_samples': 15,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 0.001,  
        'reg_alpha': 0.3,
        'reg_lambda': 1.0,
        'min_split_gain': 0.05,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    # General settings
    RANDOM_STATE = 42
    N_JOBS = -1
    VERBOSE = 1
    TOP_FEATURES = 20
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and k.isupper()}


# Threshold optimization settings for train.py
THRESHOLD_CONFIG = {
    'min_precision_strategy2': 0.50,
    'min_precision_fbeta': 0.48,
    'beta': 2.5,
    'use_calibration': True,  # Use Isotonic Regression calibration
    'selected_strategy': 3  # Use F-beta strategy
}

# Expected results with this configuration:
EXPECTED_RESULTS = {
    'test_recall': 0.8671,  # 86.71%
    'test_precision': 0.1816,  # 18.16%
    'test_f1': 0.3003,
    'test_roc_auc': 0.9898,
    'test_pr_auc': 0.5695,
    'true_positives': 1860,
    'false_negatives': 285,
    'false_positives': 8382
}
