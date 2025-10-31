"""
Configuration for ADWC-DFS algorithm
"""


class ADWCDFSConfig:
    """Configuration class for ADWC-DFS algorithm"""
    
    # Stage 1: Local Density Profiling
    K_NEIGHBORS = 30
    ALPHA = 0.4  # Weight for CCDR
    BETA = 0.3   # Weight for LID
    GAMMA = 0.3  # Weight for similarity
    
    # Stage 2: Cascade Training
    EASY_PERCENTILE = 33
    HARD_PERCENTILE = 67
    
    # Sample weights for cascade training
    EASY_MEDIUM_WEIGHT = 0.4
    MEDIUM_EASY_WEIGHT = 0.2
    MEDIUM_HARD_WEIGHT = 0.6
    HARD_MEDIUM_WEIGHT = 0.5
    
    # Scale pos weight for each specialist
    SCALE_POS_WEIGHT_EASY = 40.0
    SCALE_POS_WEIGHT_MEDIUM = 60.0
    SCALE_POS_WEIGHT_HARD = 80.0
    
    # LightGBM parameters for cascade models
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
    
    # LightGBM parameters for meta-classifier
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
