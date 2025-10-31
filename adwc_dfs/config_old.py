"""
Configuration for ADWC-DFS algorithm
"""

class ADWCDFSConfig:
    """Configuration class for ADWC-DFS algorithm"""
    
    # Stage 1: Local Density Profiling
    K_NEIGHBORS = 30  # Number of neighbors for k-NN
    ALPHA = 0.4  # Weight for CCDR in difficulty score
    BETA = 0.3   # Weight for LID in difficulty score
    GAMMA = 0.3  # Weight for max similarity in difficulty score
    
    # Stage 2: Cascade Training
    EASY_PERCENTILE = 33    # Percentile threshold for easy samples
    HARD_PERCENTILE = 67    # Percentile threshold for hard samples
    
    # Sample weights for specialists - Adjusted for better balance
    EASY_MEDIUM_WEIGHT = 0.4    # Increased from 0.3
    MEDIUM_EASY_WEIGHT = 0.2    # Keep same
    MEDIUM_HARD_WEIGHT = 0.6    # Increased from 0.5  
    HARD_MEDIUM_WEIGHT = 0.5    # Increased from 0.4
    
    # Scale pos weight - BEST configuration (86.71% recall)
    SCALE_POS_WEIGHT_EASY = 40.0
    SCALE_POS_WEIGHT_MEDIUM = 60.0
    SCALE_POS_WEIGHT_HARD = 80.0
    
    # LightGBM parameters - BEST configuration
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
    
    # Stage 4: Meta-Classifier - BEST configuration (86.71% recall)
    ALPHA_BASE = 10.0
    
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
    
    # Feature importance threshold for meta-classifier
    TOP_FEATURES = 20
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and k.isupper()}
