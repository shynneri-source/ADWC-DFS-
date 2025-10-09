"""
Stage 1: Local Density Profiling
Computes Local Intrinsic Dimensionality (LID), Class-Conditional Density Ratio (CCDR),
and Difficulty Score (DS) for each sample.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class DensityProfiler:
    """
    Computes local density features for each sample:
    - Local Intrinsic Dimensionality (LID)
    - Class-Conditional Density Ratio (CCDR)
    - Difficulty Score (DS)
    """
    
    def __init__(self, k=30, alpha=0.4, beta=0.3, gamma=0.3, n_jobs=-1, verbose=1):
        """
        Initialize DensityProfiler
        
        Args:
            k: Number of neighbors for k-NN
            alpha: Weight for CCDR in difficulty score
            beta: Weight for LID in difficulty score
            gamma: Weight for max similarity in difficulty score
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        """
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.knn_model_ = None
        self.LID_ = None
        self.CCDR_ = None
        self.DS_ = None
        self.distances_ = None
        self.indices_ = None
    
    def compute_lid(self, distances):
        """
        Compute Local Intrinsic Dimensionality
        
        LID_i = -1 / k * Σ(log(d_j / d_k))
        
        Args:
            distances: Array of distances to k nearest neighbors
            
        Returns:
            Array of LID values
        """
        # Avoid log(0) and division by zero
        distances = np.maximum(distances, 1e-10)
        
        # Skip the first column (distance to self = 0)
        dist_to_neighbors = distances[:, 1:]
        dist_to_kth = distances[:, -1:]
        
        # Compute LID
        ratio = dist_to_neighbors / (dist_to_kth + 1e-10)
        ratio = np.maximum(ratio, 1e-10)  # Avoid log(0)
        
        LID = -np.mean(np.log(ratio), axis=1)
        
        return LID
    
    def compute_ccdr(self, indices, y):
        """
        Compute Class-Conditional Density Ratio
        
        CCDR_i = log(ρ_fraud(x_i) + ε) - log(ρ_legit(x_i) + ε)
        
        Args:
            indices: Array of indices of k nearest neighbors
            y: Labels
            
        Returns:
            Array of CCDR values
        """
        # Skip the first column (self)
        neighbor_indices = indices[:, 1:]
        
        # Get labels of neighbors
        neighbor_labels = y[neighbor_indices]
        
        # Compute density of each class in neighborhood
        fraud_density = np.mean(neighbor_labels == 1, axis=1)
        legit_density = np.mean(neighbor_labels == 0, axis=1)
        
        # Compute CCDR with epsilon to avoid log(0)
        epsilon = 1e-6
        CCDR = np.log(fraud_density + epsilon) - np.log(legit_density + epsilon)
        
        return CCDR
    
    def compute_max_similarity(self, X, indices):
        """
        Compute maximum cosine similarity to neighbors
        
        Args:
            X: Feature matrix
            indices: Array of indices of k nearest neighbors
            
        Returns:
            Array of max similarity values
        """
        n_samples = X.shape[0]
        max_similarities = np.zeros(n_samples)
        
        # Process in batches for efficiency
        batch_size = 1000
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_indices = indices[i:end_idx, 1:]  # Skip self
            
            # For each sample in batch
            for j, idx in enumerate(range(i, end_idx)):
                sample = X[idx:idx+1]
                neighbors = X[batch_indices[j]]
                
                # Compute cosine similarity
                similarities = cosine_similarity(sample, neighbors)[0]
                max_similarities[idx] = np.max(similarities)
        
        return max_similarities
    
    def compute_difficulty_score(self, X, y):
        """
        Compute difficulty score for each sample
        
        DS_i = α·|CCDR_i| + β·LID_i + γ·(1 - max_similarity_i)
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Dictionary with LID, CCDR, DS, distances, and indices
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"{'Stage 1: Local Density Profiling':^60}")
            print(f"{'='*60}")
            print(f"\nComputing k-NN (k={self.k})...")
        
        # Fit k-NN model
        self.knn_model_ = NearestNeighbors(
            n_neighbors=self.k + 1,  # +1 to include self
            n_jobs=self.n_jobs,
            algorithm='auto'
        )
        self.knn_model_.fit(X)
        
        # Find k nearest neighbors
        self.distances_, self.indices_ = self.knn_model_.kneighbors(X)
        
        if self.verbose:
            print("Computing Local Intrinsic Dimensionality (LID)...")
        
        # Compute LID
        self.LID_ = self.compute_lid(self.distances_)
        
        if self.verbose:
            print("Computing Class-Conditional Density Ratio (CCDR)...")
        
        # Compute CCDR
        self.CCDR_ = self.compute_ccdr(self.indices_, y)
        
        if self.verbose:
            print("Computing maximum neighbor similarity...")
        
        # Compute max similarity
        max_sim = self.compute_max_similarity(X, self.indices_)
        
        if self.verbose:
            print("Computing Difficulty Score (DS)...")
        
        # Compute Difficulty Score
        self.DS_ = (
            self.alpha * np.abs(self.CCDR_) + 
            self.beta * self.LID_ + 
            self.gamma * (1 - max_sim)
        )
        
        if self.verbose:
            print(f"\n{'Statistics':^60}")
            print(f"{'-'*60}")
            print(f"{'LID':<20} Mean: {np.mean(self.LID_):>8.4f}  Std: {np.std(self.LID_):>8.4f}")
            print(f"{'CCDR':<20} Mean: {np.mean(self.CCDR_):>8.4f}  Std: {np.std(self.CCDR_):>8.4f}")
            print(f"{'Difficulty Score':<20} Mean: {np.mean(self.DS_):>8.4f}  Std: {np.std(self.DS_):>8.4f}")
            print(f"{'='*60}\n")
        
        return {
            'LID': self.LID_,
            'CCDR': self.CCDR_,
            'DS': self.DS_,
            'distances': self.distances_,
            'indices': self.indices_
        }
    
    def get_stratification_masks(self, easy_percentile=33, hard_percentile=67):
        """
        Get stratification masks based on difficulty score
        
        Args:
            easy_percentile: Percentile threshold for easy samples
            hard_percentile: Percentile threshold for hard samples
            
        Returns:
            Dictionary with easy, medium, hard masks
        """
        if self.DS_ is None:
            raise ValueError("Difficulty score not computed. Call compute_difficulty_score first.")
        
        easy_threshold = np.percentile(self.DS_, easy_percentile)
        hard_threshold = np.percentile(self.DS_, hard_percentile)
        
        easy_mask = self.DS_ < easy_threshold
        medium_mask = (self.DS_ >= easy_threshold) & (self.DS_ < hard_threshold)
        hard_mask = self.DS_ >= hard_threshold
        
        if self.verbose:
            print(f"Stratification:")
            print(f"  Easy samples:   {np.sum(easy_mask):>6d} ({np.sum(easy_mask)/len(self.DS_)*100:>5.2f}%)")
            print(f"  Medium samples: {np.sum(medium_mask):>6d} ({np.sum(medium_mask)/len(self.DS_)*100:>5.2f}%)")
            print(f"  Hard samples:   {np.sum(hard_mask):>6d} ({np.sum(hard_mask)/len(self.DS_)*100:>5.2f}%)")
        
        return {
            'easy': easy_mask,
            'medium': medium_mask,
            'hard': hard_mask
        }
