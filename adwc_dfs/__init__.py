"""
ADWC-DFS: Adaptive Density-Weighted Cascade with Dynamic Feature Synthesis
A meta-learning framework for imbalanced classification (fraud detection)
"""

__version__ = "1.0.0"
__author__ = "ADWC-DFS Team"

from .models.adwc_dfs import ADWCDFS

__all__ = ["ADWCDFS"]
