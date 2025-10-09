"""
Metrics calculation utilities for ADWC-DFS
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive metrics for binary classification
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    
    # Specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # False positive rate
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Probabilistic metrics
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        
        # Calculate recall at precision >= 0.9
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        high_precision_mask = precisions >= 0.9
        if np.any(high_precision_mask):
            metrics['recall_at_precision_0.9'] = np.max(recalls[high_precision_mask])
        else:
            metrics['recall_at_precision_0.9'] = 0.0
    
    return metrics


def print_metrics(metrics, title="Model Performance"):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    print(f"\n{'Metric':<30} {'Value':>10}")
    print(f"{'-'*42}")
    
    # Order metrics for display
    ordered_keys = [
        'accuracy', 'precision', 'recall', 'f1', 'specificity',
        'roc_auc', 'pr_auc', 'recall_at_precision_0.9',
        'true_positives', 'true_negatives', 
        'false_positives', 'false_negatives'
    ]
    
    for key in ordered_keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, (int, np.integer)):
                print(f"{key:<30} {value:>10d}")
            else:
                print(f"{key:<30} {value:>10.4f}")
    
    print(f"{'='*60}\n")


def calculate_business_metrics(metrics, cost_fp=1, cost_fn=10):
    """
    Calculate business-oriented metrics
    
    Args:
        metrics: Dictionary of base metrics
        cost_fp: Cost of false positive
        cost_fn: Cost of false negative
        
    Returns:
        Dictionary with business metrics
    """
    business_metrics = {}
    
    fp = metrics['false_positives']
    fn = metrics['false_negatives']
    tp = metrics['true_positives']
    tn = metrics['true_negatives']
    
    # Total cost
    business_metrics['total_cost'] = fp * cost_fp + fn * cost_fn
    
    # Cost per transaction
    total_transactions = tp + tn + fp + fn
    business_metrics['cost_per_transaction'] = business_metrics['total_cost'] / total_transactions
    
    # Fraud caught rate
    business_metrics['fraud_caught_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # False alarm rate
    business_metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return business_metrics
