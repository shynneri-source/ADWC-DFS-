"""
Visualization utilities for ADWC-DFS
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix


def plot_metrics(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve, PR curve, and confusion matrix
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_pred_proba)
    
    axes[0].plot(fpr, tpr, label=f'ROC (AUC={auc:.4f})', linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y_true, y_pred_proba)
    
    axes[1].plot(recall, precision, label=f'PR (AP={ap:.4f})', linewidth=2)
    axes[1].axhline(y=np.sum(y_true)/len(y_true), color='k', linestyle='--', 
                    label=f'Baseline ({np.sum(y_true)/len(y_true):.4f})')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    # Confusion Matrix
    y_pred = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    axes[2].set_xlabel('Predicted', fontsize=12)
    axes[2].set_ylabel('Actual', fontsize=12)
    axes[2].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_difficulty_distribution(DS, y, save_path=None):
    """
    Plot difficulty score distribution by class
    
    Args:
        DS: Difficulty scores
        y: Labels
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(DS[y == 0], bins=50, alpha=0.6, label='Legit', color='blue', density=True)
    axes[0].hist(DS[y == 1], bins=50, alpha=0.6, label='Fraud', color='red', density=True)
    axes[0].axvline(np.percentile(DS, 33), color='green', linestyle='--', 
                    label=f'33rd percentile', linewidth=2)
    axes[0].axvline(np.percentile(DS, 67), color='orange', linestyle='--', 
                    label=f'67th percentile', linewidth=2)
    axes[0].set_xlabel('Difficulty Score', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Difficulty Score Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Box plot
    data_to_plot = [DS[y == 0], DS[y == 1]]
    bp = axes[1].boxplot(data_to_plot, labels=['Legit', 'Fraud'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.6)
    axes[1].set_ylabel('Difficulty Score', fontsize=12)
    axes[1].set_title('Difficulty Score by Class', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_cascade_predictions(predictions_dict, y_true, save_path=None):
    """
    Plot prediction distributions from cascade models
    
    Args:
        predictions_dict: Dictionary with 'easy', 'medium', 'hard' predictions
        y_true: True labels
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = ['easy', 'medium', 'hard']
    colors = ['green', 'orange', 'red']
    
    for idx, (model, color) in enumerate(zip(models, colors)):
        preds = predictions_dict[model]
        
        axes[idx].hist(preds[y_true == 0], bins=50, alpha=0.6, 
                      label='Legit', color='blue', density=True)
        axes[idx].hist(preds[y_true == 1], bins=50, alpha=0.6, 
                      label='Fraud', color='red', density=True)
        axes[idx].set_xlabel('Prediction Probability', fontsize=12)
        axes[idx].set_ylabel('Density', fontsize=12)
        axes[idx].set_title(f'{model.capitalize()} Model Predictions', 
                           fontsize=14, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(feature_importance_df, top_n=20, save_path=None):
    """
    Plot feature importance
    
    Args:
        feature_importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    
    top_features = feature_importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
