"""
Evaluation metrics for AGCL.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Optional prediction probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    
    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except:
            pass
    
    return metrics


def compute_node_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute metrics for node classification.
    
    Args:
        labels: Ground truth node labels
        predictions: Predicted node labels
        
    Returns:
        Dictionary of metrics
    """
    y_true = labels.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    
    # Filter out padding (-1)
    mask = y_true != -1
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    return compute_metrics(y_true, y_pred)


def compute_graph_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute metrics for graph classification.
    
    Args:
        labels: Ground truth graph labels
        predictions: Predicted graph labels
        
    Returns:
        Dictionary of metrics
    """
    y_true = labels.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    
    return compute_metrics(y_true, y_pred)


def compute_detection_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_class: int = 1,
) -> float:
    """
    Compute detection rate for attack class.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        attack_class: Class label for attack
        
    Returns:
        Detection rate
    """
    attack_indices = y_true == attack_class
    detected = y_pred[attack_indices] == attack_class
    return detected.sum() / (attack_indices.sum() + 1e-8)
