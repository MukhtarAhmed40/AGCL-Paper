"""
Adversarial evaluation utilities for AGCL.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from torch.utils.data import DataLoader

from agcl.main import AGCL
from agcl.utils.metrics import compute_metrics


def node_feature_corruption(
    features: torch.Tensor,
    corruption_rate: float,
) -> torch.Tensor:
    """
    Corrupt node features by adding Gaussian noise.
    
    Args:
        features: Original features [N, d]
        corruption_rate: Rate of corruption
        
    Returns:
        Corrupted features
    """
    noise = torch.randn_like(features) * corruption_rate
    return features + noise


def edge_perturbation(
    adj: torch.Tensor,
    perturbation_rate: float,
) -> torch.Tensor:
    """
    Perturb edges by random addition and deletion.
    
    Args:
        adj: Original adjacency matrix [N, N]
        perturbation_rate: Rate of perturbation
        
    Returns:
        Perturbed adjacency matrix
    """
    N = adj.shape[0]
    
    # Create mask for valid edges
    mask = torch.ones_like(adj)
    mask = torch.triu(mask, diagonal=1)  # Upper triangular
    
    # Random edge flips
    flip_mask = torch.rand_like(adj) < perturbation_rate
    flip_mask = flip_mask & mask.bool()
    
    # Apply flips
    adj_perturbed = adj.clone()
    adj_perturbed[flip_mask] = 1 - adj_perturbed[flip_mask]
    
    # Symmetrize
    adj_perturbed = (adj_perturbed + adj_perturbed.T) / 2
    adj_perturbed = torch.clamp(adj_perturbed, 0, 1)
    
    return adj_perturbed


def nettack_attack(
    adj: torch.Tensor,
    features: torch.Tensor,
    target_nodes: List[int],
    num_perturbations: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified Nettack attack implementation.
    
    Args:
        adj: Original adjacency matrix [N, N]
        features: Original node features [N, d]
        target_nodes: Nodes to attack
        num_perturbations: Number of edge perturbations
        
    Returns:
        Perturbed adjacency and features
    """
    adj_adv = adj.clone()
    features_adv = features.clone()
    N = adj.shape[0]
    
    # For each target node, add edges to random nodes
    for target in target_nodes:
        # Get current degree
        current_deg = adj_adv[target].sum().item()
        
        # Add edges to non-neighbors
        non_neighbors = torch.where(adj_adv[target] == 0)[0]
        non_neighbors = non_neighbors[non_neighbors != target]
        
        num_to_add = min(num_perturbations, len(non_neighbors))
        if num_to_add > 0:
            chosen = non_neighbors[torch.randperm(len(non_neighbors))[:num_to_add]]
            adj_adv[target, chosen] = 1
            adj_adv[chosen, target] = 1
    
    return adj_adv, features_adv


def metattack_attack(
    adj: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    perturbation_rate: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified Metattack attack (poisoning) implementation.
    
    Args:
        adj: Original adjacency matrix [N, N]
        features: Original node features [N, d]
        labels: Node labels
        perturbation_rate: Rate of perturbation
        
    Returns:
        Poisoned adjacency and features
    """
    adj_adv = adj.clone()
    features_adv = features.clone()
    N = adj.shape[0]
    
    # Random edge flips
    num_edges = adj.sum().item() / 2
    num_perturbations = int(num_edges * perturbation_rate)
    
    # Get all possible edge flips
    edges = torch.triu(adj, diagonal=1).nonzero()
    
    if len(edges) > 0:
        # Randomly select edges to flip
        idx = torch.randperm(len(edges))[:min(num_perturbations, len(edges))]
        selected_edges = edges[idx]
        
        for i, j in selected_edges:
            adj_adv[i, j] = 1 - adj_adv[i, j]
            adj_adv[j, i] = 1 - adj_adv[j, i]
    
    return adj_adv, features_adv


def evaluate_robustness(
    agcl: AGCL,
    loader: DataLoader,
    config,
    perturbation_rates: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2],
) -> Dict[str, List[float]]:
    """
    Evaluate robustness under various perturbations.
    
    Args:
        agcl: Trained AGCL model
        loader: Data loader
        config: Configuration
        perturbation_rates: List of perturbation rates
        
    Returns:
        Metrics at each perturbation level
    """
    results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "detection_rate": [],
    }
    
    agcl.encoder.eval()
    
    for rate in perturbation_rates:
        batch_metrics = []
        detection_rates = []
        
        for batch in loader:
            adj = batch["adj"].to(agcl.device)
            features = batch["features"].to(agcl.device)
            labels = batch.get("labels")
            
            # Apply perturbation
            if rate > 0:
                adj_perturbed = edge_perturbation(adj, rate)
                features_perturbed = node_feature_corruption(features, rate)
            else:
                adj_perturbed = adj
                features_perturbed = features
            
            # Predict
            with torch.no_grad():
                embeddings, _ = agcl.encoder(features_perturbed, adj_perturbed)
            
            # Simple classifier for evaluation
            if hasattr(agcl, "classifier"):
                preds = agcl.classifier(embeddings)
                metrics = compute_metrics(labels.cpu().numpy(), preds.cpu().numpy())
                batch_metrics.append(metrics)
                
                # Detection rate
                det_rate = compute_detection_rate(
                    labels.cpu().numpy(),
                    preds.cpu().numpy(),
                )
                detection_rates.append(det_rate)
        
        if batch_metrics:
            for key in results.keys():
                if key != "detection_rate":
                    values = [m.get(key, 0) for m in batch_metrics if key in m]
                    results[key].append(np.mean(values))
                else:
                    results[key].append(np.mean(detection_rates))
    
    return results
