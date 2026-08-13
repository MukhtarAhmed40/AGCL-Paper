"""
Multi-view graph construction for AGCL framework.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity


class MultiViewGraphBuilder(nn.Module):
    """
    Constructs complementary topology-aware and feature-aware graph views.
    """
    
    def __init__(
        self,
        num_nodes: int,
        feature_dim: int,
        k_neighbors: int = 10,
        similarity_threshold: float = 0.5,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.k_neighbors = k_neighbors
        self.similarity_threshold = similarity_threshold
    
    def build_topology_view(
        self,
        adj: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build topology-aware graph view.
        
        Preserves original connectivity patterns and enhances
        structural dependencies through k-nearest neighbors.
        
        Args:
            adj: Original adjacency matrix [N, N]
            features: Node features [N, d]
            
        Returns:
            Topology-aware adjacency matrix [N, N]
        """
        N = adj.shape[0]
        
        # Start with original adjacency
        topo_adj = adj.clone()
        
        # Add k-nearest neighbors based on feature similarity
        features_np = features.detach().cpu().numpy()
        sim_matrix = cosine_similarity(features_np)
        
        for i in range(N):
            # Get top-k similar nodes
            sim_scores = sim_matrix[i]
            top_k_indices = np.argsort(sim_scores)[-self.k_neighbors-1:-1]
            
            for j in top_k_indices:
                if i != j and sim_scores[j] > self.similarity_threshold:
                    topo_adj[i, j] = 1.0
                    topo_adj[j, i] = 1.0
        
        return topo_adj
    
    def build_feature_view(
        self,
        adj: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build feature-aware graph view.
        
        Refines node features through adaptive neighborhood aggregation.
        
        Args:
            adj: Original adjacency matrix [N, N]
            features: Node features [N, d]
            
        Returns:
            Feature-aware adjacency matrix [N, N]
            Refined node features [N, d]
        """
        N = adj.shape[0]
        
        # Normalize adjacency
        deg = adj.sum(dim=1, keepdim=True)
        deg_inv = torch.where(deg > 0, 1.0 / (deg + 1e-8), torch.zeros_like(deg))
        norm_adj = deg_inv * adj  # Row-normalized
        
        # Feature aggregation
        agg_features = norm_adj @ features
        
        # Feature refinement through attention
        feature_diff = features - agg_features
        attention_weights = torch.sigmoid(torch.norm(feature_diff, dim=1, keepdim=True))
        
        refined_features = attention_weights * features + (1 - attention_weights) * agg_features
        
        # Build feature-aware adjacency
        feat_adj = adj.clone()
        feat_sim = torch.cosine_similarity(refined_features.unsqueeze(0), refined_features.unsqueeze(1), dim=2)
        
        # Connect nodes with high feature similarity
        threshold = torch.quantile(feat_sim.flatten(), 0.3)
        feat_adj = torch.where(feat_sim > threshold, torch.ones_like(feat_adj), torch.zeros_like(feat_adj))
        
        # Ensure symmetry
        feat_adj = (feat_adj + feat_adj.T) / 2
        feat_adj = torch.where(feat_adj > 0.5, torch.ones_like(feat_adj), torch.zeros_like(feat_adj))
        
        return feat_adj, refined_features
    
    def forward(
        self,
        adj: torch.Tensor,
        features: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Build complementary graph views.
        
        Returns:
            Dictionary containing topology view and feature view
        """
        topo_adj = self.build_topology_view(adj, features)
        feat_adj, refined_features = self.build_feature_view(adj, features)
        
        return {
            "topology_adj": topo_adj,
            "topology_features": features,
            "feature_adj": feat_adj,
            "feature_features": refined_features,
            "original_adj": adj,
            "original_features": features,
        }
