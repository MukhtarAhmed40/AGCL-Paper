"""
Adaptive contrastive learning for AGCL framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class AdaptiveContrastiveLearning(nn.Module):
    """
    Adaptive graph view alignment through contrastive learning.
    
    Dynamically selects positive and negative pairs based on semantic similarity.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        lambda_similarity: float = 0.6,
        num_negatives: int = 256,
        hard_negative_mining: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_similarity = lambda_similarity
        self.num_negatives = num_negatives
        self.hard_negative_mining = hard_negative_mining
    
    def compute_semantic_similarity(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute joint semantic similarity between views.
        
        Args:
            h1: Embeddings from view 1 [N, d]
            h2: Embeddings from view 2 [N, d]
            
        Returns:
            Semantic similarity matrix [N, N]
        """
        # Feature similarity
        sim_f = F.cosine_similarity(h1.unsqueeze(1), h2.unsqueeze(0), dim=2)
        
        # Structural similarity (using L2 distance as proxy)
        sim_g = -torch.cdist(h1, h2, p=2)
        
        # Normalize structural similarity
        sim_g = (sim_g - sim_g.min()) / (sim_g.max() - sim_g.min() + 1e-8)
        
        # Joint semantic similarity
        sim = self.lambda_similarity * sim_f + (1 - self.lambda_similarity) * sim_g
        
        return sim
    
    def select_positive_pairs(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamically select positive pairs based on semantic similarity.
        
        Args:
            h1: Embeddings from view 1 [N, d]
            h2: Embeddings from view 2 [N, d]
            
        Returns:
            Positive indices for view 1 and view 2
        """
        sim_matrix = self.compute_semantic_similarity(h1, h2)
        
        # Find best matches for each node
        best_matches = torch.argmax(sim_matrix, dim=1)
        
        # Ensure symmetric matching
        pos1 = torch.arange(h1.shape[0], device=h1.device)
        pos2 = best_matches
        
        return pos1, pos2
    
    def select_negative_pairs(
        self,
        h_anchor: torch.Tensor,
        h_pool: torch.Tensor,
        pos_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select negative samples for contrastive learning.
        
        Args:
            h_anchor: Anchor embeddings [N, d]
            h_pool: Candidate pool embeddings [M, d]
            pos_indices: Positive indices in pool
            
        Returns:
            Negative indices for each anchor
        """
        N = h_anchor.shape[0]
        M = h_pool.shape[0]
        
        if self.hard_negative_mining:
            # Compute similarity with all candidates
            sim = h_anchor @ h_pool.T  # [N, M]
            
            # Exclude positive pairs
            mask = torch.ones(N, M, device=h_anchor.device)
            for i, pos_idx in enumerate(pos_indices):
                mask[i, pos_idx] = 0
            
            # Select hardest negatives (highest similarity)
            sim_masked = sim * mask
            neg_indices = torch.topk(sim_masked, self.num_negatives, dim=1).indices
        else:
            # Random sampling
            all_indices = torch.arange(M, device=h_anchor.device)
            neg_indices = torch.zeros(N, self.num_negatives, dtype=torch.long, device=h_anchor.device)
            
            for i in range(N):
                available = all_indices[all_indices != pos_indices[i]]
                if len(available) >= self.num_negatives:
                    neg_indices[i] = available[torch.randperm(len(available))[:self.num_negatives]]
                else:
                    neg_indices[i] = available[torch.randint(0, len(available), (self.num_negatives,))]
        
        return neg_indices
    
    def contrastive_loss(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute adaptive contrastive loss.
        
        Args:
            h1: Embeddings from view 1 [N, d]
            h2: Embeddings from view 2 [N, d]
            
        Returns:
            Contrastive loss value
        """
        N = h1.shape[0]
        device = h1.device
        
        # Normalize embeddings
        h1_norm = F.normalize(h1, dim=1)
        h2_norm = F.normalize(h2, dim=1)
        
        # Select positive pairs
        pos1, pos2 = self.select_positive_pairs(h1_norm, h2_norm)
        
        # Compute positive similarities
        pos_sim = torch.sum(h1_norm * h2_norm[pos2], dim=1) / self.temperature
        
        # Select negative pairs
        neg_indices = self.select_negative_pairs(h1_norm, h2_norm, pos2)
        
        # Compute negative similarities
        neg_sim = torch.zeros(N, device=device)
        for i in range(N):
            neg_embeddings = h2_norm[neg_indices[i]]
            neg_sim_i = torch.sum(h1_norm[i:i+1] * neg_embeddings, dim=1) / self.temperature
            neg_sim[i] = torch.logsumexp(neg_sim_i, dim=0)
        
        # Contrastive loss
        loss = -torch.mean(pos_sim - neg_sim)
        
        return loss
    
    def forward(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for adaptive contrastive learning.
        
        Args:
            h1: Embeddings from view 1 [N, d]
            h2: Embeddings from view 2 [N, d]
            
        Returns:
            Contrastive loss
        """
        return self.contrastive_loss(h1, h2)
