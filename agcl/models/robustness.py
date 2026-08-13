"""
Robustness regularization for AGCL framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SpectralSmoothing(nn.Module):
    """
    Spectral smoothing regularization for graph representations.
    
    Preserves low-frequency graph structures and reduces noise.
    """
    
    def __init__(self, lambda_smooth: float = 0.01):
        super().__init__()
        self.lambda_smooth = lambda_smooth
    
    def compute_laplacian(self, adj: torch.Tensor) -> torch.Tensor:
        """Compute graph Laplacian."""
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.where(deg > 0, 1.0 / torch.sqrt(deg + 1e-8), torch.zeros_like(deg))
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = torch.eye(adj.shape[0], device=adj.device) - D_inv_sqrt @ adj @ D_inv_sqrt
        return L
    
    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply spectral smoothing regularization.
        
        Args:
            h: Node embeddings [N, d]
            adj: Adjacency matrix [N, N]
            
        Returns:
            Spectral smoothing loss
        """
        L = self.compute_laplacian(adj)
        smooth_loss = torch.trace(h.T @ L @ h) / h.shape[0]
        return self.lambda_smooth * smooth_loss


class AdversarialAugmentation(nn.Module):
    """
    Adversarial augmentation for robust representation learning.
    
    Generates perturbed graph views during training.
    """
    
    def __init__(
        self,
        perturbation_rate: float = 0.05,
        attack_iterations: int = 1,
        attack_lr: float = 0.01,
    ):
        super().__init__()
        self.perturbation_rate = perturbation_rate
        self.attack_iterations = attack_iterations
        self.attack_lr = attack_lr
    
    def perturb_features(
        self,
        features: torch.Tensor,
        perturbation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add adversarial perturbation to node features.
        
        Args:
            features: Original node features [N, d]
            perturbation: Optional pre-computed perturbation
            
        Returns:
            Perturbed features [N, d]
        """
        if perturbation is None:
            # Random perturbation
            perturbation = torch.randn_like(features) * self.perturbation_rate
        
        return features + perturbation
    
    def perturb_edges(
        self,
        adj: torch.Tensor,
        perturbation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add adversarial perturbation to graph edges.
        
        Args:
            adj: Original adjacency matrix [N, N]
            perturbation: Optional pre-computed perturbation
            
        Returns:
            Perturbed adjacency matrix [N, N]
        """
        N = adj.shape[0]
        
        if perturbation is None:
            # Random edge perturbation
            perturbation = torch.rand_like(adj) * self.perturbation_rate
        
        # Apply perturbation
        perturbed_adj = adj + perturbation
        perturbed_adj = torch.clamp(perturbed_adj, 0, 1)
        
        # Ensure symmetry
        perturbed_adj = (perturbed_adj + perturbed_adj.T) / 2
        
        return perturbed_adj
    
    def generate_adversarial_sample(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        features: torch.Tensor,
        target_model: nn.Module,
        use_optimizer: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate adversarial sample using gradient-based attack.
        
        Args:
            h: Current node embeddings [N, d]
            adj: Original adjacency matrix [N, N]
            features: Original node features [N, d]
            target_model: Model to attack
            use_optimizer: Whether to use optimization-based attack
            
        Returns:
            Perturbed embeddings, adjacency, and features
        """
        # Clone and detach
        h_adv = h.clone().detach()
        adj_adv = adj.clone().detach()
        feat_adv = features.clone().detach()
        
        if use_optimizer:
            # Set requires_grad for perturbations
            adj_adv.requires_grad = True
            feat_adv.requires_grad = True
            
            for _ in range(self.attack_iterations):
                # Forward pass through model
                h_new, _ = target_model(feat_adv, adj_adv)
                
                # Maximize contrastive loss (adversarial objective)
                loss = -torch.norm(h_new - h, p=2)
                loss.backward()
                
                # Update perturbations
                with torch.no_grad():
                    adj_grad = adj_adv.grad
                    feat_grad = feat_adv.grad
                    
                    adj_adv = adj_adv + self.attack_lr * adj_grad.sign()
                    feat_adv = feat_adv + self.attack_lr * feat_grad.sign()
                    
                    # Clip to valid range
                    adj_adv = torch.clamp(adj_adv, 0, 1)
                    
                    # Reset gradients
                    adj_adv.grad = None
                    feat_adv.grad = None
        
        return h_adv, adj_adv, feat_adv
    
    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        features: torch.Tensor,
        target_model: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate adversarial sample and compute adversarial loss.
        
        Returns:
            Adversarial loss, perturbed embeddings, adjacency, features
        """
        # Generate adversarial sample
        h_adv, adj_adv, feat_adv = self.generate_adversarial_sample(
            h, adj, features, target_model
        )
        
        # Compute adversarial loss (encourage invariance)
        h_new, _ = target_model(feat_adv, adj_adv)
        adv_loss = torch.norm(h_new - h, p=2).mean()
        
        return adv_loss, h_adv, adj_adv, feat_adv


class RobustnessRegularizer(nn.Module):
    """
    Combined robustness regularization for AGCL framework.
    """
    
    def __init__(
        self,
        spectral_lambda: float = 0.01,
        adversarial_weight: float = 0.1,
        perturbation_rate: float = 0.05,
        attack_iterations: int = 1,
    ):
        super().__init__()
        
        self.spectral = SpectralSmoothing(lambda_smooth=spectral_lambda)
        self.adversarial = AdversarialAugmentation(
            perturbation_rate=perturbation_rate,
            attack_iterations=attack_iterations,
        )
        self.adversarial_weight = adversarial_weight
    
    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        features: torch.Tensor,
        target_model: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both spectral and adversarial regularization losses.
        
        Returns:
            Total robustness loss, adversarial loss
        """
        # Spectral smoothing loss
        spectral_loss = self.spectral(h, adj)
        
        # Adversarial loss
        adv_loss, _, _, _ = self.adversarial(h, adj, features, target_model)
        
        # Combined loss
        total_loss = spectral_loss + self.adversarial_weight * adv_loss
        
        return total_loss, adv_loss
