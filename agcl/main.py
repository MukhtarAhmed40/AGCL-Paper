"""
Main entry point for AGCL training and evaluation.
"""

import os
import argparse
import torch
import numpy as np
import random
from typing import Dict, Any, Tuple
from datetime import datetime

from agcl.config import AGCLConfig
from agcl.data.dataset import load_dataset, create_dataloaders
from agcl.models.encoder import GATEncoder
from agcl.models.multi_view import MultiViewGraphBuilder
from agcl.models.contrastive import AdaptiveContrastiveLearning
from agcl.models.robustness import RobustnessRegularizer
from agcl.utils.metrics import compute_metrics
from agcl.utils.adversarial import evaluate_robustness


class AGCL:
    """
    Main AGCL framework implementation.
    """
    
    def __init__(self, config: AGCLConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup_seed()
        self._init_components()
    
    def _setup_seed(self):
        """Set random seed for reproducibility."""
        seed = self.config.training.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _init_components(self):
        """Initialize all framework components."""
        self.encoder = GATEncoder(
            in_dim=self.config.data.node_feature_dim,
            hidden_dim=self.config.model.hidden_dim,
            out_dim=self.config.model.embedding_dim,
            num_layers=self.config.model.num_gat_layers,
            num_heads=self.config.model.num_heads,
            dropout=self.config.model.dropout_rate,
            activation=self.config.model.activation,
            use_spectral_norm=self.config.model.use_spectral_norm,
        ).to(self.device)
        
        self.view_builder = MultiViewGraphBuilder(
            num_nodes=0,  # Set dynamically
            feature_dim=self.config.data.node_feature_dim,
            k_neighbors=10,
            similarity_threshold=0.5,
        )
        
        self.contrastive = AdaptiveContrastiveLearning(
            temperature=self.config.contrastive.temperature,
            lambda_similarity=self.config.contrastive.lambda_similarity,
            num_negatives=self.config.contrastive.num_negatives,
            hard_negative_mining=self.config.contrastive.hard_negative_mining,
        )
        
        self.robustness = RobustnessRegularizer(
            spectral_lambda=self.config.robustness.spectral_smoothing,
            adversarial_weight=self.config.robustness.adversarial_weight,
            perturbation_rate=self.config.robustness.adversarial_perturbation,
        )
        
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        self.scheduler = self._init_scheduler()
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        if self.config.training.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
            )
        elif self.config.training.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.num_epochs // 3,
                gamma=0.1,
            )
        else:
            return None
    
    def train_step(
        self,
        adj: torch.Tensor,
        features: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            adj: Adjacency matrix [N, N]
            features: Node features [N, d]
            
        Returns:
            Dictionary of loss values
        """
        self.encoder.train()
        
        # Build multi-view graphs
        views = self.view_builder(adj, features)
        
        # Encode both views
        h_topo, attn_topo = self.encoder(
            views["topology_features"].to(self.device),
            views["topology_adj"].to(self.device),
        )
        h_feat, attn_feat = self.encoder(
            views["feature_features"].to(self.device),
            views["feature_adj"].to(self.device),
        )
        
        # Compute contrastive loss
        con_loss = self.contrastive(h_topo, h_feat)
        
        # Compute robustness regularization
        reg_loss, adv_loss = self.robustness(
            h_topo,
            views["topology_adj"].to(self.device),
            views["topology_features"].to(self.device),
            self.encoder,
        )
        
        # Total loss
        total_loss = con_loss + reg_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.encoder.parameters(),
            self.config.training.gradient_clip_norm,
        )
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {
            "total_loss": total_loss.item(),
            "contrastive_loss": con_loss.item(),
            "regularization_loss": reg_loss.item(),
            "adversarial_loss": adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
        }
    
    def train(self, train_loader, val_loader) -> Dict[str, Any]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "best_val_loss": float("inf"),
        }
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.config.training.num_epochs):
            # Training
            epoch_losses = []
            for batch in train_loader:
                adj = batch["adj"].to(self.device)
                features = batch["features"].to(self.device)
                
                losses = self.train_step(adj, features)
                epoch_losses.append(losses["total_loss"])
            
            avg_train_loss = np.mean(epoch_losses)
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            val_loss = self.evaluate(val_loader)
            history["val_loss"].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history["best_val_loss"] = best_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            print(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
        
        return history
    
    def evaluate(self, loader) -> float:
        """
        Evaluate the model on a dataset.
        
        Args:
            loader: Data loader
            
        Returns:
            Average evaluation loss
        """
        self.encoder.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                adj = batch["adj"].to(self.device)
                features = batch["features"].to(self.device)
                
                views = self.view_builder(adj, features)
                
                h_topo, _ = self.encoder(
                    views["topology_features"].to(self.device),
                    views["topology_adj"].to(self.device),
                )
                h_feat, _ = self.encoder(
                    views["feature_features"].to(self.device),
                    views["feature_adj"].to(self.device),
                )
                
                con_loss = self.contrastive(h_topo, h_feat)
                reg_loss, _ = self.robustness(
                    h_topo,
                    views["topology_adj"].to(self.device),
                    views["topology_features"].to(self.device),
                    self.encoder,
                )
                
                total_loss += (con_loss + reg_loss).item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def predict(self, adj: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions for downstream tasks.
        
        Args:
            adj: Adjacency matrix [N, N]
            features: Node features [N, d]
            
        Returns:
            Node embeddings and attention weights
        """
        self.encoder.eval()
        
        with torch.no_grad():
            h, attn = self.encoder(features.to(self.device), adj.to(self.device))
        
        return h.cpu(), attn.cpu()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AGCL Framework")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--dataset", type=str, default="CIC-DDoS2019", help="Dataset name")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "all"])
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Load configuration
    config = AGCLConfig()
    if args.config:
        import json
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = AGCLConfig.from_dict(config_dict)
    
    config.data.dataset = args.dataset
    
    # Load dataset
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Initialize AGCL
    agcl = AGCL(config)
    
    if args.mode in ["train", "all"]:
        history = agcl.train(train_loader, val_loader)
        
        # Save model
        torch.save(agcl.encoder.state_dict(), "agcl_model.pth")
        print(f"Model saved to agcl_model.pth")
        
        if args.mode == "all":
            # Evaluate on test set
            test_loss = agcl.evaluate(test_loader)
            print(f"Test Loss: {test_loss:.4f}")
            
            # Compute metrics
            metrics = evaluate_robustness(agcl, test_loader, config)
            print("Test Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
    
    elif args.mode == "eval":
        # Load model
        agcl.encoder.load_state_dict(torch.load("agcl_model.pth"))
        
        test_loss = agcl.evaluate(test_loader)
        print(f"Test Loss: {test_loss:.4f}")
        
        metrics = evaluate_robustness(agcl, test_loader, config)
        print("Test Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
