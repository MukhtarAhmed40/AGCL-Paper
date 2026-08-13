"""
Configuration management for AGCL framework.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class DataConfig:
    """Configuration for data processing."""
    dataset: str = "CIC-DDoS2019"
    data_dir: str = "./data"
    temporal_window: int = 60
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    node_feature_dim: int = 128
    batch_size: int = 64
    num_workers: int = 4
    
    def get_dataset_path(self) -> str:
        return os.path.join(self.data_dir, self.dataset)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    hidden_dim: int = 256
    embedding_dim: int = 128
    num_heads: int = 8
    num_gat_layers: int = 2
    dropout_rate: float = 0.5
    activation: str = "relu"
    use_spectral_norm: bool = True


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning."""
    temperature: float = 0.07
    lambda_similarity: float = 0.6
    num_negatives: int = 256
    view_matching: str = "adaptive"  # "adaptive" or "fixed"
    hard_negative_mining: bool = True


@dataclass
class RobustnessConfig:
    """Configuration for robustness regularization."""
    spectral_smoothing: float = 0.01
    adversarial_weight: float = 0.1
    adversarial_perturbation: float = 0.05
    feature_corruption_rate: float = 0.1
    edge_perturbation_rate: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 0.01
    weight_decay: float = 1e-5
    num_epochs: int = 20
    warmup_epochs: int = 2
    lr_scheduler: str = "cosine"  # "cosine", "step", or "constant"
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    seed: int = 42


class AGCLConfig:
    """Main configuration container."""
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.contrastive = ContrastiveConfig()
        self.robustness = RobustnessConfig()
        self.training = TrainingConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "contrastive": self.contrastive.__dict__,
            "robustness": self.robustness.__dict__,
            "training": self.training.__dict__,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AGCLConfig":
        """Create config from dictionary."""
        config = cls()
        for section, values in config_dict.items():
            section_obj = getattr(config, section)
            for key, value in values.items():
                setattr(section_obj, key, value)
        return config
