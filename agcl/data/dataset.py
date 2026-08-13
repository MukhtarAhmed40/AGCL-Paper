"""
Dataset loading and preprocessing for AGCL.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

from agcl.config import AGCLConfig
from agcl.data.graph_builder import DynamicGraphBuilder


class NetworkGraphDataset(Dataset):
    """
    Dataset for dynamic graph-based network intrusion detection.
    """
    
    def __init__(
        self,
        data_path: str,
        temporal_window: int = 60,
        transform: Optional[callable] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.temporal_window = temporal_window
        self.transform = transform
        
        self._load_data()
        self._build_temporal_graphs()
    
    def _load_data(self):
        """Load raw network traffic data."""
        # Load flows from CSV
        flows_file = os.path.join(self.data_path, "flows.csv")
        self.flows = pd.read_csv(flows_file)
        
        # Load labels
        labels_file = os.path.join(self.data_path, "labels.csv")
        self.labels = pd.read_csv(labels_file) if os.path.exists(labels_file) else None
    
    def _build_temporal_graphs(self):
        """Build temporal graph snapshots."""
        self.graph_builder = DynamicGraphBuilder(
            temporal_window=self.temporal_window,
        )
        
        self.graphs = self.graph_builder.build_snapshots(
            self.flows,
            self.labels,
        )
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get graph snapshot at index."""
        graph = self.graphs[idx]
        
        adj = torch.FloatTensor(graph["adjacency"])
        features = torch.FloatTensor(graph["features"])
        
        if self.transform:
            adj, features = self.transform(adj, features)
        
        return {
            "adj": adj,
            "features": features,
            "labels": graph.get("labels", torch.zeros(adj.shape[0])),
            "node_ids": graph.get("node_ids", torch.arange(adj.shape[0])),
            "window": graph.get("window", idx),
        }


def collate_graphs(batch: list) -> Dict[str, Any]:
    """
    Collate function for graph batches.
    
    Handles variable-sized graphs by padding to max size.
    """
    # Find max nodes in batch
    max_nodes = max(b["adj"].shape[0] for b in batch)
    
    # Pad all graphs to max_nodes
    batched_adj = []
    batched_features = []
    batched_labels = []
    batched_masks = []
    
    for b in batch:
        N = b["adj"].shape[0]
        
        # Pad adjacency matrix
        adj_padded = torch.zeros(max_nodes, max_nodes)
        adj_padded[:N, :N] = b["adj"]
        batched_adj.append(adj_padded)
        
        # Pad features
        feat_padded = torch.zeros(max_nodes, b["features"].shape[1])
        feat_padded[:N] = b["features"]
        batched_features.append(feat_padded)
        
        # Pad labels
        if "labels" in b:
            labels_padded = torch.full((max_nodes,), -1, dtype=torch.long)
            labels_padded[:N] = b["labels"]
            batched_labels.append(labels_padded)
        
        # Create mask
        mask = torch.zeros(max_nodes, dtype=torch.bool)
        mask[:N] = True
        batched_masks.append(mask)
    
    result = {
        "adj": torch.stack(batched_adj),
        "features": torch.stack(batched_features),
        "mask": torch.stack(batched_masks),
    }
    
    if batched_labels:
        result["labels"] = torch.stack(batched_labels)
    
    return result


def create_dataloaders(
    config: AGCLConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        config: AGCL configuration
        
    Returns:
        Train, validation, and test data loaders
    """
    data_path = config.data.get_dataset_path()
    
    # Create dataset
    dataset = NetworkGraphDataset(
        data_path=data_path,
        temporal_window=config.data.temporal_window,
    )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(total_size * config.data.train_ratio)
    val_size = int(total_size * config.data.val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.training.seed),
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_graphs,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_graphs,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_graphs,
    )
    
    return train_loader, val_loader, test_loader


def load_dataset(config: AGCLConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience function for loading dataset."""
    return create_dataloaders(config)
