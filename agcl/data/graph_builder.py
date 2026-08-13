"""
Dynamic graph construction for network traffic data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import networkx as nx
from sklearn.preprocessing import StandardScaler


class DynamicGraphBuilder:
    """
    Builds dynamic graph snapshots from network flow data.
    """
    
    def __init__(
        self,
        temporal_window: int = 60,
        feature_columns: Optional[List[str]] = None,
        node_id_columns: List[str] = ["src_ip", "dst_ip"],
    ):
        self.temporal_window = temporal_window
        self.feature_columns = feature_columns
        self.node_id_columns = node_id_columns
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def build_snapshots(
        self,
        flows: pd.DataFrame,
        labels: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build temporal graph snapshots from flow data.
        
        Args:
            flows: DataFrame containing network flows
            labels: Optional DataFrame containing node labels
            
        Returns:
            List of graph snapshot dictionaries
        """
        # Parse timestamps
        flows["timestamp"] = pd.to_datetime(flows["timestamp"])
        
        # Sort by time
        flows = flows.sort_values("timestamp")
        
        # Determine time windows
        min_time = flows["timestamp"].min()
        max_time = flows["timestamp"].max()
        total_seconds = (max_time - min_time).total_seconds()
        num_windows = int(np.ceil(total_seconds / self.temporal_window))
        
        snapshots = []
        current_time = min_time
        
        for window_idx in range(num_windows):
            window_end = current_time + timedelta(seconds=self.temporal_window)
            
            # Filter flows in this window
            window_flows = flows[
                (flows["timestamp"] >= current_time) & 
                (flows["timestamp"] < window_end)
            ]
            
            if not window_flows.empty:
                snapshot = self._build_snapshot(
                    window_flows,
                    labels,
                    window_idx,
                )
                snapshots.append(snapshot)
            
            current_time = window_end
        
        return snapshots
    
    def _build_snapshot(
        self,
        flows: pd.DataFrame,
        labels: Optional[pd.DataFrame],
        window_idx: int,
    ) -> Dict[str, Any]:
        """
        Build a single graph snapshot from flows in a time window.
        """
        # Create node mapping
        nodes = set()
        for col in self.node_id_columns:
            if col in flows.columns:
                nodes.update(flows[col].unique())
        
        # Convert to list
        node_list = list(nodes)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        num_nodes = len(node_list)
        
        # Build adjacency matrix
        adj = np.zeros((num_nodes, num_nodes))
        
        for _, row in flows.iterrows():
            src = row.get("src_ip")
            dst = row.get("dst_ip")
            if src in node_to_idx and dst in node_to_idx:
                src_idx = node_to_idx[src]
                dst_idx = node_to_idx[dst]
                adj[src_idx, dst_idx] = 1
                adj[dst_idx, src_idx] = 1
        
        # Build features
        features = self._build_node_features(flows, node_list, node_to_idx)
        
        # Build labels
        node_labels = self._build_node_labels(labels, node_list, node_to_idx)
        
        return {
            "adjacency": adj,
            "features": features,
            "labels": node_labels if node_labels is not None else None,
            "node_ids": node_list,
            "nodes": node_list,
            "window": window_idx,
            "num_flows": len(flows),
        }
    
    def _build_node_features(
        self,
        flows: pd.DataFrame,
        node_list: List[str],
        node_to_idx: Dict[str, int],
    ) -> np.ndarray:
        """
        Build node features from flow statistics.
        """
        num_nodes = len(node_list)
        
        # Aggregate flow statistics per node
        features_list = []
        
        for node in node_list:
            node_flows = flows[
                (flows["src_ip"] == node) | (flows["dst_ip"] == node)
            ]
            
            if len(node_flows) > 0:
                # Basic statistics
                features = [
                    len(node_flows),  # Flow count
                    node_flows["bytes"].sum() if "bytes" in node_flows else 0,
                    node_flows["bytes"].mean() if "bytes" in node_flows else 0,
                    node_flows["packets"].sum() if "packets" in node_flows else 0,
                    node_flows["packets"].mean() if "packets" in node_flows else 0,
                    node_flows["duration"].mean() if "duration" in node_flows else 0,
                    (node_flows["src_ip"] == node).sum(),  # Outgoing flows
                    (node_flows["dst_ip"] == node).sum(),  # Incoming flows
                ]
            else:
                features = [0] * 8
            
            features_list.append(features)
        
        features = np.array(features_list, dtype=np.float32)
        
        # Scale features
        if not self.is_fitted:
            self.scaler.fit(features)
            self.is_fitted = True
        
        features = self.scaler.transform(features)
        
        return features
    
    def _build_node_labels(
        self,
        labels: Optional[pd.DataFrame],
        node_list: List[str],
        node_to_idx: Dict[str, int],
    ) -> Optional[np.ndarray]:
        """Build node labels for supervised evaluation."""
        if labels is None:
            return None
        
        num_nodes = len(node_list)
        node_labels = np.zeros(num_nodes, dtype=np.float32)
        
        for idx, node in enumerate(node_list):
            label_row = labels[labels["node_id"] == node]
            if not label_row.empty:
                node_labels[idx] = label_row["label"].values[0]
        
        return node_labels
