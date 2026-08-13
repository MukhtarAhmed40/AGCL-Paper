"""
Graph Attention Network encoder for AGCL framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network layer.
    
    Implements attention mechanism for neighborhood aggregation.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.5,
        negative_slope: float = 0.2,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        
        self.W = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)
        
        self.attn_param = nn.Parameter(torch.Tensor(1, num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attn_param)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        
        if use_spectral_norm:
            self.W = nn.utils.spectral_norm(self.W)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
    
    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of GAT layer.
        
        Args:
            h: Node features [N, in_dim]
            adj: Adjacency matrix [N, N]
            mask: Optional mask for attention [N, N]
            
        Returns:
            Updated node representations [N, out_dim]
            Attention weights [N, N]
        """
        N = h.shape[0]
        
        # Linear projection
        h_proj = self.W(h).view(N, self.num_heads, self.head_dim)  # [N, H, D']
        
        # Compute attention scores
        attn_scores = torch.zeros(N, self.num_heads, N, device=h.device)
        
        for i in range(N):
            for j in range(N):
                if adj[i, j] > 0:
                    concat = torch.cat([h_proj[i], h_proj[j]], dim=-1)  # [H, 2D']
                    score = (concat * self.attn_param).sum(dim=-1)  # [H]
                    attn_scores[i, :, j] = self.leaky_relu(score)
        
        # Mask attention scores
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax normalization
        attn_weights = F.softmax(attn_scores, dim=-1)  # [N, H, N]
        attn_weights = self.dropout(attn_weights)
        
        # Aggregate neighbors
        h_new = torch.einsum('nhj,jhd->nhd', attn_weights, h_proj)  # [N, H, D']
        
        # Concatenate heads
        h_new = h_new.contiguous().view(N, -1)  # [N, H*D']
        
        return h_new, attn_weights.mean(dim=1)  # Average over heads for return


class GATEncoder(nn.Module):
    """
    Multi-layer Graph Attention Network encoder.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.5,
        activation: str = "relu",
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = hidden_dim if i > 0 else hidden_dim
            layer_out = hidden_dim if i < num_layers - 1 else out_dim
            self.gat_layers.append(
                GraphAttentionLayer(
                    in_dim=layer_in,
                    out_dim=layer_out,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_spectral_norm=use_spectral_norm,
                )
            )
        
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout)
    
    def _get_activation(self, activation: str) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            return nn.Identity()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph to node embeddings.
        
        Args:
            x: Node features [N, in_dim]
            adj: Adjacency matrix [N, N]
            
        Returns:
            Node embeddings [N, out_dim]
            Attention weights from last layer
        """
        h = self.dropout(self.activation(self.input_proj(x)))
        
        attn_weights = None
        for i, gat_layer in enumerate(self.gat_layers):
            h, attn_weights = gat_layer(h, adj)
            if i < self.num_layers - 1:
                h = self.dropout(self.activation(h))
        
        return h, attn_weights
    
    def get_parameters(self):
        """Get all trainable parameters."""
        return list(self.parameters())
