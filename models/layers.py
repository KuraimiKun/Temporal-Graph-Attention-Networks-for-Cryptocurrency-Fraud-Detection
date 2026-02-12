"""
Custom GNN Layers for Ethereum Fraud Detection

This module contains custom layers that extend PyTorch Geometric's
functionality for handling temporal graphs with edge features.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


class EdgeFeatureConv(MessagePassing):
    """
    Graph Convolution Layer that incorporates edge features.
    
    Standard GCN only uses node features and graph structure.
    This layer extends it to also consider edge features (e.g.,
    transaction value, gas) when aggregating neighbor information.
    
    Message = W_node * h_neighbor + W_edge * e_ij
    
    Args:
        in_channels: Input node feature dimension
        out_channels: Output node feature dimension
        edge_dim: Edge feature dimension
        aggr: Aggregation method ('add', 'mean', 'max')
        
    Example:
        >>> conv = EdgeFeatureConv(64, 32, edge_dim=5)
        >>> out = conv(x, edge_index, edge_attr)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 0,
        aggr: str = 'add',
        bias: bool = True,
    ):
        super().__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        # Node feature transformation
        self.lin_node = nn.Linear(in_channels, out_channels, bias=False)
        
        # Edge feature transformation (if edge features exist)
        if edge_dim > 0:
            self.lin_edge = nn.Linear(edge_dim, out_channels, bias=False)
        else:
            self.lin_edge = None
        
        # Root node transformation (self-loop)
        self.lin_root = nn.Linear(in_channels, out_channels, bias=bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_root.weight)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
        if self.lin_root.bias is not None:
            nn.init.zeros_(self.lin_root.bias)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of EdgeFeatureConv.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Transform node features
        x_j = self.lin_node(x)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x_j, edge_attr=edge_attr)
        
        # Add transformed root (self-loop equivalent)
        out = out + self.lin_root(x)
        
        return out
    
    def message(
        self,
        x_j: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Construct messages from neighbor nodes.
        
        Args:
            x_j: Features of source nodes [num_edges, out_channels]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Messages [num_edges, out_channels]
        """
        msg = x_j
        
        # Add edge feature contribution if available
        if edge_attr is not None and self.lin_edge is not None:
            msg = msg + self.lin_edge(edge_attr)
        
        return msg


class TemporalAttention(nn.Module):
    """
    Temporal Attention Layer for weighting historical graph snapshots.
    
    In temporal graphs, not all historical timesteps are equally important.
    This layer learns to attend to the most relevant past snapshots when
    making predictions about the current timestep.
    
    Attention(Q, K, V) where:
    - Q: Current timestep representation
    - K, V: Historical timestep representations
    
    Args:
        hidden_dim: Dimension of hidden representations
        num_heads: Number of attention heads
        dropout: Dropout rate
        
    Example:
        >>> attn = TemporalAttention(hidden_dim=64, num_heads=4)
        >>> # h_sequence: list of [num_nodes, hidden_dim] for each timestep
        >>> attended = attn(h_sequence)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        h_sequence: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply temporal attention over a sequence of graph representations.
        
        Args:
            h_sequence: Tensor of shape [seq_len, num_nodes, hidden_dim]
                        or list of [num_nodes, hidden_dim] tensors
            mask: Optional attention mask [seq_len, seq_len]
            
        Returns:
            Attended representation [num_nodes, hidden_dim]
        """
        # Convert list to tensor if needed
        if isinstance(h_sequence, list):
            h_sequence = torch.stack(h_sequence, dim=0)  # [T, N, D]
        
        T, N, D = h_sequence.shape
        
        # Use last timestep as query
        query = h_sequence[-1]  # [N, D]
        
        # All timesteps as keys and values
        keys = h_sequence  # [T, N, D]
        values = h_sequence  # [T, N, D]
        
        # Project and reshape for multi-head attention
        Q = self.W_q(query).view(N, self.num_heads, self.head_dim)  # [N, H, d]
        K = self.W_k(keys).view(T, N, self.num_heads, self.head_dim)  # [T, N, H, d]
        V = self.W_v(values).view(T, N, self.num_heads, self.head_dim)  # [T, N, H, d]
        
        # Compute attention scores: Q @ K^T / sqrt(d)
        # Q: [N, H, d] -> [N, H, 1, d]
        # K: [T, N, H, d] -> [N, H, T, d]
        Q = Q.unsqueeze(2)  # [N, H, 1, d]
        K = K.permute(1, 2, 0, 3)  # [N, H, T, d]
        V = V.permute(1, 2, 0, 3)  # [N, H, T, d]
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [N, H, 1, T]
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attended = torch.matmul(attn_probs, V)  # [N, H, 1, d]
        attended = attended.squeeze(2)  # [N, H, d]
        attended = attended.reshape(N, D)  # [N, D]
        
        # Output projection and residual
        output = self.W_o(attended)
        output = self.layer_norm(output + query)
        
        return output


class GRUCell(nn.Module):
    """
    Standard GRU Cell for temporal modeling.
    
    Used within GConvGRU to maintain temporal state across graph snapshots.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden state dimension
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Reset gate
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # Update gate  
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # Candidate hidden state
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)
    
    def forward(
        self,
        x: Tensor,
        h: Optional[Tensor] = None,
    ) -> Tensor:
        """
        One step of GRU.
        
        Args:
            x: Input [batch, input_dim]
            h: Previous hidden state [batch, hidden_dim]
            
        Returns:
            New hidden state [batch, hidden_dim]
        """
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        
        combined = torch.cat([x, h], dim=-1)
        
        r = torch.sigmoid(self.W_r(combined))  # Reset gate
        z = torch.sigmoid(self.W_z(combined))  # Update gate
        
        combined_reset = torch.cat([x, r * h], dim=-1)
        h_candidate = torch.tanh(self.W_h(combined_reset))
        
        h_new = (1 - z) * h + z * h_candidate
        
        return h_new


class GraphNorm(nn.Module):
    """
    Graph Normalization layer.
    
    Normalizes node features within each graph, which is more appropriate
    for graph data than batch normalization.
    
    Args:
        num_features: Number of features to normalize
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """
        Apply graph normalization.
        
        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch assignment vector (optional)
            
        Returns:
            Normalized features [num_nodes, num_features]
        """
        if batch is None:
            # Single graph - normalize across all nodes
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
        else:
            # Multiple graphs - normalize per graph
            # This is a simplified version
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
        
        x_norm = (x - mean) / (std + self.eps)
        return self.gamma * x_norm + self.beta
