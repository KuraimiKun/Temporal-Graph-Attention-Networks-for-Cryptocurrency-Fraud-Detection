"""
Temporal Graph Neural Network Architectures for Fraud Detection

This module implements the core TGN architectures:
1. GConvGRU - Graph Convolution + GRU for spatial-temporal learning
2. EvolveGCN - Evolving Graph Convolutional Networks

These models process sequences of graph snapshots and learn both:
- SPATIAL patterns: How fraud propagates through the transaction network
- TEMPORAL patterns: How fraudulent behavior evolves over time

Key Architecture Insight:
Traditional GNNs process a single static graph. Temporal GNNs maintain
a hidden state that evolves as new graph snapshots arrive, allowing
the model to learn patterns like "burst of high-value transactions
from a new wallet" that only emerge across time.
"""

from typing import List, Optional, Tuple, Dict, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# PyTorch Geometric imports
from torch_geometric.nn import GCNConv, ChebConv, GATConv
from torch_geometric.data import Data

# PyTorch Geometric Temporal imports (if available)
try:
    from torch_geometric_temporal.nn.recurrent import GConvGRU, EvolveGCNH, EvolveGCNO
    HAS_TEMPORAL = True
except ImportError:
    HAS_TEMPORAL = False
    print("Warning: torch_geometric_temporal not installed. Using custom implementations.")

from .layers import EdgeFeatureConv, TemporalAttention, GRUCell, GraphNorm


class TemporalGNN(nn.Module, ABC):
    """
    Abstract base class for Temporal Graph Neural Networks.
    
    All temporal GNN architectures should inherit from this class
    and implement the forward method for processing temporal graph sequences.
    
    The general flow is:
    1. Receive sequence of graph snapshots [G_1, G_2, ..., G_T]
    2. Process each snapshot while maintaining temporal hidden state
    3. Output node-level fraud probabilities for the final timestep
    
    Args:
        node_input_dim: Dimension of input node features
        edge_input_dim: Dimension of input edge features  
        hidden_channels: List of hidden dimensions for each layer
        num_classes: Number of output classes (2 for binary fraud detection)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_channels: List[int],
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dropout = dropout
        
        # To be defined by subclasses
        self.temporal_layers = None
        self.classifier = None
    
    @abstractmethod
    def forward(
        self,
        x_sequence: List[Tensor],
        edge_index_sequence: List[Tensor],
        edge_attr_sequence: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """
        Process a sequence of graph snapshots.
        
        Args:
            x_sequence: List of node feature tensors [num_nodes_t, node_dim]
            edge_index_sequence: List of edge index tensors [2, num_edges_t]
            edge_attr_sequence: Optional list of edge feature tensors
            
        Returns:
            Node classification logits for the final timestep [num_nodes_T, num_classes]
        """
        pass
    
    @abstractmethod
    def reset_hidden_state(self):
        """Reset the temporal hidden state between sequences."""
        pass
    
    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GConvGRUModel(TemporalGNN):
    """
    Graph Convolutional GRU for Temporal Graph Learning.
    
    This architecture combines:
    - Graph Convolution: Aggregates information from neighbor nodes
    - GRU Cells: Maintains temporal memory across timesteps
    
    At each timestep t:
    1. Apply graph convolution to get spatial representation
    2. Update GRU hidden state with the spatial representation
    3. The hidden state carries information from all previous timesteps
    
    Architecture Diagram:
    
        G_1 ──► GConv ──► GRU ──┐
                               │ h_1
        G_2 ──► GConv ──► GRU ◄┘──┐
                                  │ h_2
        G_3 ──► GConv ──► GRU ◄───┘──┐
                                     │ h_3
        ...                          │
                                     ▼
        G_T ──► GConv ──► GRU ──► Classifier ──► P(fraud)
    
    Args:
        node_input_dim: Input node feature dimension
        edge_input_dim: Input edge feature dimension (optional)
        hidden_channels: Hidden dimensions [64, 32] means 2 layers
        num_classes: Output classes (2 for fraud/legitimate)
        dropout: Dropout rate
        filter_size: Chebyshev polynomial order (spatial receptive field)
        use_edge_features: Whether to incorporate edge features
        
    Example:
        >>> model = GConvGRUModel(
        ...     node_input_dim=166,
        ...     edge_input_dim=4,
        ...     hidden_channels=[64, 32],
        ...     num_classes=2
        ... )
        >>> # x_seq: list of [N_t, 166] tensors for T timesteps
        >>> # edge_seq: list of [2, E_t] tensors
        >>> logits = model(x_seq, edge_seq)  # [N_T, 2]
    """
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int = 0,
        hidden_channels: List[int] = [64, 32],
        num_classes: int = 2,
        dropout: float = 0.3,
        filter_size: int = 3,
        use_edge_features: bool = True,
    ):
        super().__init__(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        self.filter_size = filter_size
        self.use_edge_features = use_edge_features and edge_input_dim > 0
        
        # Build temporal layers
        self._build_layers()
        
        # Classification head
        self._build_classifier()
        
        # Hidden states for each layer (initialized on first forward)
        self._hidden_states: List[Optional[Tensor]] = [None] * len(hidden_channels)
    
    def _build_layers(self):
        """Construct the GConvGRU layers."""
        layers = nn.ModuleList()
        norms = nn.ModuleList()
        
        # Input projection if using edge features
        if self.use_edge_features:
            self.edge_encoder = nn.Linear(self.edge_input_dim, self.hidden_channels[0])
        
        # Stack of GConvGRU layers
        in_dim = self.node_input_dim
        
        for i, out_dim in enumerate(self.hidden_channels):
            if HAS_TEMPORAL:
                # Use PyTorch Geometric Temporal's implementation
                layer = GConvGRU(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    K=self.filter_size,  # Chebyshev polynomial order
                )
            else:
                # Fallback to custom implementation
                layer = CustomGConvGRU(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    filter_size=self.filter_size,
                )
            
            layers.append(layer)
            norms.append(GraphNorm(out_dim))
            in_dim = out_dim
        
        self.temporal_layers = layers
        self.norms = norms
    
    def _build_classifier(self):
        """Build the classification head."""
        last_hidden = self.hidden_channels[-1]
        
        self.classifier = nn.Sequential(
            nn.Linear(last_hidden, last_hidden // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(last_hidden // 2, self.num_classes),
        )
    
    def reset_hidden_state(self):
        """Reset hidden states between sequences."""
        self._hidden_states = [None] * len(self.hidden_channels)
    
    def forward(
        self,
        x_sequence: List[Tensor],
        edge_index_sequence: List[Tensor],
        edge_attr_sequence: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """
        Process temporal graph sequence and predict fraud.
        
        Args:
            x_sequence: List of node features for each timestep
            edge_index_sequence: List of edge indices for each timestep
            edge_attr_sequence: Optional edge features
            
        Returns:
            Classification logits for nodes at final timestep
        """
        # Reset hidden state for new sequence
        self.reset_hidden_state()
        
        # Process each timestep
        for t, (x, edge_index) in enumerate(zip(x_sequence, edge_index_sequence)):
            edge_attr = edge_attr_sequence[t] if edge_attr_sequence else None
            
            # Forward through each layer
            h = x
            for i, (layer, norm) in enumerate(zip(self.temporal_layers, self.norms)):
                # Get previous hidden state for this layer
                prev_h = self._hidden_states[i]
                
                # Ensure hidden state has correct shape
                if prev_h is not None and prev_h.size(0) != h.size(0):
                    # Node count changed between timesteps - reset hidden
                    prev_h = None
                
                # GConvGRU forward
                if HAS_TEMPORAL:
                    h = layer(h, edge_index, H=prev_h)
                else:
                    h = layer(h, edge_index, prev_h)
                
                # Store hidden state for next timestep
                self._hidden_states[i] = h.detach()  # Detach for memory efficiency
                
                # Normalize and dropout (except last layer at last timestep)
                h = norm(h)
                if i < len(self.temporal_layers) - 1:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Classify nodes at final timestep
        logits = self.classifier(h)
        
        return logits
    
    def forward_single_snapshot(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Process a single graph snapshot (for streaming inference).
        
        Unlike forward(), this doesn't reset hidden state, allowing
        continuous processing of streaming graph data.
        """
        h = x
        
        for i, (layer, norm) in enumerate(zip(self.temporal_layers, self.norms)):
            prev_h = self._hidden_states[i]
            
            if prev_h is not None and prev_h.size(0) != h.size(0):
                prev_h = None
            
            if HAS_TEMPORAL:
                h = layer(h, edge_index, H=prev_h)
            else:
                h = layer(h, edge_index, prev_h)
            
            self._hidden_states[i] = h.detach()
            h = norm(h)
            
            if i < len(self.temporal_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return self.classifier(h)


class EvolveGCNModel(TemporalGNN):
    """
    Evolving Graph Convolutional Network (EvolveGCN).
    
    Unlike GConvGRU which maintains hidden states for nodes,
    EvolveGCN evolves the GCN WEIGHTS over time using an RNN.
    This captures how the entire graph's dynamics change.
    
    Two variants:
    - EvolveGCN-H: Uses GRU to evolve hidden representations
    - EvolveGCN-O: Uses LSTM to evolve GCN weight matrices
    
    Reference:
        Pareja et al., "EvolveGCN: Evolving Graph Convolutional 
        Networks for Dynamic Graphs", AAAI 2020
    
    Args:
        node_input_dim: Input node feature dimension
        hidden_channels: Hidden dimension (single value)
        num_classes: Number of output classes
        variant: 'H' for EvolveGCN-H, 'O' for EvolveGCN-O
        num_layers: Number of evolving GCN layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int = 0,
        hidden_channels: int = 64,
        num_classes: int = 2,
        variant: str = 'H',
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        # Convert single hidden_channels to list for parent class
        hidden_list = [hidden_channels] * num_layers
        
        super().__init__(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_channels=hidden_list,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        self.variant = variant
        self.num_layers = num_layers
        self._hidden_channels = hidden_channels
        
        self._build_layers()
        self._build_classifier()
    
    def _build_layers(self):
        """Build EvolveGCN layers."""
        if not HAS_TEMPORAL:
            raise ImportError(
                "EvolveGCN requires torch_geometric_temporal. "
                "Install with: pip install torch-geometric-temporal"
            )
        
        layers = nn.ModuleList()
        
        in_dim = self.node_input_dim
        for i in range(self.num_layers):
            if self.variant == 'H':
                layer = EvolveGCNH(
                    in_channels=in_dim,
                    out_channels=self._hidden_channels,
                )
            else:  # variant == 'O'
                layer = EvolveGCNO(
                    in_channels=in_dim,
                    out_channels=self._hidden_channels,
                )
            layers.append(layer)
            in_dim = self._hidden_channels
        
        self.temporal_layers = layers
    
    def _build_classifier(self):
        """Build classification head."""
        self.classifier = nn.Sequential(
            nn.Linear(self._hidden_channels, self._hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self._hidden_channels // 2, self.num_classes),
        )
    
    def reset_hidden_state(self):
        """Reset evolving weights."""
        # EvolveGCN handles this internally
        pass
    
    def forward(
        self,
        x_sequence: List[Tensor],
        edge_index_sequence: List[Tensor],
        edge_attr_sequence: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Process temporal sequence with EvolveGCN."""
        for t, (x, edge_index) in enumerate(zip(x_sequence, edge_index_sequence)):
            h = x
            for layer in self.temporal_layers:
                h = layer(h, edge_index)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Classify at final timestep
        return self.classifier(h)


class CustomGConvGRU(nn.Module):
    """
    Custom GConvGRU implementation (fallback when temporal library unavailable).
    
    Combines Chebyshev spectral graph convolution with GRU gating.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int = 3,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        
        # Graph convolutions for GRU gates
        self.conv_z = ChebConv(in_channels + out_channels, out_channels, K=filter_size)
        self.conv_r = ChebConv(in_channels + out_channels, out_channels, K=filter_size)
        self.conv_h = ChebConv(in_channels + out_channels, out_channels, K=filter_size)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        h: Optional[Tensor] = None,
    ) -> Tensor:
        """
        One timestep of GConvGRU.
        
        Args:
            x: Input node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            h: Previous hidden state [N, out_channels]
            
        Returns:
            New hidden state [N, out_channels]
        """
        if h is None:
            h = torch.zeros(x.size(0), self.out_channels, device=x.device)
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=-1)
        
        # Update gate
        z = torch.sigmoid(self.conv_z(combined, edge_index))
        
        # Reset gate
        r = torch.sigmoid(self.conv_r(combined, edge_index))
        
        # Candidate hidden state
        combined_reset = torch.cat([x, r * h], dim=-1)
        h_candidate = torch.tanh(self.conv_h(combined_reset, edge_index))
        
        # New hidden state
        h_new = (1 - z) * h + z * h_candidate
        
        return h_new


# =============================================================================
# Attention-Enhanced Temporal GNN (Advanced Architecture)
# =============================================================================

class TemporalGATModel(TemporalGNN):
    """
    Temporal Graph Attention Network.
    
    Combines:
    - GAT layers for learning attention over neighbors
    - Temporal attention for weighting historical snapshots
    - GRU for temporal state maintenance
    
    This is a more advanced architecture that can learn which
    neighbors and which historical timesteps are most relevant
    for fraud detection.
    """
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int = 0,
        hidden_channels: List[int] = [64, 32],
        num_classes: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        self.num_heads = num_heads
        self._build_layers()
        self._build_classifier()
        self._hidden_states = [None] * len(hidden_channels)
    
    def _build_layers(self):
        """Build GAT + GRU layers."""
        self.gat_layers = nn.ModuleList()
        self.gru_cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        in_dim = self.node_input_dim
        
        for out_dim in self.hidden_channels:
            # GAT layer with multiple heads
            gat = GATConv(
                in_channels=in_dim,
                out_channels=out_dim // self.num_heads,
                heads=self.num_heads,
                dropout=self.dropout,
                concat=True,  # Concatenate head outputs
            )
            
            # GRU for temporal modeling
            gru = GRUCell(out_dim, out_dim)
            
            self.gat_layers.append(gat)
            self.gru_cells.append(gru)
            self.norms.append(GraphNorm(out_dim))
            
            in_dim = out_dim
        
        # Temporal attention over sequence
        self.temporal_attention = TemporalAttention(
            hidden_dim=self.hidden_channels[-1],
            num_heads=self.num_heads,
        )
    
    def _build_classifier(self):
        """Build classification head."""
        last_dim = self.hidden_channels[-1]
        
        self.classifier = nn.Sequential(
            nn.Linear(last_dim, last_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(last_dim // 2, self.num_classes),
        )
    
    def reset_hidden_state(self):
        """Reset temporal hidden states."""
        self._hidden_states = [None] * len(self.hidden_channels)
        self._sequence_hidden = []  # Store for temporal attention
    
    def forward(
        self,
        x_sequence: List[Tensor],
        edge_index_sequence: List[Tensor],
        edge_attr_sequence: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Forward pass with attention mechanisms."""
        self.reset_hidden_state()
        sequence_outputs = []
        
        for t, (x, edge_index) in enumerate(zip(x_sequence, edge_index_sequence)):
            h = x
            
            for i, (gat, gru, norm) in enumerate(
                zip(self.gat_layers, self.gru_cells, self.norms)
            ):
                # Spatial attention via GAT
                h_spatial = gat(h, edge_index)
                h_spatial = F.elu(h_spatial)
                
                # Temporal update via GRU
                prev_h = self._hidden_states[i]
                if prev_h is not None and prev_h.size(0) != h_spatial.size(0):
                    prev_h = None
                
                h_temporal = gru(h_spatial, prev_h)
                self._hidden_states[i] = h_temporal.detach()
                
                h = norm(h_temporal)
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            sequence_outputs.append(h)
        
        # For variable-sized graphs across timesteps, we cannot use temporal attention
        # directly because node counts differ. Instead, use only the final timestep.
        # The GRU cells already capture temporal dependencies through hidden states.
        h_final = sequence_outputs[-1]
        
        return self.classifier(h_final)


# =============================================================================
# Model Factory
# =============================================================================

def create_model(
    architecture: str,
    node_input_dim: int,
    edge_input_dim: int = 0,
    hidden_channels: Union[int, List[int]] = [64, 32],
    num_classes: int = 2,
    dropout: float = 0.3,
    **kwargs
) -> TemporalGNN:
    """
    Factory function to create temporal GNN models.
    
    Args:
        architecture: One of 'gconv_gru', 'evolve_gcn', 'temporal_gat'
        node_input_dim: Input node feature dimension
        edge_input_dim: Input edge feature dimension
        hidden_channels: Hidden layer dimensions
        num_classes: Number of output classes
        dropout: Dropout rate
        **kwargs: Architecture-specific arguments
        
    Returns:
        Initialized TemporalGNN model
        
    Example:
        >>> model = create_model(
        ...     architecture='gconv_gru',
        ...     node_input_dim=166,
        ...     hidden_channels=[64, 32],
        ... )
    """
    # Normalize hidden_channels to list
    if isinstance(hidden_channels, int):
        hidden_channels = [hidden_channels]
    
    architecture = architecture.lower().replace('-', '_')
    
    if architecture == 'gconv_gru':
        return GConvGRUModel(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            dropout=dropout,
            filter_size=kwargs.get('filter_size', 3),
        )
    
    elif architecture == 'evolve_gcn':
        return EvolveGCNModel(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_channels=hidden_channels[0],  # EvolveGCN uses single hidden dim
            num_classes=num_classes,
            variant=kwargs.get('variant', 'H'),
            num_layers=len(hidden_channels),
            dropout=dropout,
        )
    
    elif architecture in ['temporal_gat', 'tgat']:
        return TemporalGATModel(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_heads=kwargs.get('num_heads', 4),
            dropout=dropout,
        )
    
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Choose from: 'gconv_gru', 'evolve_gcn', 'temporal_gat'"
        )


if __name__ == "__main__":
    # Quick model test
    print("Testing Temporal GNN Models")
    print("=" * 50)
    
    # Create dummy data
    num_timesteps = 5
    num_nodes = 100
    node_dim = 166
    
    x_sequence = [torch.randn(num_nodes, node_dim) for _ in range(num_timesteps)]
    edge_index_sequence = [
        torch.randint(0, num_nodes, (2, 500)) for _ in range(num_timesteps)
    ]
    
    # Test GConvGRU
    model = create_model(
        architecture='gconv_gru',
        node_input_dim=node_dim,
        hidden_channels=[64, 32],
    )
    
    print(f"\nGConvGRU Model:")
    print(f"  Parameters: {model.get_num_params():,}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(x_sequence, edge_index_sequence)
    
    print(f"  Input: {num_timesteps} snapshots × {num_nodes} nodes × {node_dim} features")
    print(f"  Output: {logits.shape}")
    
    # Test predictions
    probs = F.softmax(logits, dim=-1)
    print(f"  Fraud probability range: [{probs[:, 1].min():.3f}, {probs[:, 1].max():.3f}]")
