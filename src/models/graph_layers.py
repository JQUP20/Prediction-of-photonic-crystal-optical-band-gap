"""
Custom Graph Neural Network Layers for Photonic Crystal Band Gap Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class CrystalGraphConv(MessagePassing):
    """
    Crystal Graph Convolutional Layer

    Specialized graph convolution for photonic crystal structures
    that considers both node features and edge distances.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 1,
        aggr: str = 'add'
    ):
        """
        Initialize the Crystal Graph Convolution layer

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            edge_dim: Edge feature dimension
            aggr: Aggregation scheme ('add', 'mean', 'max')
        """
        super(CrystalGraphConv, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear transformations
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_dim, out_channels)
        self.lin_root = nn.Linear(in_channels, out_channels)

        # Attention mechanism
        self.att = nn.Linear(2 * out_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters"""
        self.lin_node.reset_parameters()
        self.lin_edge.reset_parameters()
        self.lin_root.reset_parameters()
        self.att.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass

        Args:
            x: Node features (N, in_channels)
            edge_index: Edge indices (2, E)
            edge_attr: Edge features (E, edge_dim)

        Returns:
            Updated node features (N, out_channels)
        """
        # Add self-loops
        edge_index, edge_attr = self._add_self_loops(edge_index, edge_attr, x.size(0))

        # Transform node features
        x_transformed = self.lin_node(x)

        # Propagate messages
        out = self.propagate(
            edge_index,
            x=x_transformed,
            edge_attr=edge_attr,
            original_x=x
        )

        # Add root node transformation
        out = out + self.lin_root(x)

        return out

    def message(self, x_j, edge_attr, x_i, original_x_i, original_x_j):
        """
        Construct messages from neighboring nodes

        Args:
            x_j: Transformed features of neighboring nodes
            edge_attr: Edge attributes
            x_i: Transformed features of target nodes
            original_x_i: Original features of target nodes
            original_x_j: Original features of neighboring nodes

        Returns:
            Messages to be aggregated
        """
        # Transform edge attributes
        edge_features = self.lin_edge(edge_attr)

        # Compute attention weights
        att_input = torch.cat([x_i, x_j], dim=-1)
        alpha = torch.sigmoid(self.att(att_input))

        # Weighted message
        message = alpha * (x_j + edge_features)

        return message

    def _add_self_loops(self, edge_index, edge_attr, num_nodes):
        """Add self-loops to the graph"""
        # Create self-loop indices
        loop_index = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)

        # Create self-loop attributes (distance = 0)
        loop_attr = torch.zeros((num_nodes, edge_attr.size(1)),
                               dtype=edge_attr.dtype, device=edge_attr.device)

        # Concatenate
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        return edge_index, edge_attr


class EdgeConv(MessagePassing):
    """
    Edge Convolution Layer for capturing local structure
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(EdgeConv, self).__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        """Forward pass"""
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        """Construct messages"""
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.mlp(edge_features)


class GlobalAttentionPooling(nn.Module):
    """
    Global attention pooling layer
    """

    def __init__(self, hidden_dim: int):
        super(GlobalAttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, batch):
        """
        Forward pass

        Args:
            x: Node features (N, hidden_dim)
            batch: Batch assignment vector (N,)

        Returns:
            Graph-level features
        """
        # Compute attention weights
        weights = self.attention(x)
        weights = F.softmax(weights, dim=0)

        # Weighted sum
        x_pooled = torch.sum(weights * x, dim=0, keepdim=True)

        return x_pooled
