"""
GNN Model for Photonic Crystal Band Gap Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from typing import Optional

from .graph_layers import CrystalGraphConv, EdgeConv, GlobalAttentionPooling


class GNNBandGapPredictor(nn.Module):
    """
    Graph Neural Network for predicting photonic crystal band gaps

    Architecture:
        1. Multiple CrystalGraphConv layers for feature extraction
        2. Global pooling for graph-level representation
        3. MLP for band gap prediction
    """

    def __init__(
        self,
        node_feature_dim: int = 6,
        edge_feature_dim: int = 1,
        hidden_dim: int = 128,
        num_conv_layers: int = 4,
        num_fc_layers: int = 3,
        dropout: float = 0.2,
        pooling: str = 'attention'
    ):
        """
        Initialize the GNN model

        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden layer dimension
            num_conv_layers: Number of graph convolution layers
            num_fc_layers: Number of fully connected layers
            dropout: Dropout rate
            pooling: Pooling method ('mean', 'max', 'add', 'attention')
        """
        super(GNNBandGapPredictor, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling

        # Input embedding
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)

        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_conv_layers):
            self.conv_layers.append(
                CrystalGraphConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    edge_dim=edge_feature_dim
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Global pooling
        if pooling == 'attention':
            self.pool = GlobalAttentionPooling(hidden_dim)
        else:
            self.pool = None  # Will use functional pooling

        # Fully connected layers for prediction
        fc_layers = []
        current_dim = hidden_dim

        for i in range(num_fc_layers):
            if i < num_fc_layers - 1:
                fc_layers.extend([
                    nn.Linear(current_dim, current_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(current_dim // 2)
                ])
                current_dim = current_dim // 2
            else:
                # Output layer
                fc_layers.append(nn.Linear(current_dim, 1))

        self.fc = nn.Sequential(*fc_layers)

        # Additional layers for auxiliary predictions
        self.predict_transmission = nn.Linear(hidden_dim, 100)  # Transmission spectrum
        self.predict_dispersion = nn.Linear(hidden_dim, 50)     # Dispersion relation points

    def forward(self, data):
        """
        Forward pass

        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features
                - edge_index: Edge indices
                - edge_attr: Edge attributes
                - batch: Batch assignment (for batched graphs)

        Returns:
            Dictionary containing:
                - band_gap: Predicted band gap
                - transmission: Predicted transmission spectrum (optional)
                - dispersion: Predicted dispersion points (optional)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Input embedding
        x = self.node_embedding(x)
        x = F.relu(x)

        # Graph convolution layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x_residual = x
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)

            # Residual connection
            if i > 0:
                x = x + x_residual

        # Global pooling
        if self.pooling == 'mean':
            x_graph = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x_graph = global_max_pool(x, batch)
        elif self.pooling == 'add':
            x_graph = global_add_pool(x, batch)
        elif self.pooling == 'attention':
            # For attention pooling, we need to handle batches differently
            unique_batches = torch.unique(batch)
            x_graph_list = []
            for b in unique_batches:
                mask = batch == b
                x_batch = x[mask]
                x_pooled = self.pool(x_batch, batch[mask])
                x_graph_list.append(x_pooled)
            x_graph = torch.cat(x_graph_list, dim=0)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Band gap prediction
        band_gap = self.fc(x_graph)

        # Auxiliary predictions
        transmission = torch.sigmoid(self.predict_transmission(x_graph))
        dispersion = self.predict_dispersion(x_graph)

        return {
            'band_gap': band_gap,
            'transmission': transmission,
            'dispersion': dispersion,
            'graph_embedding': x_graph
        }

    def predict_band_gap(self, data):
        """
        Predict only the band gap (for inference)

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Predicted band gap value
        """
        with torch.no_grad():
            outputs = self.forward(data)
            return outputs['band_gap']


class EnsembleGNNPredictor(nn.Module):
    """
    Ensemble of GNN models for improved prediction accuracy
    """

    def __init__(
        self,
        num_models: int = 5,
        node_feature_dim: int = 6,
        edge_feature_dim: int = 1,
        hidden_dim: int = 128,
        **kwargs
    ):
        """
        Initialize ensemble of GNN models

        Args:
            num_models: Number of models in ensemble
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden layer dimension
            **kwargs: Additional arguments for GNNBandGapPredictor
        """
        super(EnsembleGNNPredictor, self).__init__()

        self.num_models = num_models
        self.models = nn.ModuleList([
            GNNBandGapPredictor(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=hidden_dim,
                **kwargs
            )
            for _ in range(num_models)
        ])

    def forward(self, data):
        """
        Forward pass through ensemble

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Dictionary with averaged predictions and uncertainty estimates
        """
        predictions = []

        for model in self.models:
            output = model(data)
            predictions.append(output['band_gap'])

        # Stack predictions
        predictions = torch.stack(predictions, dim=0)

        # Compute mean and standard deviation
        mean_prediction = predictions.mean(dim=0)
        std_prediction = predictions.std(dim=0)

        return {
            'band_gap': mean_prediction,
            'uncertainty': std_prediction,
            'all_predictions': predictions
        }

    def predict_with_uncertainty(self, data):
        """
        Predict with uncertainty estimation

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Tuple of (mean prediction, uncertainty)
        """
        with torch.no_grad():
            outputs = self.forward(data)
            return outputs['band_gap'], outputs['uncertainty']
