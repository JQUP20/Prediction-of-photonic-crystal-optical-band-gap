"""
Graph Representation Converter for Photonic Crystals

Converts photonic crystal structures into graph representations
suitable for Graph Neural Networks (GNN).
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Optional
from .photonic_crystal_generator import PhotonicCrystal


class CrystalToGraphConverter:
    """
    Converts photonic crystal structures to graph representations
    """

    def __init__(self, neighbor_cutoff: float = 1.5, max_neighbors: int = 12):
        """
        Initialize the graph converter

        Args:
            neighbor_cutoff: Distance cutoff for neighbor connections (in lattice units)
            max_neighbors: Maximum number of neighbors per node
        """
        self.neighbor_cutoff = neighbor_cutoff
        self.max_neighbors = max_neighbors

    def _create_lattice_nodes(self, crystal: PhotonicCrystal) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create nodes from crystal lattice points

        Args:
            crystal: PhotonicCrystal object

        Returns:
            Tuple of (positions, features) arrays
        """
        structure = crystal.structure
        dims = structure.shape

        # Sample lattice points (reduce resolution for computational efficiency)
        if len(dims) == 3 and dims[2] > 1:  # 3D crystal
            step = max(1, dims[0] // 16)  # Sample ~16x16x16 points
            indices = np.mgrid[0:dims[0]:step, 0:dims[1]:step, 0:dims[2]:step]
        elif dims[2] == 1:  # 2D crystal
            step = max(1, dims[0] // 32)  # Sample ~32x32 points
            indices = np.mgrid[0:dims[0]:step, 0:dims[1]:step]
            indices = np.concatenate([indices, np.zeros((1,) + indices.shape[1:])], axis=0)
        else:  # 1D crystal
            step = max(1, dims[0] // 64)  # Sample ~64 points
            indices = np.mgrid[0:dims[0]:step]
            indices = np.stack([indices, np.zeros_like(indices), np.zeros_like(indices)])

        # Flatten indices
        positions = np.stack([idx.flatten() for idx in indices], axis=1)

        # Extract refractive index values at these positions
        if len(dims) == 3 and dims[2] > 1:
            features = structure[indices[0], indices[1], indices[2]].flatten()
        elif dims[2] == 1:
            features = structure[indices[0], indices[1], 0].flatten()
        else:
            features = structure[indices[0], 0, 0].flatten()

        return positions, features

    def _compute_edge_index(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute edge indices based on spatial proximity

        Args:
            positions: Node positions (N, 3)

        Returns:
            Tuple of (edge_index, edge_attr) arrays
        """
        num_nodes = positions.shape[0]
        edges = []
        edge_distances = []

        # Compute pairwise distances
        for i in range(num_nodes):
            distances = np.linalg.norm(positions - positions[i], axis=1)
            # Find neighbors within cutoff
            neighbors = np.where((distances > 0) & (distances < self.neighbor_cutoff))[0]

            # Limit to max_neighbors closest neighbors
            if len(neighbors) > self.max_neighbors:
                sorted_idx = np.argsort(distances[neighbors])
                neighbors = neighbors[sorted_idx[:self.max_neighbors]]

            # Add edges
            for j in neighbors:
                edges.append([i, j])
                edge_distances.append(distances[j])

        if len(edges) == 0:
            # No edges found, create self-loops
            edges = [[i, i] for i in range(num_nodes)]
            edge_distances = [0.0] * num_nodes

        edge_index = np.array(edges).T
        edge_attr = np.array(edge_distances).reshape(-1, 1)

        return edge_index, edge_attr

    def crystal_to_graph(
        self,
        crystal: PhotonicCrystal,
        target_band_gap: Optional[float] = None
    ) -> Data:
        """
        Convert a photonic crystal to a PyTorch Geometric Data object

        Args:
            crystal: PhotonicCrystal object
            target_band_gap: Target band gap value (for supervised learning)

        Returns:
            PyTorch Geometric Data object
        """
        # Create nodes
        positions, refractive_indices = self._create_lattice_nodes(crystal)

        # Normalize positions
        positions = positions / np.max(positions, axis=0, keepdims=True) if np.max(positions) > 0 else positions

        # Compute edges
        edge_index, edge_attr = self._compute_edge_index(positions)

        # Create node features: [refractive_index, x, y, z, filling_fraction, lattice_constant]
        node_features = np.column_stack([
            refractive_indices,
            positions,
            np.ones(len(positions)) * crystal.filling_fraction,
            np.ones(len(positions)) * crystal.lattice_constant
        ])

        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create graph data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(positions)
        )

        # Add target if provided
        if target_band_gap is not None:
            data.y = torch.tensor([target_band_gap], dtype=torch.float)

        # Add metadata
        data.lattice_type = crystal.lattice_type
        data.lattice_constant = crystal.lattice_constant

        return data

    def convert_dataset(
        self,
        crystals: List[PhotonicCrystal],
        target_band_gaps: Optional[List[float]] = None
    ) -> List[Data]:
        """
        Convert a list of photonic crystals to graph dataset

        Args:
            crystals: List of PhotonicCrystal objects
            target_band_gaps: List of target band gap values

        Returns:
            List of PyTorch Geometric Data objects
        """
        graph_dataset = []

        for idx, crystal in enumerate(crystals):
            target = target_band_gaps[idx] if target_band_gaps is not None else None
            graph_data = self.crystal_to_graph(crystal, target)
            graph_dataset.append(graph_data)

        return graph_dataset

    def batch_to_graph(self, batch_crystals: List[PhotonicCrystal]) -> List[Data]:
        """
        Convert a batch of crystals to graphs

        Args:
            batch_crystals: List of PhotonicCrystal objects

        Returns:
            List of PyTorch Geometric Data objects
        """
        return [self.crystal_to_graph(crystal) for crystal in batch_crystals]
