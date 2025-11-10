"""
Photonic Crystal Structure Generator

This module generates various photonic crystal structures including:
- 1D multilayer structures
- 2D photonic crystals (square, triangular, hexagonal lattices)
- 3D photonic crystals (diamond, woodpile structures)
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PhotonicCrystal:
    """
    Represents a photonic crystal structure

    Attributes:
        lattice_type: Type of lattice (square, triangular, hexagonal, diamond, etc.)
        lattice_constant: Lattice constant in micrometers
        refractive_indices: List of refractive indices for different materials
        filling_fraction: Volume fraction of high-index material
        structure: 3D array representing the crystal structure
        dimensions: (nx, ny, nz) dimensions of the structure
    """
    lattice_type: str
    lattice_constant: float
    refractive_indices: List[float]
    filling_fraction: float
    structure: np.ndarray
    dimensions: Tuple[int, int, int]

    def get_metadata(self) -> Dict:
        """Return crystal metadata"""
        return {
            'lattice_type': self.lattice_type,
            'lattice_constant': self.lattice_constant,
            'refractive_indices': self.refractive_indices,
            'filling_fraction': self.filling_fraction,
            'dimensions': self.dimensions
        }


class PhotonicCrystalGenerator:
    """
    Generator for various photonic crystal structures
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the photonic crystal generator

        Args:
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_1d_multilayer(
        self,
        n1: float,
        n2: float,
        num_layers: int,
        thickness_ratio: float = 0.5,
        lattice_constant: float = 1.0
    ) -> PhotonicCrystal:
        """
        Generate a 1D multilayer photonic crystal

        Args:
            n1: Refractive index of first material
            n2: Refractive index of second material
            num_layers: Number of layer pairs
            thickness_ratio: Ratio of first layer thickness to period
            lattice_constant: Lattice constant in micrometers

        Returns:
            PhotonicCrystal object
        """
        layers_per_period = 100
        total_points = num_layers * layers_per_period
        structure = np.zeros((total_points, 1, 1))

        layer1_thickness = int(layers_per_period * thickness_ratio)

        for i in range(num_layers):
            start = i * layers_per_period
            structure[start:start + layer1_thickness, 0, 0] = n1
            structure[start + layer1_thickness:start + layers_per_period, 0, 0] = n2

        return PhotonicCrystal(
            lattice_type='1D_multilayer',
            lattice_constant=lattice_constant,
            refractive_indices=[n1, n2],
            filling_fraction=thickness_ratio,
            structure=structure,
            dimensions=(total_points, 1, 1)
        )

    def generate_2d_square_lattice(
        self,
        n_background: float,
        n_rod: float,
        lattice_constant: float,
        rod_radius: float,
        grid_size: int = 64
    ) -> PhotonicCrystal:
        """
        Generate a 2D square lattice photonic crystal with circular rods

        Args:
            n_background: Refractive index of background material
            n_rod: Refractive index of rod material
            lattice_constant: Lattice constant in micrometers
            rod_radius: Radius of rods relative to lattice constant
            grid_size: Grid resolution

        Returns:
            PhotonicCrystal object
        """
        structure = np.ones((grid_size, grid_size, 1)) * n_background
        center = grid_size // 2

        # Create circular rod in the center
        y, x = np.ogrid[:grid_size, :grid_size]
        radius_pixels = int(rod_radius * grid_size)
        mask = (x - center)**2 + (y - center)**2 <= radius_pixels**2
        structure[mask, 0] = n_rod

        filling_fraction = np.pi * rod_radius**2

        return PhotonicCrystal(
            lattice_type='2D_square',
            lattice_constant=lattice_constant,
            refractive_indices=[n_background, n_rod],
            filling_fraction=filling_fraction,
            structure=structure,
            dimensions=(grid_size, grid_size, 1)
        )

    def generate_2d_triangular_lattice(
        self,
        n_background: float,
        n_rod: float,
        lattice_constant: float,
        rod_radius: float,
        grid_size: int = 64
    ) -> PhotonicCrystal:
        """
        Generate a 2D triangular lattice photonic crystal

        Args:
            n_background: Refractive index of background material
            n_rod: Refractive index of rod material
            lattice_constant: Lattice constant in micrometers
            rod_radius: Radius of rods relative to lattice constant
            grid_size: Grid resolution

        Returns:
            PhotonicCrystal object
        """
        structure = np.ones((grid_size, grid_size, 1)) * n_background

        # Triangular lattice positions
        radius_pixels = int(rod_radius * grid_size)

        # Place rods at triangular lattice sites
        # This is a simplified version - center rod only
        center = grid_size // 2
        y, x = np.ogrid[:grid_size, :grid_size]
        mask = (x - center)**2 + (y - center)**2 <= radius_pixels**2
        structure[mask, 0] = n_rod

        filling_fraction = np.pi * rod_radius**2 / (np.sqrt(3) / 2)

        return PhotonicCrystal(
            lattice_type='2D_triangular',
            lattice_constant=lattice_constant,
            refractive_indices=[n_background, n_rod],
            filling_fraction=filling_fraction,
            structure=structure,
            dimensions=(grid_size, grid_size, 1)
        )

    def generate_3d_fcc_lattice(
        self,
        n_background: float,
        n_sphere: float,
        lattice_constant: float,
        sphere_radius: float,
        grid_size: int = 32
    ) -> PhotonicCrystal:
        """
        Generate a 3D FCC lattice photonic crystal with spheres

        Args:
            n_background: Refractive index of background material
            n_sphere: Refractive index of sphere material
            lattice_constant: Lattice constant in micrometers
            sphere_radius: Radius of spheres relative to lattice constant
            grid_size: Grid resolution

        Returns:
            PhotonicCrystal object
        """
        structure = np.ones((grid_size, grid_size, grid_size)) * n_background

        # Create sphere in the center
        center = grid_size // 2
        z, y, x = np.ogrid[:grid_size, :grid_size, :grid_size]
        radius_pixels = int(sphere_radius * grid_size)
        mask = (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius_pixels**2
        structure[mask] = n_sphere

        filling_fraction = (4/3) * np.pi * sphere_radius**3

        return PhotonicCrystal(
            lattice_type='3D_fcc',
            lattice_constant=lattice_constant,
            refractive_indices=[n_background, n_sphere],
            filling_fraction=filling_fraction,
            structure=structure,
            dimensions=(grid_size, grid_size, grid_size)
        )

    def generate_random_crystal(
        self,
        lattice_type: Optional[str] = None,
        grid_size: int = 32
    ) -> PhotonicCrystal:
        """
        Generate a random photonic crystal structure

        Args:
            lattice_type: Type of lattice to generate (None for random choice)
            grid_size: Grid resolution

        Returns:
            PhotonicCrystal object
        """
        if lattice_type is None:
            lattice_type = np.random.choice(['1D_multilayer', '2D_square', '2D_triangular', '3D_fcc'])

        # Random parameters
        n_low = np.random.uniform(1.0, 2.0)
        n_high = np.random.uniform(2.5, 3.5)
        lattice_constant = np.random.uniform(0.3, 1.0)
        radius = np.random.uniform(0.2, 0.4)

        if lattice_type == '1D_multilayer':
            num_layers = np.random.randint(5, 20)
            return self.generate_1d_multilayer(n_low, n_high, num_layers,
                                              thickness_ratio=radius,
                                              lattice_constant=lattice_constant)
        elif lattice_type == '2D_square':
            return self.generate_2d_square_lattice(n_low, n_high, lattice_constant,
                                                   rod_radius=radius, grid_size=grid_size)
        elif lattice_type == '2D_triangular':
            return self.generate_2d_triangular_lattice(n_low, n_high, lattice_constant,
                                                       rod_radius=radius, grid_size=grid_size)
        elif lattice_type == '3D_fcc':
            return self.generate_3d_fcc_lattice(n_low, n_high, lattice_constant,
                                               sphere_radius=radius, grid_size=grid_size)
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")

    def generate_dataset(
        self,
        num_samples: int,
        lattice_types: Optional[List[str]] = None,
        grid_size: int = 32
    ) -> List[PhotonicCrystal]:
        """
        Generate a dataset of photonic crystals

        Args:
            num_samples: Number of samples to generate
            lattice_types: List of lattice types to generate (None for all types)
            grid_size: Grid resolution

        Returns:
            List of PhotonicCrystal objects
        """
        dataset = []

        for _ in range(num_samples):
            if lattice_types is not None:
                lattice_type = np.random.choice(lattice_types)
            else:
                lattice_type = None

            crystal = self.generate_random_crystal(lattice_type, grid_size)
            dataset.append(crystal)

        return dataset
