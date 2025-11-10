"""
Band Structure Calculator using Plane Wave Expansion Method

This module implements a simplified DFT-like approach for calculating
photonic band structures using the plane wave expansion (PWE) method.
"""

import numpy as np
from scipy import linalg
from typing import List, Tuple, Optional
import sys
sys.path.append('..')
from data.photonic_crystal_generator import PhotonicCrystal


class BandStructureCalculator:
    """
    Calculate photonic band structure using plane wave expansion
    """

    def __init__(
        self,
        num_g_vectors: int = 289,
        num_k_points: int = 50,
        max_frequency: float = 1.0
    ):
        """
        Initialize the band structure calculator

        Args:
            num_g_vectors: Number of reciprocal lattice vectors (odd perfect square)
            num_k_points: Number of k-points to sample
            max_frequency: Maximum normalized frequency (c/a units)
        """
        self.num_g_vectors = num_g_vectors
        self.num_k_points = num_k_points
        self.max_frequency = max_frequency

    def _generate_reciprocal_vectors(self, crystal: PhotonicCrystal) -> np.ndarray:
        """
        Generate reciprocal lattice vectors

        Args:
            crystal: PhotonicCrystal object

        Returns:
            Array of reciprocal lattice vectors (num_g_vectors, 3)
        """
        # For simplicity, generate 2D reciprocal lattice vectors
        n_side = int(np.sqrt(self.num_g_vectors))
        n_range = np.arange(-(n_side // 2), n_side // 2 + 1)

        # Generate G vectors
        gx, gy = np.meshgrid(n_range, n_range)
        g_vectors = np.stack([
            gx.flatten(),
            gy.flatten(),
            np.zeros(self.num_g_vectors)
        ], axis=1)

        # Scale by 2π/a
        g_vectors = g_vectors * (2 * np.pi / crystal.lattice_constant)

        return g_vectors[:self.num_g_vectors]

    def _compute_fourier_coefficients(self, crystal: PhotonicCrystal) -> np.ndarray:
        """
        Compute Fourier coefficients of dielectric function

        Args:
            crystal: PhotonicCrystal object

        Returns:
            Fourier coefficients array
        """
        structure = crystal.structure

        # Compute epsilon(r) = n(r)^2
        epsilon = structure ** 2

        # Compute FFT
        if len(structure.shape) == 3 and structure.shape[2] > 1:
            epsilon_fft = np.fft.fftn(epsilon)
            epsilon_fft = np.fft.fftshift(epsilon_fft)
        else:
            # 2D case
            epsilon_2d = epsilon[:, :, 0]
            epsilon_fft = np.fft.fft2(epsilon_2d)
            epsilon_fft = np.fft.fftshift(epsilon_fft)

        # Extract coefficients
        center = np.array(epsilon_fft.shape) // 2
        n_side = int(np.sqrt(self.num_g_vectors))
        half = n_side // 2

        if len(epsilon_fft.shape) == 3:
            coeffs = epsilon_fft[
                center[0] - half:center[0] + half + 1,
                center[1] - half:center[1] + half + 1,
                center[2]
            ].flatten()[:self.num_g_vectors]
        else:
            coeffs = epsilon_fft[
                center[0] - half:center[0] + half + 1,
                center[1] - half:center[1] + half + 1
            ].flatten()[:self.num_g_vectors]

        # Normalize
        coeffs = coeffs / np.prod(structure.shape[:2])

        return coeffs

    def _construct_hamiltonian(
        self,
        k_point: np.ndarray,
        g_vectors: np.ndarray,
        epsilon_g: np.ndarray
    ) -> np.ndarray:
        """
        Construct the plane wave Hamiltonian matrix

        Args:
            k_point: Current k-point
            g_vectors: Reciprocal lattice vectors
            epsilon_g: Fourier coefficients of dielectric function

        Returns:
            Hamiltonian matrix
        """
        num_g = len(g_vectors)
        H = np.zeros((num_g, num_g), dtype=complex)

        for i in range(num_g):
            for j in range(num_g):
                k_plus_g_i = k_point + g_vectors[i]
                k_plus_g_j = k_point + g_vectors[j]

                # Kinetic energy term
                if i == j:
                    H[i, j] = np.dot(k_plus_g_i, k_plus_g_i)

                # Potential energy term (simplified)
                g_diff_idx = j - i
                if abs(g_diff_idx) < len(epsilon_g):
                    H[i, j] += epsilon_g[g_diff_idx] * 0.1

        return H

    def calculate_band_structure(
        self,
        crystal: PhotonicCrystal,
        k_path: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate photonic band structure

        Args:
            crystal: PhotonicCrystal object
            k_path: Path in k-space (num_k_points, 3). If None, use default path.

        Returns:
            Tuple of (k_points, frequencies)
                - k_points: Array of k-points (num_k_points, 3)
                - frequencies: Array of eigenfrequencies (num_k_points, num_bands)
        """
        # Generate reciprocal lattice vectors
        g_vectors = self._generate_reciprocal_vectors(crystal)

        # Compute Fourier coefficients
        epsilon_g = self._compute_fourier_coefficients(crystal)

        # Default k-path: Gamma -> X -> M -> Gamma
        if k_path is None:
            k_max = np.pi / crystal.lattice_constant
            k_path = self._generate_default_k_path(k_max)

        # Calculate eigenfrequencies at each k-point
        num_bands = min(10, self.num_g_vectors)  # Calculate first 10 bands
        frequencies = np.zeros((len(k_path), num_bands))

        for i, k in enumerate(k_path):
            # Construct Hamiltonian
            H = self._construct_hamiltonian(k, g_vectors, epsilon_g)

            # Solve eigenvalue problem
            eigenvalues = linalg.eigvalsh(H)

            # Take first num_bands eigenvalues and convert to frequencies
            frequencies[i] = np.sqrt(np.abs(eigenvalues[:num_bands]))

        return k_path, frequencies

    def _generate_default_k_path(self, k_max: float) -> np.ndarray:
        """
        Generate default k-path in Brillouin zone

        Args:
            k_max: Maximum k value

        Returns:
            Array of k-points
        """
        # Gamma -> X -> M -> Gamma
        n_points = self.num_k_points // 3

        # Gamma to X: (0,0,0) -> (k_max, 0, 0)
        gamma_to_x = np.linspace([0, 0, 0], [k_max, 0, 0], n_points)

        # X to M: (k_max, 0, 0) -> (k_max, k_max, 0)
        x_to_m = np.linspace([k_max, 0, 0], [k_max, k_max, 0], n_points)

        # M to Gamma: (k_max, k_max, 0) -> (0, 0, 0)
        m_to_gamma = np.linspace([k_max, k_max, 0], [0, 0, 0], n_points)

        k_path = np.vstack([gamma_to_x, x_to_m, m_to_gamma])

        return k_path

    def calculate_band_gap(
        self,
        crystal: PhotonicCrystal,
        k_path: Optional[np.ndarray] = None
    ) -> dict:
        """
        Calculate photonic band gap

        Args:
            crystal: PhotonicCrystal object
            k_path: Path in k-space

        Returns:
            Dictionary containing:
                - band_gap: Band gap value (normalized frequency)
                - gap_ratio: Gap-to-midgap ratio
                - lower_band_edge: Maximum frequency of lower band
                - upper_band_edge: Minimum frequency of upper band
        """
        k_points, frequencies = self.calculate_band_structure(crystal, k_path)

        # Find band gap
        band_gaps = []
        for band_idx in range(frequencies.shape[1] - 1):
            lower_band_max = np.max(frequencies[:, band_idx])
            upper_band_min = np.min(frequencies[:, band_idx + 1])

            if upper_band_min > lower_band_max:
                gap = upper_band_min - lower_band_max
                midgap = (upper_band_min + lower_band_max) / 2
                gap_ratio = gap / midgap if midgap > 0 else 0

                band_gaps.append({
                    'band_gap': gap,
                    'gap_ratio': gap_ratio,
                    'lower_band_edge': lower_band_max,
                    'upper_band_edge': upper_band_min,
                    'band_index': band_idx
                })

        if band_gaps:
            # Return the largest band gap
            largest_gap = max(band_gaps, key=lambda x: x['band_gap'])
            return largest_gap
        else:
            # No band gap found
            return {
                'band_gap': 0.0,
                'gap_ratio': 0.0,
                'lower_band_edge': 0.0,
                'upper_band_edge': 0.0,
                'band_index': -1
            }

    def calculate_dos(
        self,
        crystal: PhotonicCrystal,
        num_k_samples: int = 100,
        frequency_bins: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate photonic density of states (DOS)

        Args:
            crystal: PhotonicCrystal object
            num_k_samples: Number of k-points to sample
            frequency_bins: Number of frequency bins

        Returns:
            Tuple of (frequencies, dos)
        """
        # Sample k-points in Brillouin zone
        k_max = np.pi / crystal.lattice_constant
        k_samples = np.random.uniform(-k_max, k_max, (num_k_samples, 3))

        # Calculate band structure at sampled k-points
        _, frequencies = self.calculate_band_structure(crystal, k_samples)

        # Compute DOS histogram
        freq_range = (0, self.max_frequency)
        dos, freq_bins = np.histogram(
            frequencies.flatten(),
            bins=frequency_bins,
            range=freq_range,
            density=True
        )

        freq_centers = (freq_bins[:-1] + freq_bins[1:]) / 2

        return freq_centers, dos
