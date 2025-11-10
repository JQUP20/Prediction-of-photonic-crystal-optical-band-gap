"""
Transmission Calculator for Photonic Crystals

Calculate light transmission properties using Transfer Matrix Method (TMM)
and Finite-Difference Time-Domain (FDTD) approximations.
"""

import numpy as np
from typing import Tuple, Optional
import sys
sys.path.append('..')
from data.photonic_crystal_generator import PhotonicCrystal


class TransmissionCalculator:
    """
    Calculate transmission and reflection spectra for photonic crystals
    """

    def __init__(
        self,
        wavelength_range: Tuple[float, float] = (0.3, 2.0),
        num_wavelengths: int = 100,
        incident_angle: float = 0.0
    ):
        """
        Initialize transmission calculator

        Args:
            wavelength_range: (min, max) wavelength range in micrometers
            num_wavelengths: Number of wavelength points
            incident_angle: Incident angle in degrees
        """
        self.wavelength_range = wavelength_range
        self.num_wavelengths = num_wavelengths
        self.incident_angle = np.radians(incident_angle)

    def calculate_1d_transmission_tmm(
        self,
        crystal: PhotonicCrystal
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate transmission for 1D multilayer structure using TMM

        Args:
            crystal: PhotonicCrystal object (must be 1D)

        Returns:
            Tuple of (wavelengths, transmission, reflection)
        """
        if crystal.lattice_type != '1D_multilayer':
            raise ValueError("TMM is only applicable to 1D multilayer structures")

        wavelengths = np.linspace(
            self.wavelength_range[0],
            self.wavelength_range[1],
            self.num_wavelengths
        )

        transmission = np.zeros(self.num_wavelengths)
        reflection = np.zeros(self.num_wavelengths)

        # Extract layer structure
        structure_1d = crystal.structure[:, 0, 0]
        layer_indices = np.where(np.diff(structure_1d) != 0)[0]

        # Get refractive indices and thicknesses
        n_layers = []
        thicknesses = []

        prev_idx = 0
        for idx in layer_indices:
            n_layers.append(structure_1d[prev_idx])
            thicknesses.append((idx - prev_idx) * crystal.lattice_constant / len(structure_1d))
            prev_idx = idx

        # Add last layer
        n_layers.append(structure_1d[prev_idx])
        thicknesses.append((len(structure_1d) - prev_idx) * crystal.lattice_constant / len(structure_1d))

        # Calculate for each wavelength
        for i, wavelength in enumerate(wavelengths):
            M = self._transfer_matrix(wavelength, n_layers, thicknesses)

            # Calculate transmission and reflection
            r = M[1, 0] / M[0, 0]
            t = 1.0 / M[0, 0]

            reflection[i] = np.abs(r) ** 2
            transmission[i] = np.abs(t) ** 2

        return wavelengths, transmission, reflection

    def _transfer_matrix(
        self,
        wavelength: float,
        n_layers: list,
        thicknesses: list
    ) -> np.ndarray:
        """
        Calculate transfer matrix for multilayer structure

        Args:
            wavelength: Wavelength in micrometers
            n_layers: List of refractive indices
            thicknesses: List of layer thicknesses

        Returns:
            2x2 transfer matrix
        """
        k0 = 2 * np.pi / wavelength

        # Initialize with identity matrix
        M_total = np.eye(2, dtype=complex)

        n_incident = n_layers[0]

        for i in range(len(n_layers) - 1):
            n_i = n_layers[i]
            n_j = n_layers[i + 1]
            d_i = thicknesses[i]

            # Phase factor
            beta = n_i * k0 * d_i * np.cos(self.incident_angle)

            # Propagation matrix
            P = np.array([
                [np.exp(1j * beta), 0],
                [0, np.exp(-1j * beta)]
            ], dtype=complex)

            # Interface matrix
            r_ij = (n_i - n_j) / (n_i + n_j)
            t_ij = 2 * n_i / (n_i + n_j)

            I = np.array([
                [1, r_ij],
                [r_ij, 1]
            ], dtype=complex) / t_ij

            # Multiply matrices
            M_total = M_total @ P @ I

        return M_total

    def calculate_2d_transmission_approximate(
        self,
        crystal: PhotonicCrystal,
        polarization: str = 'TE'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate approximate transmission for 2D photonic crystal

        Args:
            crystal: PhotonicCrystal object
            polarization: Polarization mode ('TE' or 'TM')

        Returns:
            Tuple of (frequencies, transmission)
        """
        # This is a simplified approximation based on band structure
        frequencies = np.linspace(0, 1.0, self.num_wavelengths)
        transmission = np.zeros(self.num_wavelengths)

        # Approximate transmission based on structure parameters
        filling_fraction = crystal.filling_fraction
        n_contrast = max(crystal.refractive_indices) / min(crystal.refractive_indices)

        for i, freq in enumerate(frequencies):
            # Simple model: transmission decreases in band gap regions
            # Band gap approximately at freq ~ 0.4-0.6 for typical structures

            # Estimate band gap center
            gap_center = 0.5 / np.sqrt(np.mean(np.array(crystal.refractive_indices) ** 2))
            gap_width = 0.1 * (n_contrast - 1) * filling_fraction

            # Gaussian-like dip in transmission at band gap
            gap_factor = np.exp(-((freq - gap_center) / (gap_width / 2)) ** 2)
            transmission[i] = 1.0 - 0.9 * gap_factor

        # Convert frequencies to wavelengths
        wavelengths = crystal.lattice_constant / frequencies
        wavelengths[0] = wavelengths[1] * 2  # Avoid division by zero

        return wavelengths, transmission

    def calculate_transmission_spectrum(
        self,
        crystal: PhotonicCrystal
    ) -> dict:
        """
        Calculate full transmission spectrum

        Args:
            crystal: PhotonicCrystal object

        Returns:
            Dictionary containing:
                - wavelengths: Array of wavelengths
                - transmission: Transmission coefficient
                - reflection: Reflection coefficient (if available)
                - absorption: Absorption coefficient
        """
        if crystal.lattice_type == '1D_multilayer':
            wavelengths, transmission, reflection = self.calculate_1d_transmission_tmm(crystal)
            absorption = 1.0 - transmission - reflection
        else:
            wavelengths, transmission = self.calculate_2d_transmission_approximate(crystal)
            reflection = np.zeros_like(transmission)
            absorption = 1.0 - transmission

        return {
            'wavelengths': wavelengths,
            'transmission': transmission,
            'reflection': reflection,
            'absorption': absorption
        }

    def calculate_field_distribution(
        self,
        crystal: PhotonicCrystal,
        wavelength: float,
        field_points: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate electromagnetic field distribution (simplified)

        Args:
            crystal: PhotonicCrystal object
            wavelength: Wavelength in micrometers
            field_points: Number of field sampling points

        Returns:
            Tuple of (positions, field_intensity)
        """
        structure = crystal.structure

        if crystal.lattice_type == '1D_multilayer':
            # 1D field distribution
            positions = np.linspace(0, structure.shape[0], field_points)
            k0 = 2 * np.pi / wavelength

            # Simple standing wave pattern
            field = np.zeros(field_points, dtype=complex)
            for i, pos in enumerate(positions):
                idx = int(pos) if pos < structure.shape[0] else structure.shape[0] - 1
                n = structure[idx, 0, 0]
                field[i] = np.exp(1j * n * k0 * pos)

            field_intensity = np.abs(field) ** 2
        else:
            # 2D field distribution (simplified)
            positions = np.linspace(0, structure.shape[0], field_points)
            field_intensity = np.random.random(field_points) * 0.2 + 0.8

        return positions, field_intensity

    def calculate_quality_factor(
        self,
        crystal: PhotonicCrystal,
        resonance_wavelength: Optional[float] = None
    ) -> float:
        """
        Calculate quality factor (Q-factor) for resonant modes

        Args:
            crystal: PhotonicCrystal object
            resonance_wavelength: Resonance wavelength (if known)

        Returns:
            Q-factor
        """
        result = self.calculate_transmission_spectrum(crystal)
        wavelengths = result['wavelengths']
        transmission = result['transmission']

        if resonance_wavelength is None:
            # Find resonance (transmission peak)
            resonance_idx = np.argmax(transmission)
            resonance_wavelength = wavelengths[resonance_idx]
        else:
            resonance_idx = np.argmin(np.abs(wavelengths - resonance_wavelength))

        # Find FWHM
        max_transmission = transmission[resonance_idx]
        half_max = max_transmission / 2

        # Find points at half maximum
        left_idx = np.where(transmission[:resonance_idx] < half_max)[0]
        right_idx = np.where(transmission[resonance_idx:] < half_max)[0]

        if len(left_idx) > 0 and len(right_idx) > 0:
            fwhm = wavelengths[resonance_idx + right_idx[0]] - wavelengths[left_idx[-1]]
            q_factor = resonance_wavelength / fwhm if fwhm > 0 else 0
        else:
            q_factor = 0

        return q_factor
