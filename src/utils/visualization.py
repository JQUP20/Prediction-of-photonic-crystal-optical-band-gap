"""
Visualization utilities for photonic crystal band structures and transmission
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple
import sys
sys.path.append('..')


def plot_band_structure(
    k_points: np.ndarray,
    frequencies: np.ndarray,
    band_gap_info: Optional[dict] = None,
    save_path: Optional[str] = None,
    title: str = "Photonic Band Structure"
) -> plt.Figure:
    """
    Plot photonic band structure

    Args:
        k_points: Array of k-points (N, 3)
        frequencies: Array of frequencies (N, num_bands)
        band_gap_info: Dictionary with band gap information
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate k-path distance
    k_distances = np.zeros(len(k_points))
    for i in range(1, len(k_points)):
        k_distances[i] = k_distances[i - 1] + np.linalg.norm(k_points[i] - k_points[i - 1])

    # Plot each band
    num_bands = frequencies.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, num_bands))

    for band_idx in range(num_bands):
        ax.plot(k_distances, frequencies[:, band_idx],
               color=colors[band_idx], linewidth=2, label=f'Band {band_idx + 1}')

    # Highlight band gap if provided
    if band_gap_info is not None and band_gap_info['band_gap'] > 0:
        lower_edge = band_gap_info['lower_band_edge']
        upper_edge = band_gap_info['upper_band_edge']
        ax.axhspan(lower_edge, upper_edge, alpha=0.3, color='red',
                  label=f'Band Gap = {band_gap_info["band_gap"]:.4f}')

    ax.set_xlabel('Wave Vector k', fontsize=12)
    ax.set_ylabel('Frequency (ω a/2πc)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add high-symmetry point labels
    n_points = len(k_distances)
    labels = ['Γ', 'X', 'M', 'Γ']
    positions = [0, n_points // 3, 2 * n_points // 3, n_points - 1]

    for pos, label in zip(positions, labels):
        if pos < len(k_distances):
            ax.axvline(k_distances[pos], color='k', linestyle='--', alpha=0.5)
            ax.text(k_distances[pos], ax.get_ylim()[0], label,
                   ha='center', va='top', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_transmission(
    wavelengths: np.ndarray,
    transmission: np.ndarray,
    reflection: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Transmission Spectrum"
) -> plt.Figure:
    """
    Plot transmission and reflection spectra

    Args:
        wavelengths: Wavelength array
        transmission: Transmission coefficient array
        reflection: Reflection coefficient array (optional)
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(wavelengths, transmission, 'b-', linewidth=2, label='Transmission')

    if reflection is not None:
        ax.plot(wavelengths, reflection, 'r-', linewidth=2, label='Reflection')

        # Also plot absorption if both T and R are available
        absorption = 1.0 - transmission - reflection
        ax.plot(wavelengths, absorption, 'g--', linewidth=2, label='Absorption')

    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_crystal_structure(
    crystal,
    slice_axis: int = 2,
    slice_index: Optional[int] = None,
    save_path: Optional[str] = None,
    title: str = "Photonic Crystal Structure"
) -> plt.Figure:
    """
    Plot photonic crystal structure

    Args:
        crystal: PhotonicCrystal object
        slice_axis: Axis to slice (0, 1, or 2)
        slice_index: Index of slice (None for middle)
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    structure = crystal.structure

    # Get slice
    if slice_index is None:
        slice_index = structure.shape[slice_axis] // 2

    if slice_axis == 0:
        slice_data = structure[slice_index, :, :]
    elif slice_axis == 1:
        slice_data = structure[:, slice_index, :]
    else:
        slice_data = structure[:, :, slice_index]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Custom colormap
    colors = ['blue', 'cyan', 'yellow', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

    im = ax.imshow(slice_data, cmap=cmap, origin='lower', interpolation='nearest')
    ax.set_title(f'{title}\n{crystal.lattice_type}, a = {crystal.lattice_constant:.3f} μm',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('x (pixels)', fontsize=10)
    ax.set_ylabel('y (pixels)', fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Refractive Index', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_dos(
    frequencies: np.ndarray,
    dos: np.ndarray,
    band_gap_info: Optional[dict] = None,
    save_path: Optional[str] = None,
    title: str = "Density of States"
) -> plt.Figure:
    """
    Plot photonic density of states

    Args:
        frequencies: Frequency array
        dos: Density of states array
        band_gap_info: Band gap information dictionary
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(frequencies, dos, alpha=0.6, color='blue')
    ax.plot(frequencies, dos, 'b-', linewidth=2)

    # Highlight band gap
    if band_gap_info is not None and band_gap_info['band_gap'] > 0:
        lower_edge = band_gap_info['lower_band_edge']
        upper_edge = band_gap_info['upper_band_edge']
        ax.axvspan(lower_edge, upper_edge, alpha=0.3, color='red',
                  label=f'Band Gap = {band_gap_info["band_gap"]:.4f}')

    ax.set_xlabel('Frequency (ω a/2πc)', fontsize=12)
    ax.set_ylabel('Density of States', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if band_gap_info is not None:
        ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history

    Args:
        history: Dictionary with training history
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    axes[0].plot(history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot metrics (MAE)
    if 'train_mae' in history:
        axes[1].plot(history['train_mae'], 'b-', label='Training MAE', linewidth=2)
    if 'val_mae' in history:
        axes[1].plot(history['val_mae'], 'r-', label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Predicted vs Actual Band Gap"
) -> plt.Figure:
    """
    Plot predicted vs actual band gap values

    Args:
        y_true: True band gap values
        y_pred: Predicted band gap values
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Calculate R²
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    r_squared = correlation ** 2

    ax.set_xlabel('Actual Band Gap', fontsize=12)
    ax.set_ylabel('Predicted Band Gap', fontsize=12)
    ax.set_title(f'{title}\nR² = {r_squared:.4f}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
