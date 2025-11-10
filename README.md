# Photonic Crystal Optical Band Gap Prediction

This project combines Graph Neural Networks (GNN) with DFT-inspired calculations to predict optical band gaps and light transmission properties of photonic crystals.

## 🎯 Project Overview

Photonic crystals are periodic optical nanostructures that affect the motion of photons in a similar way that ionic lattices affect electrons in solids. This project provides:

- **Data Generation**: Automated generation of various photonic crystal structures (1D, 2D, 3D)
- **DFT-like Calculations**: Band structure and transmission calculations using plane wave expansion
- **GNN Model**: Deep learning model for rapid band gap prediction
- **Visualization**: Comprehensive tools for analyzing results

## 🏗️ Project Structure

```
Prediction-of-photonic-crystal-optical-band-gap/
├── src/
│   ├── data/
│   │   ├── photonic_crystal_generator.py    # Crystal structure generation
│   │   └── graph_converter.py               # Convert crystals to graphs
│   ├── models/
│   │   ├── gnn_bandgap.py                   # GNN model architecture
│   │   └── graph_layers.py                   # Custom graph layers
│   ├── dft/
│   │   ├── band_structure_calculator.py      # Band structure calculations
│   │   └── transmission_calculator.py        # Transmission calculations
│   └── utils/
│       ├── visualization.py                  # Plotting utilities
│       └── metrics.py                        # Evaluation metrics
├── notebooks/
│   └── examples/                             # Jupyter notebooks
├── data/
│   ├── raw/                                  # Raw data
│   └── processed/                            # Processed data
├── results/                                  # Training results
├── configs/                                  # Configuration files
├── train.py                                  # Training script
├── evaluate.py                               # Evaluation script
├── requirements.txt                          # Dependencies
└── README.md                                 # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Prediction-of-photonic-crystal-optical-band-gap
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage

### Training a Model

Train a GNN model to predict photonic band gaps:

```bash
python train.py \
    --num_samples 1000 \
    --grid_size 32 \
    --hidden_dim 128 \
    --num_conv_layers 4 \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --output_dir results/experiment1
```

Key arguments:
- `--num_samples`: Number of training samples to generate
- `--grid_size`: Resolution of crystal structures
- `--hidden_dim`: Hidden dimension of GNN
- `--num_conv_layers`: Number of graph convolution layers
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--output_dir`: Directory to save results

### Evaluating a Model

Evaluate a trained model on test data:

```bash
python evaluate.py \
    --model_path results/experiment1/best_model.pt \
    --num_test_samples 200 \
    --output_dir evaluation_results
```

### Using Python API

```python
from src.data.photonic_crystal_generator import PhotonicCrystalGenerator
from src.data.graph_converter import CrystalToGraphConverter
from src.models.gnn_bandgap import GNNBandGapPredictor
from src.dft.band_structure_calculator import BandStructureCalculator

# Generate a photonic crystal
generator = PhotonicCrystalGenerator()
crystal = generator.generate_2d_square_lattice(
    n_background=1.0,
    n_rod=3.4,
    lattice_constant=0.5,
    rod_radius=0.3,
    grid_size=64
)

# Calculate band structure
calculator = BandStructureCalculator()
k_points, frequencies = calculator.calculate_band_structure(crystal)
band_gap_info = calculator.calculate_band_gap(crystal)

print(f"Band gap: {band_gap_info['band_gap']:.4f}")

# Convert to graph and predict with GNN
converter = CrystalToGraphConverter()
graph = converter.crystal_to_graph(crystal)

model = GNNBandGapPredictor()
# Load trained weights...
prediction = model.predict_band_gap(graph)
```

## 🔬 Photonic Crystal Types

### 1D Multilayer Structures
- Alternating layers of different refractive indices
- Used in Bragg reflectors and distributed feedback lasers

### 2D Photonic Crystals
- **Square Lattice**: Circular rods in square arrangement
- **Triangular Lattice**: Hexagonal arrangement of rods
- Common in photonic crystal fibers

### 3D Photonic Crystals
- **FCC Lattice**: Face-centered cubic arrangement of spheres
- **Diamond Structure**: Complete photonic band gaps
- Applications in optical computing

## 📈 Model Architecture

### GNN Architecture

```
Input (Graph Representation)
    ↓
Node Embedding Layer
    ↓
Crystal Graph Convolution Layers (×4)
    ├─ Message Passing
    ├─ Edge Feature Integration
    └─ Residual Connections
    ↓
Global Attention Pooling
    ↓
Fully Connected Layers (×3)
    ↓
Output: Band Gap Prediction
```

### Key Features

1. **Custom Graph Convolution**: Specialized for photonic crystals
   - Edge distance encoding
   - Attention mechanism
   - Material property integration

2. **Multi-task Learning**:
   - Primary: Band gap prediction
   - Auxiliary: Transmission spectrum
   - Auxiliary: Dispersion relation

3. **Uncertainty Estimation**: Ensemble model for confidence intervals

## 🎨 Visualization

The project includes comprehensive visualization tools:

- **Band Structure Plots**: Dispersion relation with band gap highlighting
- **Transmission Spectra**: T, R, A vs wavelength
- **Crystal Structure**: 2D/3D refractive index distribution
- **Density of States**: Photonic DOS
- **Training Curves**: Loss and metrics over epochs
- **Prediction Analysis**: Predicted vs actual scatter plots

## 📐 DFT Calculations

### Band Structure Calculator

Uses plane wave expansion method (PWE):

1. Fourier decomposition of dielectric function
2. Construction of eigenvalue problem in reciprocal space
3. Solution of Maxwell's equations
4. Extraction of photonic bands and gaps

### Transmission Calculator

Implements:

- **Transfer Matrix Method (TMM)**: For 1D multilayer structures
- **Approximate Methods**: For 2D/3D structures
- Field distribution calculations
- Quality factor (Q-factor) estimation

## 📊 Performance Metrics

The model is evaluated using:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

Typical performance on test set:
- MAE: ~0.02 (normalized frequency units)
- R²: >0.95

## 🔧 Advanced Usage

### Custom Crystal Structures

```python
import numpy as np
from src.data.photonic_crystal_generator import PhotonicCrystal

# Define custom structure
custom_structure = np.ones((64, 64, 1)) * 1.0  # Background
# Add features manually...

crystal = PhotonicCrystal(
    lattice_type='custom',
    lattice_constant=0.5,
    refractive_indices=[1.0, 3.0],
    filling_fraction=0.3,
    structure=custom_structure,
    dimensions=(64, 64, 1)
)
```

### Hyperparameter Tuning

```bash
# Train with different configurations
for hidden_dim in 64 128 256; do
    python train.py \
        --hidden_dim $hidden_dim \
        --output_dir results/hidden_${hidden_dim}
done
```

### Ensemble Prediction

```python
from src.models.gnn_bandgap import EnsembleGNNPredictor

ensemble = EnsembleGNNPredictor(num_models=5)
mean_pred, uncertainty = ensemble.predict_with_uncertainty(graph)
```

## 📚 Scientific Background

### Photonic Band Gaps

A photonic band gap is a range of wavelengths for which light cannot propagate through the crystal. Similar to electronic band gaps in semiconductors, photonic band gaps arise from:

1. **Periodic Modulation**: Refractive index varies periodically
2. **Bragg Scattering**: Constructive interference of scattered waves
3. **Forbidden Frequencies**: Complete reflection at certain wavelengths

### Applications

- **Optical Filters**: Wavelength-selective devices
- **Lasers**: Low-threshold lasing in photonic crystal cavities
- **Waveguides**: Light confinement and routing
- **Sensors**: High-sensitivity optical sensors
- **Solar Cells**: Enhanced light trapping

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📖 References

1. Joannopoulos, J. D., et al. "Photonic Crystals: Molding the Flow of Light" (2008)
2. Sakoda, K. "Optical Properties of Photonic Crystals" (2005)
3. Johnson, S. G., & Joannopoulos, J. D. "Block-iterative frequency-domain methods for Maxwell's equations in a planewave basis" (2001)

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

## 🙏 Acknowledgments

- PyTorch Geometric team for the graph neural network framework
- SciPy developers for numerical algorithms
- Matplotlib for visualization tools

---

**Note**: This is a research/educational project. For production applications, consider using specialized photonic simulation software like MEEP, MPB, or Lumerical.
