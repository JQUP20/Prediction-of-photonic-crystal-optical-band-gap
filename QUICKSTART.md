# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Prediction-of-photonic-crystal-optical-band-gap

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Demo

### 1. Generate Photonic Crystal Structures

```python
from src.data.photonic_crystal_generator import PhotonicCrystalGenerator

# Create generator
generator = PhotonicCrystalGenerator()

# Generate a 2D square lattice
crystal = generator.generate_2d_square_lattice(
    n_background=1.0,  # Air
    n_rod=3.4,         # Silicon
    lattice_constant=0.5,
    rod_radius=0.3,
    grid_size=64
)

print(f"Crystal type: {crystal.lattice_type}")
print(f"Filling fraction: {crystal.filling_fraction:.3f}")
```

### 2. Calculate Band Structure

```python
from src.dft.band_structure_calculator import BandStructureCalculator
from src.utils.visualization import plot_band_structure

# Calculate band structure
calculator = BandStructureCalculator()
k_points, frequencies = calculator.calculate_band_structure(crystal)

# Find band gap
band_gap_info = calculator.calculate_band_gap(crystal)
print(f"Band gap: {band_gap_info['band_gap']:.6f}")

# Visualize
fig = plot_band_structure(k_points, frequencies, band_gap_info)
```

### 3. Train GNN Model

```bash
# Train with default parameters
python train.py --num_samples 100 --epochs 50 --output_dir results/demo

# Train with custom settings
python train.py \
    --num_samples 500 \
    --grid_size 32 \
    --hidden_dim 128 \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --output_dir results/experiment1
```

### 4. Evaluate Model

```bash
python evaluate.py \
    --model_path results/demo/best_model.pt \
    --num_test_samples 100 \
    --output_dir evaluation_results
```

### 5. Use Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_crystal_generation_demo.ipynb
# 2. notebooks/02_band_structure_calculation.ipynb
# 3. notebooks/03_gnn_training.ipynb
```

## Example: End-to-End Workflow

```python
import torch
from src.data.photonic_crystal_generator import PhotonicCrystalGenerator
from src.data.graph_converter import CrystalToGraphConverter
from src.models.gnn_bandgap import GNNBandGapPredictor
from src.dft.band_structure_calculator import BandStructureCalculator

# 1. Generate crystal
generator = PhotonicCrystalGenerator(random_seed=42)
crystal = generator.generate_2d_square_lattice(1.0, 3.4, 0.5, 0.3, 64)

# 2. Calculate true band gap
calculator = BandStructureCalculator()
true_band_gap = calculator.calculate_band_gap(crystal)['band_gap']
print(f"True band gap: {true_band_gap:.6f}")

# 3. Convert to graph
converter = CrystalToGraphConverter()
graph = converter.crystal_to_graph(crystal)

# 4. Load trained model and predict
model = GNNBandGapPredictor()
checkpoint = torch.load('results/demo/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    prediction = model.predict_band_gap(graph)
    print(f"Predicted band gap: {prediction.item():.6f}")
```

## Common Use Cases

### Generate Dataset for Training

```python
generator = PhotonicCrystalGenerator(random_seed=42)
dataset = generator.generate_dataset(
    num_samples=1000,
    lattice_types=['2D_square', '2D_triangular'],
    grid_size=32
)
```

### Calculate Transmission Spectrum

```python
from src.dft.transmission_calculator import TransmissionCalculator

trans_calc = TransmissionCalculator(
    wavelength_range=(0.3, 2.0),
    num_wavelengths=200
)

result = trans_calc.calculate_transmission_spectrum(crystal)
wavelengths = result['wavelengths']
transmission = result['transmission']
```

### Ensemble Prediction with Uncertainty

```python
from src.models.gnn_bandgap import EnsembleGNNPredictor

ensemble = EnsembleGNNPredictor(num_models=5)
# Load trained models...
mean_prediction, uncertainty = ensemble.predict_with_uncertainty(graph)
print(f"Prediction: {mean_prediction.item():.6f} ± {uncertainty.item():.6f}")
```

## Performance Tips

1. **Use GPU for Training**: Add `--device cuda` to training command
2. **Reduce Grid Size**: Use `--grid_size 32` for faster processing
3. **Batch Processing**: Increase `--batch_size` if you have enough memory
4. **Early Stopping**: Training will stop automatically if no improvement

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size`
- Reduce `--grid_size`
- Use CPU: `--device cpu`

### Slow Training
- Reduce `--num_samples`
- Reduce `--grid_size`
- Reduce `--num_conv_layers`

### Poor Predictions
- Increase `--num_samples`
- Increase `--epochs`
- Tune `--hidden_dim` and `--num_conv_layers`

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore [notebooks](notebooks/) for interactive tutorials
- Modify model architecture in `src/models/gnn_bandgap.py`
- Add custom crystal types in `src/data/photonic_crystal_generator.py`

## Citation

If you use this code in your research, please cite:

```
@software{photonic_crystal_bandgap,
  title={Photonic Crystal Optical Band Gap Prediction using GNN},
  author={Photonic Crystal Research Team},
  year={2024},
  url={https://github.com/yourusername/Prediction-of-photonic-crystal-optical-band-gap}
}
```
