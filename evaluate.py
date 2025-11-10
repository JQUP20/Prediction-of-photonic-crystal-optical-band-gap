"""
Evaluation script for trained GNN models
"""

import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import json
import os

from src.data.photonic_crystal_generator import PhotonicCrystalGenerator
from src.data.graph_converter import CrystalToGraphConverter
from src.models.gnn_bandgap import GNNBandGapPredictor
from src.dft.band_structure_calculator import BandStructureCalculator
from src.utils.metrics import evaluate_predictions
from src.utils.visualization import plot_prediction_vs_actual


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate trained GNN model')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--num_test_samples', type=int, default=200,
                       help='Number of test samples')
    parser.add_argument('--grid_size', type=int, default=32,
                       help='Grid resolution')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed for test data')

    return parser.parse_args()


def generate_test_dataset(num_samples, grid_size, seed):
    """Generate test dataset"""
    print(f"Generating {num_samples} test samples...")

    crystal_gen = PhotonicCrystalGenerator(random_seed=seed)
    graph_converter = CrystalToGraphConverter()
    band_calc = BandStructureCalculator()

    crystals = crystal_gen.generate_dataset(num_samples, grid_size=grid_size)

    graph_dataset = []
    band_gaps = []

    for i, crystal in enumerate(crystals):
        try:
            band_gap_info = band_calc.calculate_band_gap(crystal)
            band_gap = band_gap_info['band_gap']

            graph_data = graph_converter.crystal_to_graph(crystal, band_gap)
            graph_dataset.append(graph_data)
            band_gaps.append(band_gap)

        except Exception as e:
            print(f"Error processing crystal {i}: {e}")
            continue

    print(f"Generated {len(graph_dataset)} test samples")
    return graph_dataset


def evaluate_model(model, loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            predictions = outputs['band_gap']

            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())

    return np.array(all_preds), np.array(all_targets)


def main():
    """Main evaluation function"""
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print()

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=args.device)

    model = GNNBandGapPredictor(
        node_feature_dim=6,
        edge_feature_dim=1,
        hidden_dim=128,
        num_conv_layers=4,
        num_fc_layers=3,
        dropout=0.2,
        pooling='attention'
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    print()

    # Generate test dataset
    test_dataset = generate_test_dataset(args.num_test_samples, args.grid_size, args.seed)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate
    print("Evaluating model...")
    predictions, targets = evaluate_model(model, test_loader, args.device)

    # Calculate metrics
    metrics = evaluate_predictions(targets, predictions)

    print("\nTest Results:")
    print("-" * 80)
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")

    # Save results
    results = {
        'metrics': metrics,
        'predictions': predictions.tolist(),
        'targets': targets.tolist()
    }

    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Plot predictions
    plot_prediction_vs_actual(
        targets, predictions,
        save_path=os.path.join(args.output_dir, 'test_predictions.png'),
        title="Test Set: Predicted vs Actual Band Gap"
    )

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
