"""
Training script for GNN-based photonic band gap prediction
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import json
from tqdm import tqdm

from src.data.photonic_crystal_generator import PhotonicCrystalGenerator
from src.data.graph_converter import CrystalToGraphConverter
from src.models.gnn_bandgap import GNNBandGapPredictor
from src.dft.band_structure_calculator import BandStructureCalculator
from src.utils.metrics import evaluate_predictions
from src.utils.visualization import plot_training_history, plot_prediction_vs_actual


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train GNN for photonic band gap prediction')

    # Data parameters
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of training samples')
    parser.add_argument('--grid_size', type=int, default=32,
                       help='Grid resolution for crystal structures')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--num_conv_layers', type=int, default=4,
                       help='Number of graph convolution layers')
    parser.add_argument('--num_fc_layers', type=int, default=3,
                       help='Number of fully connected layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--pooling', type=str, default='attention',
                       choices=['mean', 'max', 'add', 'attention'],
                       help='Global pooling method')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')

    # System parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Checkpoint save frequency')

    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def generate_dataset(num_samples, grid_size, seed):
    """Generate photonic crystal dataset with band gap labels"""
    print(f"Generating {num_samples} photonic crystal samples...")

    # Initialize generators
    crystal_gen = PhotonicCrystalGenerator(random_seed=seed)
    graph_converter = CrystalToGraphConverter()
    band_calc = BandStructureCalculator()

    # Generate crystals
    crystals = crystal_gen.generate_dataset(num_samples, grid_size=grid_size)

    # Calculate band gaps and convert to graphs
    graph_dataset = []
    band_gaps = []

    for i, crystal in enumerate(tqdm(crystals, desc="Calculating band gaps")):
        try:
            # Calculate band gap using DFT-like method
            band_gap_info = band_calc.calculate_band_gap(crystal)
            band_gap = band_gap_info['band_gap']

            # Convert to graph
            graph_data = graph_converter.crystal_to_graph(crystal, band_gap)
            graph_dataset.append(graph_data)
            band_gaps.append(band_gap)

        except Exception as e:
            print(f"Error processing crystal {i}: {e}")
            continue

    print(f"Successfully generated {len(graph_dataset)} samples")
    print(f"Band gap statistics: min={np.min(band_gaps):.4f}, "
          f"max={np.max(band_gaps):.4f}, mean={np.mean(band_gaps):.4f}")

    return graph_dataset


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)
        predictions = outputs['band_gap']

        # Compute loss
        loss = criterion(predictions, batch.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_preds.extend(predictions.detach().cpu().numpy())
        all_targets.extend(batch.y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = evaluate_predictions(np.array(all_targets), np.array(all_preds))

    return avg_loss, metrics


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            outputs = model(batch)
            predictions = outputs['band_gap']

            # Compute loss
            loss = criterion(predictions, batch.y)

            total_loss += loss.item() * batch.num_graphs
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = evaluate_predictions(np.array(all_targets), np.array(all_preds))

    return avg_loss, metrics, np.array(all_preds), np.array(all_targets)


def main():
    """Main training function"""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print("=" * 80)
    print("Photonic Crystal Band Gap Prediction - Training")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Generate dataset
    dataset = generate_dataset(args.num_samples, args.grid_size, args.seed)

    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    indices = np.random.permutation(len(dataset))
    train_dataset = [dataset[i] for i in indices[:train_size]]
    val_dataset = [dataset[i] for i in indices[train_size:]]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    # Initialize model
    model = GNNBandGapPredictor(
        node_feature_dim=6,
        edge_feature_dim=1,
        hidden_dim=args.hidden_dim,
        num_conv_layers=args.num_conv_layers,
        num_fc_layers=args.num_fc_layers,
        dropout=args.dropout,
        pooling=args.pooling
    ).to(args.device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=args.patience // 2, verbose=True)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'train_r2': [],
        'val_r2': []
    }

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Training loop
    print("Starting training...")
    print("-" * 80)

    for epoch in range(args.epochs):
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer,
                                                criterion, args.device)

        # Validate
        val_loss, val_metrics, val_preds, val_targets = validate(model, val_loader,
                                                                  criterion, args.device)

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_metrics['mae'])
        history['val_mae'].append(val_metrics['mae'])
        history['train_r2'].append(train_metrics['r2'])
        history['val_r2'].append(val_metrics['r2'])

        # Print progress
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.6f} | MAE: {train_metrics['mae']:.6f} | R²: {train_metrics['r2']:.4f}")
        print(f"  Val Loss:   {val_loss:.6f} | MAE: {val_metrics['mae']:.6f} | R²: {val_metrics['r2']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print("  >>> Saved best model")
        else:
            epochs_without_improvement += 1

        # Periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pt'))

        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break

        print()

    print("-" * 80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Save training history
    np.save(os.path.join(args.output_dir, 'history.npy'), history)

    # Plot training history
    fig = plot_training_history(history, save_path=os.path.join(args.output_dir, 'training_history.png'))

    # Plot final predictions
    fig = plot_prediction_vs_actual(val_targets, val_preds,
                                   save_path=os.path.join(args.output_dir, 'predictions.png'))

    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
