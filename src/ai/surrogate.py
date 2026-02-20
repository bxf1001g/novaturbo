"""
NovaTurbo — Neural Network Surrogate Model

PyTorch neural network that learns the mapping:
    Engine geometry parameters → Performance metrics

This is 1000x faster than the physics solver, enabling
rapid optimization over the design space.

Architecture: Multi-layer perceptron (MLP)
    Input:  ~17 geometry parameters (normalized)
    Output: ~8 performance metrics (thrust, SFC, mass, etc.)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class SurrogateConfig:
    """Configuration for the surrogate neural network."""
    # Architecture
    hidden_layers: List[int] = None  # Default set in __post_init__
    activation: str = "relu"
    dropout: float = 0.1
    batch_norm: bool = True

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    patience: int = 20          # Early stopping patience
    lr_scheduler: bool = True

    # Data
    val_split: float = 0.15
    test_split: float = 0.10

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 256, 256, 128, 64]


# Input parameter names (geometry)
INPUT_FEATURES = [
    'compressor_pressure_ratio', 'compressor_efficiency',
    'compressor_diameter_mm', 'compressor_blade_count',
    'combustor_length_mm', 'combustor_outer_diameter_mm',
    'combustor_inner_diameter_mm', 'combustor_liner_thickness_mm',
    'combustor_num_injectors', 'combustor_air_fuel_ratio',
    'turbine_inlet_temp_K', 'turbine_efficiency',
    'turbine_blade_count', 'turbine_hub_tip_ratio',
    'nozzle_exit_diameter_mm', 'mass_flow_kg_s', 'rpm'
]

# Output performance names
OUTPUT_FEATURES = [
    'thrust_N', 'specific_thrust', 'fuel_flow_kg_s',
    'tsfc_kg_N_s', 'exhaust_velocity_m_s', 'exhaust_temp_K',
    'thermal_efficiency', 'total_mass_kg', 'thrust_to_weight'
]


class EngineDataset(Dataset):
    """PyTorch dataset for engine design data."""

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = torch.FloatTensor(inputs)
        self.outputs = torch.FloatTensor(outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class SurrogateModel(nn.Module):
    """
    MLP surrogate model for engine performance prediction.
    """

    def __init__(self, n_inputs: int, n_outputs: int, config: SurrogateConfig):
        super().__init__()

        self.config = config
        layers = []
        prev_size = n_inputs

        for hidden_size in config.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "gelu":
                layers.append(nn.GELU())
            elif config.activation == "silu":
                layers.append(nn.SiLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            prev_size = hidden_size

        # Output layer (no activation — regression)
        layers.append(nn.Linear(prev_size, n_outputs))

        self.network = nn.Sequential(*layers)

        # Normalization parameters (set during training)
        self.register_buffer('input_mean', torch.zeros(n_inputs))
        self.register_buffer('input_std', torch.ones(n_inputs))
        self.register_buffer('output_mean', torch.zeros(n_outputs))
        self.register_buffer('output_std', torch.ones(n_outputs))

    def forward(self, x):
        # Normalize inputs
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        # Predict normalized outputs
        y_norm = self.network(x_norm)
        # Denormalize
        y = y_norm * (self.output_std + 1e-8) + self.output_mean
        return y

    def predict(self, params: dict) -> dict:
        """Predict performance from a parameter dictionary."""
        self.eval()
        x = torch.FloatTensor([[params.get(f, 0) for f in INPUT_FEATURES]])
        with torch.no_grad():
            y = self.forward(x)
        result = {}
        for i, name in enumerate(OUTPUT_FEATURES):
            result[name] = y[0, i].item()
        return result


def load_and_prepare_data(csv_path: str, config: SurrogateConfig) -> Tuple:
    """
    Load dataset CSV and prepare train/val/test splits.
    Returns: (train_loader, val_loader, test_loader, input_stats, output_stats)
    """
    df = pd.read_csv(csv_path)

    # Filter valid designs only
    if 'is_valid' in df.columns:
        df = df[df['is_valid'] == True].reset_index(drop=True)

    # Extract inputs and outputs
    inputs = df[INPUT_FEATURES].values.astype(np.float32)
    outputs = df[OUTPUT_FEATURES].values.astype(np.float32)

    # Replace inf/nan
    inputs = np.nan_to_num(inputs, nan=0, posinf=0, neginf=0)
    outputs = np.nan_to_num(outputs, nan=0, posinf=0, neginf=0)

    # Compute normalization stats
    input_mean = inputs.mean(axis=0)
    input_std = inputs.std(axis=0)
    output_mean = outputs.mean(axis=0)
    output_std = outputs.std(axis=0)

    # Create dataset
    dataset = EngineDataset(inputs, outputs)

    # Split
    n = len(dataset)
    n_test = int(n * config.test_split)
    n_val = int(n * config.val_split)
    n_train = n - n_val - n_test

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    stats = {
        'input_mean': input_mean, 'input_std': input_std,
        'output_mean': output_mean, 'output_std': output_std,
        'n_train': n_train, 'n_val': n_val, 'n_test': n_test
    }

    return train_loader, val_loader, test_loader, stats


def train_surrogate(csv_path: str, config: Optional[SurrogateConfig] = None,
                    save_dir: str = "data/trained_models",
                    verbose: bool = True) -> Tuple[SurrogateModel, dict]:
    """
    Train the surrogate neural network on the generated dataset.

    Args:
        csv_path: Path to the dataset CSV
        config: Training configuration
        save_dir: Directory to save trained model
        verbose: Print training progress

    Returns:
        (trained_model, training_history)
    """
    if config is None:
        config = SurrogateConfig()

    # Load data
    train_loader, val_loader, test_loader, stats = load_and_prepare_data(csv_path, config)

    if verbose:
        print(f"NovaTurbo Surrogate Model Training")
        print(f"  Train: {stats['n_train']}, Val: {stats['n_val']}, Test: {stats['n_test']}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SurrogateModel(len(INPUT_FEATURES), len(OUTPUT_FEATURES), config).to(device)

    # Set normalization parameters
    model.input_mean = torch.FloatTensor(stats['input_mean']).to(device)
    model.input_std = torch.FloatTensor(stats['input_std']).to(device)
    model.output_mean = torch.FloatTensor(stats['output_mean']).to(device)
    model.output_std = torch.FloatTensor(stats['output_std']).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Model parameters: {n_params:,}")
        print(f"  Device: {device}")

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    ) if config.lr_scheduler else None

    criterion = nn.MSELoss()

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # LR scheduling
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'stats': stats,
                'epoch': epoch,
                'val_loss': val_loss
            }, os.path.join(save_dir, 'best_surrogate.pt'))
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{config.epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"lr={current_lr:.6f}")

        if patience_counter >= config.patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Test evaluation
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    test_loss /= len(test_loader)

    if verbose:
        print(f"\n  Test loss: {test_loss:.6f}")

        # Per-output R² scores
        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        print(f"\n  Per-output R² scores:")
        for i, name in enumerate(OUTPUT_FEATURES):
            ss_res = np.sum((targets[:, i] - preds[:, i])**2)
            ss_tot = np.sum((targets[:, i] - targets[:, i].mean())**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            print(f"    {name:25s} R² = {r2:.4f}")

    history['test_loss'] = test_loss
    return model, history


def load_surrogate(model_path: str, device: str = 'cpu') -> SurrogateModel:
    """Load a trained surrogate model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    stats = checkpoint['stats']

    model = SurrogateModel(len(INPUT_FEATURES), len(OUTPUT_FEATURES), config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.input_mean = torch.FloatTensor(stats['input_mean'])
    model.input_std = torch.FloatTensor(stats['input_std'])
    model.output_mean = torch.FloatTensor(stats['output_mean'])
    model.output_std = torch.FloatTensor(stats['output_std'])
    model.eval()

    return model


if __name__ == "__main__":
    print("NovaTurbo Surrogate Model")
    print("Usage: First generate dataset with dataset.py, then train with:")
    print("  python -m src.ai.surrogate --data data/generated/dataset_10000.csv")
