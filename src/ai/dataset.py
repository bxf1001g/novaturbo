"""
NovaTurbo — AI Dataset Generator

Generates training data for the neural network surrogate model by:
1. Sampling the design parameter space (Latin Hypercube Sampling)
2. Running each design through the physics solver
3. Collecting geometry params (inputs) + performance metrics (outputs)
4. Saving as CSV for PyTorch training

This is how we create 10K–50K design variants without needing external data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import os

from ..physics.brayton import (
    FlightConditions, EngineInputs, CycleResults,
    solve_brayton_cycle
)


@dataclass
class DesignSpaceBounds:
    """Min/max bounds for each design parameter."""
    # Compressor
    compressor_pressure_ratio: Tuple[float, float] = (2.0, 5.0)
    compressor_efficiency: Tuple[float, float] = (0.70, 0.85)
    compressor_diameter_mm: Tuple[float, float] = (60.0, 150.0)
    compressor_blade_count: Tuple[int, int] = (8, 20)

    # Combustor
    combustor_length_mm: Tuple[float, float] = (50.0, 150.0)
    combustor_outer_diameter_mm: Tuple[float, float] = (80.0, 160.0)
    combustor_inner_diameter_mm: Tuple[float, float] = (40.0, 90.0)
    combustor_liner_thickness_mm: Tuple[float, float] = (0.8, 2.0)
    combustor_num_injectors: Tuple[int, int] = (4, 12)
    combustor_air_fuel_ratio: Tuple[float, float] = (40.0, 80.0)

    # Turbine
    turbine_inlet_temp_K: Tuple[float, float] = (800.0, 1200.0)
    turbine_efficiency: Tuple[float, float] = (0.75, 0.88)
    turbine_blade_count: Tuple[int, int] = (11, 23)
    turbine_hub_tip_ratio: Tuple[float, float] = (0.6, 0.85)

    # Nozzle
    nozzle_exit_diameter_mm: Tuple[float, float] = (30.0, 80.0)

    # Operating
    mass_flow_kg_s: Tuple[float, float] = (0.08, 0.30)
    rpm: Tuple[float, float] = (60000.0, 150000.0)

    # TPMS Lattice parameters
    lattice_cell_size_mm: Tuple[float, float] = (4.0, 14.0)
    lattice_density: Tuple[float, float] = (0.15, 0.55)  # Fill fraction


def latin_hypercube_sample(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """
    Generate Latin Hypercube Samples in [0, 1]^n_dims.
    Ensures even coverage of the design space.
    """
    rng = np.random.RandomState(seed)
    samples = np.zeros((n_samples, n_dims))

    for dim in range(n_dims):
        # Create evenly spaced intervals
        intervals = np.linspace(0, 1, n_samples + 1)
        # Random point within each interval
        points = intervals[:-1] + rng.uniform(0, 1, n_samples) * (1.0 / n_samples)
        # Shuffle to remove correlation between dimensions
        rng.shuffle(points)
        samples[:, dim] = points

    return samples


def scale_samples(lhs_samples: np.ndarray, bounds: DesignSpaceBounds) -> pd.DataFrame:
    """
    Scale LHS samples from [0,1] to actual parameter ranges.
    Returns a DataFrame with named columns.
    """
    param_names = []
    param_bounds = []

    for field_name in bounds.__dataclass_fields__:
        bound = getattr(bounds, field_name)
        param_names.append(field_name)
        param_bounds.append(bound)

    n_params = len(param_names)
    assert lhs_samples.shape[1] == n_params, \
        f"LHS has {lhs_samples.shape[1]} dims but {n_params} params defined"

    scaled = np.zeros_like(lhs_samples)
    for i, (lo, hi) in enumerate(param_bounds):
        scaled[:, i] = lo + (hi - lo) * lhs_samples[:, i]

    df = pd.DataFrame(scaled, columns=param_names)

    # Round integer parameters
    int_params = ['compressor_blade_count', 'combustor_num_injectors', 'turbine_blade_count']
    for p in int_params:
        if p in df.columns:
            df[p] = df[p].round().astype(int)

    return df


def evaluate_design(row: pd.Series) -> Dict:
    """
    Evaluate a single design point through the physics solver.
    Returns performance metrics dictionary.
    """
    try:
        flight = FlightConditions()
        engine = EngineInputs(
            compressor_pressure_ratio=row['compressor_pressure_ratio'],
            compressor_isentropic_efficiency=row['compressor_efficiency'],
            combustor_exit_temperature_K=row['turbine_inlet_temp_K'],
            combustion_efficiency=0.95,
            turbine_isentropic_efficiency=row['turbine_efficiency'],
            mass_flow_air_kg_s=row['mass_flow_kg_s']
        )

        results = solve_brayton_cycle(flight, engine)

        # Estimate engine mass (with lattice weight reduction)
        lattice_density = row.get('lattice_density', 0.3)
        lattice_reduction = 1.0 - (1.0 - lattice_density) * 0.6  # Up to 60% wall weight saved

        compressor_mass = estimate_component_mass(
            row['compressor_diameter_mm'], 25, 4430)  # Ti-6Al-4V
        combustor_mass = estimate_component_mass(
            row['combustor_outer_diameter_mm'], row['combustor_length_mm'], 8190)  # Inconel 718
        combustor_mass *= lattice_reduction  # Lattice-reduced combustor
        turbine_mass = estimate_component_mass(
            row['compressor_diameter_mm'] * 1.05, 30, 8190)
        nozzle_mass = estimate_component_mass(
            row['nozzle_exit_diameter_mm'] * 1.5, 50, 8190)
        nozzle_mass *= lattice_reduction * 1.1  # Lighter lattice in nozzle
        total_mass = (compressor_mass + combustor_mass + turbine_mass + nozzle_mass) * 1.15

        thrust_to_weight = results.thrust_N / (total_mass * 9.81) if total_mass > 0 else 0

        return {
            'thrust_N': results.thrust_N,
            'specific_thrust': results.specific_thrust_N_s_kg,
            'fuel_flow_kg_s': results.fuel_flow_kg_s,
            'tsfc_kg_N_s': results.tsfc_kg_N_s,
            'exhaust_velocity_m_s': results.exhaust_velocity_m_s,
            'exhaust_temp_K': results.exhaust_temperature_K,
            'compressor_power_W': results.compressor_power_W,
            'thermal_efficiency': results.thermal_efficiency,
            'total_mass_kg': total_mass,
            'thrust_to_weight': thrust_to_weight,
            'lattice_surface_area_ratio': 2.5 + 1.5 * lattice_density,  # TPMS SA multiplier
            'is_valid': results.is_valid and results.thrust_N > 0,
            'n_warnings': len(results.warnings)
        }

    except Exception as e:
        return {
            'thrust_N': 0, 'specific_thrust': 0, 'fuel_flow_kg_s': 0,
            'tsfc_kg_N_s': 0, 'exhaust_velocity_m_s': 0, 'exhaust_temp_K': 0,
            'compressor_power_W': 0, 'thermal_efficiency': 0,
            'total_mass_kg': 0, 'thrust_to_weight': 0,
            'is_valid': False, 'n_warnings': -1
        }


def estimate_component_mass(diameter_mm: float, length_mm: float,
                            density_kg_m3: float) -> float:
    """Quick mass estimate from envelope dimensions (shell approximation)."""
    r = diameter_mm / 2000.0
    L = length_mm / 1000.0
    wall_t = 0.0015  # 1.5mm average wall
    shell_vol = 2 * np.pi * r * wall_t * L
    return shell_vol * density_kg_m3


def generate_dataset(n_samples: int = 10000,
                     output_dir: str = "data/generated",
                     seed: int = 42,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Generate the complete training dataset.

    1. Sample design space with LHS
    2. Evaluate each design through physics solver
    3. Save combined CSV

    Args:
        n_samples: Number of design variants to generate
        output_dir: Directory to save CSV files
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        DataFrame with inputs + outputs
    """
    bounds = DesignSpaceBounds()
    n_params = len(bounds.__dataclass_fields__)

    if verbose:
        print(f"NovaTurbo Dataset Generator")
        print(f"  Generating {n_samples} design variants...")
        print(f"  Parameters: {n_params}")

    # Step 1: Latin Hypercube Sampling
    t0 = time.time()
    lhs = latin_hypercube_sample(n_samples, n_params, seed)
    inputs_df = scale_samples(lhs, bounds)

    if verbose:
        print(f"  LHS sampling: {time.time()-t0:.2f}s")

    # Step 2: Evaluate each design
    t0 = time.time()
    outputs = []
    valid_count = 0

    for i in range(n_samples):
        result = evaluate_design(inputs_df.iloc[i])
        outputs.append(result)

        if result['is_valid']:
            valid_count += 1

        if verbose and (i + 1) % (n_samples // 10) == 0:
            pct = (i + 1) / n_samples * 100
            print(f"  Progress: {pct:.0f}% ({valid_count}/{i+1} valid)")

    outputs_df = pd.DataFrame(outputs)

    if verbose:
        print(f"  Evaluation: {time.time()-t0:.2f}s")
        print(f"  Valid designs: {valid_count}/{n_samples} ({valid_count/n_samples*100:.1f}%)")

    # Step 3: Combine and save
    dataset = pd.concat([inputs_df, outputs_df], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"dataset_{n_samples}.csv")
    dataset.to_csv(filepath, index=False)

    if verbose:
        print(f"  Saved to: {filepath}")
        print(f"\n  Dataset shape: {dataset.shape}")
        print(f"  Input columns: {list(inputs_df.columns)}")
        print(f"  Output columns: {list(outputs_df.columns)}")

        # Summary statistics for valid designs
        valid = dataset[dataset['is_valid'] == True]
        if len(valid) > 0:
            print(f"\n  Valid Design Statistics:")
            print(f"    Thrust range:     {valid['thrust_N'].min():.1f} — {valid['thrust_N'].max():.1f} N")
            print(f"    TSFC range:       {valid['tsfc_kg_N_s'].min():.6f} — {valid['tsfc_kg_N_s'].max():.6f}")
            print(f"    Mass range:       {valid['total_mass_kg'].min()*1000:.0f} — {valid['total_mass_kg'].max()*1000:.0f} g")
            print(f"    T/W range:        {valid['thrust_to_weight'].min():.1f} — {valid['thrust_to_weight'].max():.1f}")

    return dataset


if __name__ == "__main__":
    # Quick test with 100 samples
    dataset = generate_dataset(n_samples=100, verbose=True)
