"""
Tests for the AI dataset generation and Latin Hypercube Sampling utilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.ai.dataset import (
    DesignSpaceBounds,
    latin_hypercube_sample,
    scale_samples,
    evaluate_design,
    estimate_component_mass,
    generate_dataset,
)


# ---------------------------------------------------------------------------
# latin_hypercube_sample
# ---------------------------------------------------------------------------

def test_lhs_shape():
    """LHS output shape should match requested n_samples × n_dims."""
    samples = latin_hypercube_sample(50, 5)
    assert samples.shape == (50, 5)


def test_lhs_values_in_unit_hypercube():
    """All LHS values must lie in [0, 1]."""
    samples = latin_hypercube_sample(100, 10)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


def test_lhs_coverage():
    """Each dimension should have at least one sample in every decile."""
    samples = latin_hypercube_sample(100, 1)
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        assert np.any((samples[:, 0] >= lo) & (samples[:, 0] < hi + 1e-9)), (
            f"No sample found in decile [{lo:.1f}, {hi:.1f}]"
        )


def test_lhs_reproducible():
    """Same seed should produce identical samples."""
    s1 = latin_hypercube_sample(20, 4, seed=7)
    s2 = latin_hypercube_sample(20, 4, seed=7)
    np.testing.assert_array_equal(s1, s2)


def test_lhs_different_seeds():
    """Different seeds should produce different samples."""
    s1 = latin_hypercube_sample(20, 4, seed=1)
    s2 = latin_hypercube_sample(20, 4, seed=2)
    assert not np.array_equal(s1, s2)


# ---------------------------------------------------------------------------
# scale_samples
# ---------------------------------------------------------------------------

def test_scale_samples_shape():
    """Scaled DataFrame should have the right number of rows and columns."""
    bounds = DesignSpaceBounds()
    n_params = len(bounds.__dataclass_fields__)
    lhs = latin_hypercube_sample(30, n_params)
    df = scale_samples(lhs, bounds)
    assert df.shape == (30, n_params)


def test_scale_samples_within_bounds():
    """All scaled values must lie within the declared min/max for each parameter."""
    bounds = DesignSpaceBounds()
    n_params = len(bounds.__dataclass_fields__)
    lhs = latin_hypercube_sample(200, n_params, seed=0)
    df = scale_samples(lhs, bounds)
    for field_name in bounds.__dataclass_fields__:
        lo, hi = getattr(bounds, field_name)
        assert df[field_name].min() >= lo - 1e-9, (
            f"{field_name} min {df[field_name].min()} < {lo}"
        )
        assert df[field_name].max() <= hi + 1e-9, (
            f"{field_name} max {df[field_name].max()} > {hi}"
        )


def test_scale_samples_integer_columns():
    """Integer parameters should contain only integer values."""
    bounds = DesignSpaceBounds()
    n_params = len(bounds.__dataclass_fields__)
    lhs = latin_hypercube_sample(50, n_params)
    df = scale_samples(lhs, bounds)
    for col in ('compressor_blade_count', 'combustor_num_injectors', 'turbine_blade_count'):
        assert (df[col] == df[col].round()).all(), f"{col} contains non-integer values"


# ---------------------------------------------------------------------------
# estimate_component_mass
# ---------------------------------------------------------------------------

def test_estimate_component_mass_positive():
    """Mass estimate should always be positive."""
    m = estimate_component_mass(80, 25, 4430)
    assert m > 0


def test_estimate_component_mass_scales_with_size():
    """Larger components should have greater mass."""
    m_small = estimate_component_mass(40, 15, 8190)
    m_large = estimate_component_mass(120, 60, 8190)
    assert m_large > m_small


# ---------------------------------------------------------------------------
# evaluate_design
# ---------------------------------------------------------------------------

def _default_row() -> pd.Series:
    """Create a representative design row using default parameter values."""
    return pd.Series({
        'compressor_pressure_ratio': 3.5,
        'compressor_efficiency': 0.78,
        'compressor_diameter_mm': 80.0,
        'compressor_blade_count': 12,
        'combustor_length_mm': 80.0,
        'combustor_outer_diameter_mm': 110.0,
        'combustor_inner_diameter_mm': 60.0,
        'combustor_liner_thickness_mm': 1.2,
        'combustor_num_injectors': 8,
        'combustor_air_fuel_ratio': 60.0,
        'turbine_inlet_temp_K': 1100.0,
        'turbine_efficiency': 0.82,
        'turbine_blade_count': 17,
        'turbine_hub_tip_ratio': 0.70,
        'nozzle_exit_diameter_mm': 50.0,
        'mass_flow_kg_s': 0.15,
        'rpm': 100000.0,
        'lattice_cell_size_mm': 8.0,
        'lattice_density': 0.30,
    })


def test_evaluate_design_returns_dict():
    """evaluate_design should return a dictionary."""
    result = evaluate_design(_default_row())
    assert isinstance(result, dict)


def test_evaluate_design_has_required_keys():
    """Output dictionary must contain all expected performance keys."""
    result = evaluate_design(_default_row())
    for key in ('thrust_N', 'tsfc_kg_N_s', 'total_mass_kg',
                'thrust_to_weight', 'thermal_efficiency', 'is_valid'):
        assert key in result, f"Missing key '{key}'"


def test_evaluate_design_valid_for_default():
    """Default design parameters should produce a valid, positive-thrust design."""
    result = evaluate_design(_default_row())
    assert result['is_valid'] is True
    assert result['thrust_N'] > 0


def test_evaluate_design_mass_positive():
    """Total mass should always be positive."""
    result = evaluate_design(_default_row())
    assert result['total_mass_kg'] > 0


def test_evaluate_design_thrust_to_weight_consistent():
    """thrust_to_weight should equal thrust / (mass * g)."""
    result = evaluate_design(_default_row())
    if result['total_mass_kg'] > 0:
        expected = result['thrust_N'] / (result['total_mass_kg'] * 9.81)
        assert abs(result['thrust_to_weight'] - expected) < 1e-6


# ---------------------------------------------------------------------------
# generate_dataset  (small smoke test)
# ---------------------------------------------------------------------------

def test_generate_dataset_small(tmp_path):
    """generate_dataset with n=20 should produce a CSV with the right shape."""
    df = generate_dataset(n_samples=20, output_dir=str(tmp_path), verbose=False)
    assert len(df) == 20
    # Must have both input and output columns
    assert 'compressor_pressure_ratio' in df.columns
    assert 'thrust_N' in df.columns
    assert 'is_valid' in df.columns


def test_generate_dataset_saves_csv(tmp_path):
    """generate_dataset should write a CSV file to the output directory."""
    import os
    generate_dataset(n_samples=5, output_dir=str(tmp_path), verbose=False)
    csv_path = os.path.join(str(tmp_path), 'dataset_5.csv')
    assert os.path.exists(csv_path)
