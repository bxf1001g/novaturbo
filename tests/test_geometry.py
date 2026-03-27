"""
Tests for the engine geometry assembly module.
"""

import pytest

from src.geometry.assembly import (
    EngineAssemblyParams,
    EngineAssembly,
    assemble_engine,
)


def test_assemble_engine_returns_assembly():
    """assemble_engine should return an EngineAssembly instance."""
    params = EngineAssemblyParams()
    asm = assemble_engine(params)
    assert isinstance(asm, EngineAssembly)


def test_total_length_positive():
    """Total engine length must be positive."""
    asm = assemble_engine(EngineAssemblyParams())
    assert asm.total_length_mm > 0


def test_total_length_reasonable():
    """Default engine should be between 100 and 600 mm long."""
    asm = assemble_engine(EngineAssemblyParams())
    assert 100 < asm.total_length_mm < 600


def test_max_diameter_positive():
    """Max engine diameter must be positive."""
    asm = assemble_engine(EngineAssemblyParams())
    assert asm.max_diameter_mm > 0


def test_total_mass_positive():
    """Total engine mass must be positive."""
    asm = assemble_engine(EngineAssemblyParams())
    assert asm.total_mass_kg > 0


def test_total_mass_reasonable():
    """Default engine mass should be between 0.1 and 10 kg."""
    asm = assemble_engine(EngineAssemblyParams())
    assert 0.1 < asm.total_mass_kg < 10.0


def test_mass_breakdown_sums_to_total():
    """Sum of mass breakdown should equal total_mass_kg."""
    asm = assemble_engine(EngineAssemblyParams())
    breakdown_sum = sum(asm.mass_breakdown.values())
    assert abs(breakdown_sum - asm.total_mass_kg) < 1e-9


def test_mass_breakdown_has_all_components():
    """Mass breakdown should include all main components."""
    asm = assemble_engine(EngineAssemblyParams())
    for comp in ('inlet', 'compressor', 'combustor', 'turbine', 'nozzle', 'shaft'):
        assert comp in asm.mass_breakdown, f"Missing '{comp}' in mass breakdown"


def test_component_positions_ordered():
    """Component axial positions should be strictly increasing (front to back)."""
    asm = assemble_engine(EngineAssemblyParams())
    positions = [asm.component_positions[c]
                 for c in ('inlet', 'compressor', 'combustor', 'turbine', 'nozzle')]
    for i in range(len(positions) - 1):
        assert positions[i] < positions[i + 1], (
            f"Position of component {i} ({positions[i]:.1f}) should be less than "
            f"component {i+1} ({positions[i+1]:.1f})"
        )


def test_component_geometries_populated():
    """All five component geometry objects should be set after assembly."""
    asm = assemble_engine(EngineAssemblyParams())
    assert asm.inlet_geo is not None
    assert asm.compressor_geo is not None
    assert asm.combustor_geo is not None
    assert asm.turbine_geo is not None
    assert asm.nozzle_geo is not None


def test_compressor_tip_speed_reasonable():
    """Compressor tip speed at 100k RPM and 80mm tip should be ~419 m/s."""
    asm = assemble_engine(EngineAssemblyParams())
    # tip_speed = pi * d * RPM / 60
    import math
    params = EngineAssemblyParams()
    r_tip = params.compressor.impeller_tip_diameter_mm / 2000.0  # metres
    omega = params.compressor.rpm * 2 * math.pi / 60
    expected = r_tip * omega
    assert abs(asm.compressor_geo.tip_speed_m_s - expected) < 1.0
