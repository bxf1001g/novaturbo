"""
Tests for the Brayton cycle thermodynamic solver.
"""

import math
import pytest

from src.physics.brayton import (
    FlightConditions,
    EngineInputs,
    CycleResults,
    solve_brayton_cycle,
)


def test_default_cycle_is_valid():
    """Default engine inputs should produce a valid, positive-thrust cycle."""
    results = solve_brayton_cycle(FlightConditions(), EngineInputs())
    assert results.is_valid
    assert results.thrust_N > 0


def test_thrust_positive():
    """Thrust must be non-negative for any reasonable input."""
    results = solve_brayton_cycle(FlightConditions(), EngineInputs())
    assert results.thrust_N >= 0


def test_tsfc_positive_and_finite():
    """TSFC should be a positive, finite number for a valid cycle."""
    results = solve_brayton_cycle(FlightConditions(), EngineInputs())
    assert results.tsfc_kg_N_s > 0
    assert math.isfinite(results.tsfc_kg_N_s)


def test_fuel_flow_positive():
    """Fuel flow must be positive."""
    results = solve_brayton_cycle(FlightConditions(), EngineInputs())
    assert results.fuel_flow_kg_s > 0


def test_exhaust_velocity_positive():
    """Exhaust velocity must be greater than zero for thrust to exist."""
    results = solve_brayton_cycle(FlightConditions(), EngineInputs())
    assert results.exhaust_velocity_m_s > 0


def test_station_temperatures_increase():
    """Temperature should increase from inlet through combustor."""
    results = solve_brayton_cycle(FlightConditions(), EngineInputs())
    T_inlet = results.stations['2_inlet_exit'].T_total_K
    T_compressor = results.stations['3_compressor_exit'].T_total_K
    T_combustor = results.stations['4_combustor_exit'].T_total_K
    assert T_compressor > T_inlet, "Compressor should raise total temperature"
    assert T_combustor > T_compressor, "Combustor should raise total temperature further"


def test_thermal_efficiency_bounded():
    """Thermal efficiency must be in (0, 1) for a valid cycle."""
    results = solve_brayton_cycle(FlightConditions(), EngineInputs())
    assert 0 < results.thermal_efficiency < 1


def test_compressor_power_less_than_turbine_power():
    """Turbine power must be at least as large as compressor power (single spool)."""
    results = solve_brayton_cycle(FlightConditions(), EngineInputs())
    # Turbine power drives the compressor, so turbine >= compressor
    assert results.turbine_power_W >= results.compressor_power_W * 0.95  # 5% tolerance


def test_higher_pressure_ratio_increases_thrust():
    """Increasing compressor pressure ratio (all else equal) should raise thrust."""
    engine_lo = EngineInputs(compressor_pressure_ratio=2.5)
    engine_hi = EngineInputs(compressor_pressure_ratio=4.5)
    r_lo = solve_brayton_cycle(FlightConditions(), engine_lo)
    r_hi = solve_brayton_cycle(FlightConditions(), engine_hi)
    assert r_hi.thrust_N > r_lo.thrust_N


def test_higher_mass_flow_increases_thrust():
    """More air flow should produce more thrust."""
    engine_lo = EngineInputs(mass_flow_air_kg_s=0.10)
    engine_hi = EngineInputs(mass_flow_air_kg_s=0.25)
    r_lo = solve_brayton_cycle(FlightConditions(), engine_lo)
    r_hi = solve_brayton_cycle(FlightConditions(), engine_hi)
    assert r_hi.thrust_N > r_lo.thrust_N


def test_high_tit_produces_warning():
    """TIT above 1300 K should trigger a warning."""
    engine = EngineInputs(combustor_exit_temperature_K=1350)
    results = solve_brayton_cycle(FlightConditions(), engine)
    assert any("TIT" in w for w in results.warnings)


def test_stations_dict_has_expected_keys():
    """All 6 stations should be present in the results."""
    results = solve_brayton_cycle(FlightConditions(), EngineInputs())
    expected = {
        '1_ambient', '2_inlet_exit', '3_compressor_exit',
        '4_combustor_exit', '5_turbine_exit', '6_nozzle_exit',
    }
    assert expected.issubset(results.stations.keys())


def test_tsfc_g_kN_s_consistent():
    """tsfc_g_kN_s should equal tsfc_kg_N_s * 1e6."""
    results = solve_brayton_cycle(FlightConditions(), EngineInputs())
    assert abs(results.tsfc_g_kN_s - results.tsfc_kg_N_s * 1e6) < 1e-9


def test_flight_mach_effect():
    """Non-zero Mach number should alter the cycle (ram pressure effect)."""
    r_static = solve_brayton_cycle(FlightConditions(M_flight=0.0), EngineInputs())
    r_mach = solve_brayton_cycle(FlightConditions(M_flight=0.5), EngineInputs())
    # Ram effect increases total pressure at inlet
    P01_static = r_static.stations['1_ambient'].P_total_Pa
    P01_mach = r_mach.stations['1_ambient'].P_total_Pa
    assert P01_mach > P01_static
