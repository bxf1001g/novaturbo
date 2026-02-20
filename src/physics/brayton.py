"""
NovaTurbo — Brayton Cycle Thermodynamic Solver

Solves the ideal and real Brayton cycle for a micro turbojet engine.
Computes temperature, pressure, enthalpy at each station:

    Station 1: Ambient / Inlet
    Station 2: Compressor exit
    Station 3: Combustor exit (turbine inlet)
    Station 4: Turbine exit
    Station 5: Nozzle exit

Reference: Mattingly, "Elements of Propulsion: Gas Turbines and Rockets"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FlightConditions:
    """Ambient / flight conditions."""
    T_ambient_K: float = 288.15        # ISA sea level
    P_ambient_Pa: float = 101325.0     # 1 atm
    M_flight: float = 0.0              # Mach number (0 = static)
    altitude_m: float = 0.0


@dataclass
class EngineInputs:
    """Engine design parameters for cycle analysis."""
    # Inlet
    inlet_pressure_recovery: float = 0.98

    # Compressor
    compressor_pressure_ratio: float = 3.5
    compressor_isentropic_efficiency: float = 0.78

    # Combustor
    combustor_exit_temperature_K: float = 1100.0   # TIT
    combustor_pressure_drop_fraction: float = 0.04  # 4%
    combustion_efficiency: float = 0.95
    fuel_lower_heating_value_J_kg: float = 43e6    # Jet-A1

    # Turbine
    turbine_isentropic_efficiency: float = 0.82

    # Nozzle
    nozzle_discharge_coefficient: float = 0.98
    nozzle_isentropic_efficiency: float = 0.95

    # Fluid properties
    gamma_cold: float = 1.4            # Air (compressor side)
    gamma_hot: float = 1.33            # Combustion products
    cp_cold: float = 1005.0            # J/(kg·K)
    cp_hot: float = 1150.0             # J/(kg·K)
    R_air: float = 287.0               # J/(kg·K)

    # Mass flow
    mass_flow_air_kg_s: float = 0.15


@dataclass
class StationConditions:
    """Thermodynamic conditions at a single station."""
    name: str = ""
    T_total_K: float = 0.0             # Total (stagnation) temperature
    P_total_Pa: float = 0.0            # Total (stagnation) pressure
    T_static_K: float = 0.0            # Static temperature
    P_static_Pa: float = 0.0           # Static pressure
    h_total_J_kg: float = 0.0          # Total enthalpy
    velocity_m_s: float = 0.0          # Flow velocity
    mach: float = 0.0                  # Mach number
    density_kg_m3: float = 0.0


@dataclass
class CycleResults:
    """Complete Brayton cycle analysis results."""
    # Station conditions
    stations: dict = field(default_factory=dict)

    # Performance metrics
    thrust_N: float = 0.0
    specific_thrust_N_s_kg: float = 0.0
    fuel_flow_kg_s: float = 0.0
    air_fuel_ratio: float = 0.0
    tsfc_kg_N_s: float = 0.0           # Thrust-specific fuel consumption
    tsfc_g_kN_s: float = 0.0           # More readable unit
    thermal_efficiency: float = 0.0
    propulsive_efficiency: float = 0.0
    overall_efficiency: float = 0.0

    # Power
    compressor_power_W: float = 0.0
    turbine_power_W: float = 0.0
    net_power_W: float = 0.0

    # Exhaust
    exhaust_velocity_m_s: float = 0.0
    exhaust_temperature_K: float = 0.0

    # Validity checks
    is_valid: bool = True
    warnings: list = field(default_factory=list)


def solve_brayton_cycle(flight: FlightConditions,
                        engine: EngineInputs) -> CycleResults:
    """
    Solve the Brayton cycle for a turbojet engine.
    Returns complete station analysis and performance metrics.
    """
    results = CycleResults()
    gc = engine.gamma_cold
    gh = engine.gamma_hot
    cpc = engine.cp_cold
    cph = engine.cp_hot
    R = engine.R_air
    mdot_air = engine.mass_flow_air_kg_s

    # =============== STATION 1: Ambient / Inlet ===============
    T1 = flight.T_ambient_K
    P1 = flight.P_ambient_Pa

    # Ram effect (for non-zero flight Mach)
    T01 = T1 * (1 + (gc - 1) / 2 * flight.M_flight**2)
    P01 = P1 * (1 + (gc - 1) / 2 * flight.M_flight**2) ** (gc / (gc - 1))

    # Apply inlet pressure recovery
    P02 = P01 * engine.inlet_pressure_recovery
    T02 = T01  # Adiabatic inlet

    V_inlet = flight.M_flight * np.sqrt(gc * R * T1)

    results.stations['1_ambient'] = StationConditions(
        name='Ambient', T_total_K=T01, P_total_Pa=P01,
        T_static_K=T1, P_static_Pa=P1, velocity_m_s=V_inlet,
        mach=flight.M_flight
    )
    results.stations['2_inlet_exit'] = StationConditions(
        name='Inlet Exit', T_total_K=T02, P_total_Pa=P02
    )

    # =============== STATION 3: Compressor Exit ===============
    PR_c = engine.compressor_pressure_ratio
    eta_c = engine.compressor_isentropic_efficiency

    # Ideal compressor exit temperature
    T03_ideal = T02 * PR_c ** ((gc - 1) / gc)
    # Real compressor exit temperature (with efficiency)
    T03 = T02 + (T03_ideal - T02) / eta_c
    P03 = P02 * PR_c

    compressor_work = cpc * (T03 - T02)  # J/kg

    results.stations['3_compressor_exit'] = StationConditions(
        name='Compressor Exit', T_total_K=T03, P_total_Pa=P03,
        h_total_J_kg=cpc * T03
    )

    # =============== STATION 4: Combustor Exit (TIT) ===============
    T04 = engine.combustor_exit_temperature_K
    P04 = P03 * (1 - engine.combustor_pressure_drop_fraction)

    # Fuel flow rate: Q_in = mdot_fuel * LHV * eta_comb = (mdot_air + mdot_fuel) * cp_hot * T04 - mdot_air * cp_cold * T03
    # Simplified: mdot_fuel = mdot_air * cp_hot * (T04 - T03) / (LHV * eta_comb - cp_hot * T04)
    LHV = engine.fuel_lower_heating_value_J_kg
    eta_comb = engine.combustion_efficiency
    
    fuel_flow = mdot_air * cph * (T04 - T03) / (LHV * eta_comb - cph * T04)
    if fuel_flow < 0:
        results.is_valid = False
        results.warnings.append("Negative fuel flow — TIT too low or efficiency too low")
        fuel_flow = 0.001

    mdot_total = mdot_air + fuel_flow
    air_fuel_ratio = mdot_air / fuel_flow if fuel_flow > 0 else float('inf')

    results.stations['4_combustor_exit'] = StationConditions(
        name='Combustor Exit (TIT)', T_total_K=T04, P_total_Pa=P04,
        h_total_J_kg=cph * T04
    )

    # =============== STATION 5: Turbine Exit ===============
    # Turbine must produce enough work to drive the compressor
    # W_turbine = W_compressor (single spool, no power extraction)
    turbine_work = compressor_work * mdot_air / mdot_total  # Adjusted for fuel addition

    T05 = T04 - turbine_work / cph

    # Turbine pressure ratio from efficiency
    eta_t = engine.turbine_isentropic_efficiency
    T05_ideal = T04 - (T04 - T05) / eta_t
    PR_t = (T04 / T05_ideal) ** (gh / (gh - 1))
    P05 = P04 / PR_t

    if T05 < 0 or P05 < 0:
        results.is_valid = False
        results.warnings.append("Invalid turbine exit conditions")
        T05 = max(T05, 300)
        P05 = max(P05, flight.P_ambient_Pa)

    results.stations['5_turbine_exit'] = StationConditions(
        name='Turbine Exit', T_total_K=T05, P_total_Pa=P05,
        h_total_J_kg=cph * T05
    )

    # =============== STATION 6: Nozzle Exit ===============
    eta_n = engine.nozzle_isentropic_efficiency
    P_exit = flight.P_ambient_Pa  # Perfectly expanded nozzle assumption

    # Check if nozzle is choked
    P_critical = P05 * (2 / (gh + 1)) ** (gh / (gh - 1))

    if P_exit < P_critical:
        # Choked nozzle — exit at sonic conditions
        T06 = T05 * (2 / (gh + 1))
        P06 = P_critical
        V_exit = np.sqrt(gh * R * T06)  # Sonic velocity
        mach_exit = 1.0
        results.warnings.append("Nozzle is choked (sonic exit)")
    else:
        # Subsonic nozzle exit
        T06_ideal = T05 * (P_exit / P05) ** ((gh - 1) / gh)
        T06 = T05 - eta_n * (T05 - T06_ideal)
        P06 = P_exit
        V_exit = np.sqrt(2 * cph * (T05 - T06))
        mach_exit = V_exit / np.sqrt(gh * R * T06)

    V_exit *= engine.nozzle_discharge_coefficient

    results.stations['6_nozzle_exit'] = StationConditions(
        name='Nozzle Exit', T_total_K=T05, P_total_Pa=P06,
        T_static_K=T06, P_static_Pa=P06,
        velocity_m_s=V_exit, mach=mach_exit
    )

    # =============== PERFORMANCE METRICS ===============
    # Thrust
    F = mdot_total * V_exit - mdot_air * V_inlet + (P06 - flight.P_ambient_Pa) * 0  # Assume matched
    results.thrust_N = max(F, 0)
    results.specific_thrust_N_s_kg = F / mdot_air if mdot_air > 0 else 0
    results.exhaust_velocity_m_s = V_exit
    results.exhaust_temperature_K = T06

    # Fuel consumption
    results.fuel_flow_kg_s = fuel_flow
    results.air_fuel_ratio = air_fuel_ratio
    results.tsfc_kg_N_s = fuel_flow / F if F > 0 else float('inf')
    results.tsfc_g_kN_s = results.tsfc_kg_N_s * 1e6

    # Power
    results.compressor_power_W = compressor_work * mdot_air
    results.turbine_power_W = turbine_work * mdot_total
    results.net_power_W = 0.5 * mdot_total * V_exit**2 - 0.5 * mdot_air * V_inlet**2

    # Efficiencies
    Q_in = fuel_flow * LHV
    if Q_in > 0:
        results.thermal_efficiency = results.net_power_W / Q_in
    if V_exit > 0 and V_inlet == 0:
        results.propulsive_efficiency = 0  # Static thrust
    elif V_exit > 0:
        results.propulsive_efficiency = 2 / (1 + V_exit / V_inlet) if V_inlet > 0 else 0
    results.overall_efficiency = results.thermal_efficiency * max(results.propulsive_efficiency, 0.5)

    # Validation
    if T04 > 1300:
        results.warnings.append(f"TIT={T04:.0f}K exceeds typical micro turbojet limits")
    if T06 > 900:
        results.warnings.append(f"Exhaust temp {T06:.0f}K is very high")

    return results


def print_cycle_results(results: CycleResults):
    """Print formatted Brayton cycle results."""
    print(f"\n{'='*60}")
    print(f"  NovaTurbo Brayton Cycle Analysis")
    print(f"{'='*60}")

    print(f"\n  Station Analysis:")
    print(f"  {'Station':<25s} {'T_total(K)':>10s} {'P_total(kPa)':>12s} {'V(m/s)':>8s}")
    print(f"  {'-'*55}")
    for key, st in results.stations.items():
        print(f"  {st.name:<25s} {st.T_total_K:>10.1f} {st.P_total_Pa/1000:>12.1f} "
              f"{st.velocity_m_s:>8.1f}")

    print(f"\n  Performance:")
    print(f"    Thrust:              {results.thrust_N:.1f} N ({results.thrust_N/9.81:.2f} kgf)")
    print(f"    Specific thrust:     {results.specific_thrust_N_s_kg:.1f} N·s/kg")
    print(f"    Fuel flow:           {results.fuel_flow_kg_s*3600:.1f} g/hr")
    print(f"    Air-fuel ratio:      {results.air_fuel_ratio:.1f}")
    print(f"    TSFC:                {results.tsfc_g_kN_s:.1f} g/(kN·s)")
    print(f"    Exhaust velocity:    {results.exhaust_velocity_m_s:.1f} m/s")
    print(f"    Exhaust temperature: {results.exhaust_temperature_K:.0f} K "
          f"({results.exhaust_temperature_K-273:.0f} °C)")

    print(f"\n  Power:")
    print(f"    Compressor power:    {results.compressor_power_W:.0f} W ({results.compressor_power_W/1000:.1f} kW)")
    print(f"    Turbine power:       {results.turbine_power_W:.0f} W ({results.turbine_power_W/1000:.1f} kW)")
    print(f"    Net jet power:       {results.net_power_W:.0f} W ({results.net_power_W/1000:.1f} kW)")

    print(f"\n  Efficiency:")
    print(f"    Thermal:             {results.thermal_efficiency*100:.1f}%")
    print(f"    Overall:             {results.overall_efficiency*100:.1f}%")

    if results.warnings:
        print(f"\n  ⚠ Warnings:")
        for w in results.warnings:
            print(f"    • {w}")

    print(f"\n  Valid: {'✓' if results.is_valid else '✗'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    flight = FlightConditions()
    engine = EngineInputs()
    results = solve_brayton_cycle(flight, engine)
    print_cycle_results(results)
