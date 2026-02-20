"""
NovaTurbo — Simulation Module

Computes thermal, airflow, and stress fields across the engine
using real Brayton cycle data + material properties.

Outputs per-station field data that the 3D viewer maps onto meshes
as vertex-colored heatmaps, flow arrows, and stress overlays.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.physics.brayton import FlightConditions, EngineInputs, solve_brayton_cycle
from src.geometry.assembly import EngineAssemblyParams, assemble_engine


@dataclass
class ThermalField:
    """Temperature distribution across the engine."""
    # Per-component: z-positions (mm) and corresponding temperatures (K)
    component_temps: Dict[str, Dict] = field(default_factory=dict)
    # Global min/max for color scale
    t_min: float = 0.0
    t_max: float = 0.0
    # Material thermal limits per component
    material_limits: Dict[str, float] = field(default_factory=dict)


@dataclass
class FlowField:
    """Airflow velocity and pressure distribution."""
    # Streamline points: list of (x, y, z, vx, vy, vz, mach, pressure_kPa)
    streamlines: List[Dict] = field(default_factory=list)
    # Per-component flow data
    component_flow: Dict[str, Dict] = field(default_factory=dict)
    v_min: float = 0.0
    v_max: float = 0.0


@dataclass
class StressField:
    """Mechanical stress distribution."""
    # Per-component: z-positions and stress values (MPa)
    component_stress: Dict[str, Dict] = field(default_factory=dict)
    s_min: float = 0.0
    s_max: float = 0.0
    # Yield margins per component
    yield_margins: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationResults:
    """Complete simulation output."""
    thermal: ThermalField = field(default_factory=ThermalField)
    flow: FlowField = field(default_factory=FlowField)
    stress: StressField = field(default_factory=StressField)
    cycle: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# Material assignments and properties
COMPONENT_MATERIALS = {
    'inlet':      {'name': 'Ti-6Al-4V',    'max_temp_K': 673,  'yield_MPa': 880,  'density': 4430, 'alpha': 8.6e-6},
    'compressor': {'name': 'Ti-6Al-4V',    'max_temp_K': 673,  'yield_MPa': 880,  'density': 4430, 'alpha': 8.6e-6},
    'combustor':  {'name': 'Inconel 718',  'max_temp_K': 973,  'yield_MPa': 1034, 'density': 8190, 'alpha': 13e-6},
    'turbine':    {'name': 'Inconel 718',  'max_temp_K': 973,  'yield_MPa': 1034, 'density': 8190, 'alpha': 13e-6},
    'nozzle':     {'name': 'Inconel 625',  'max_temp_K': 1253, 'yield_MPa': 758,  'density': 8440, 'alpha': 12.8e-6},
    'shaft':      {'name': 'SS 316L',      'max_temp_K': 1143, 'yield_MPa': 205,  'density': 7990, 'alpha': 16e-6},
    'casing':     {'name': 'SS 316L',      'max_temp_K': 1143, 'yield_MPa': 205,  'density': 7990, 'alpha': 16e-6},
    'transitions':{'name': 'Inconel 718',  'max_temp_K': 973,  'yield_MPa': 1034, 'density': 8190, 'alpha': 13e-6},
}


def run_thermal_simulation(params: EngineAssemblyParams,
                           asm, cycle_results) -> ThermalField:
    """
    Compute temperature field across the engine using Brayton station data.
    Interpolates between stations and accounts for wall cooling effects.
    """
    thermal = ThermalField()
    stations = cycle_results.stations
    positions = asm.component_positions

    # Station temperatures (gas path total temperature)
    T_ambient = stations['1_ambient'].T_total_K
    T_inlet_exit = stations['2_inlet_exit'].T_total_K
    T_comp_exit = stations['3_compressor_exit'].T_total_K
    T_comb_exit = stations['4_combustor_exit'].T_total_K
    T_turb_exit = stations['5_turbine_exit'].T_total_K
    T_noz_exit = stations['6_nozzle_exit'].T_static_K

    # Wall temperature is typically 70-85% of gas temperature due to cooling
    wall_factor = 0.78

    # --- Inlet: ambient → slight compression heating ---
    z_inlet = np.linspace(0, params.inlet.length_mm, 20)
    t_profile = np.linspace(T_ambient, T_inlet_exit, 20)
    t_wall = t_profile * 0.95  # Minimal wall heating at inlet
    thermal.component_temps['inlet'] = {
        'z': z_inlet.tolist(),
        'gas_temp': t_profile.tolist(),
        'wall_temp': t_wall.tolist(),
        'z_offset': positions['inlet'],
    }

    # --- Compressor: inlet temp → compressed temp ---
    z_comp = np.linspace(0, params.compressor.axial_length_mm, 20)
    t = np.linspace(0, 1, 20)
    # Compression follows ~polynomial curve (more heating near exit)
    t_profile = T_inlet_exit + (T_comp_exit - T_inlet_exit) * (t ** 0.7)
    t_wall = t_profile * 0.92
    thermal.component_temps['compressor'] = {
        'z': z_comp.tolist(),
        'gas_temp': t_profile.tolist(),
        'wall_temp': t_wall.tolist(),
        'z_offset': positions['compressor'],
    }

    # --- Combustor: compressed temp → TIT (biggest jump) ---
    z_comb = np.linspace(0, params.combustor.length_mm, 30)
    t = np.linspace(0, 1, 30)
    # Primary zone: rapid rise in first 40%, then gradual to TIT
    t_profile = T_comp_exit + (T_comb_exit - T_comp_exit) * (1 - np.exp(-3.5 * t))
    # Wall temp with film cooling effect
    cooling_effectiveness = 0.3 + 0.15 * np.sin(np.pi * t)  # Cooling holes help in middle
    t_wall = t_profile * (1 - cooling_effectiveness) + T_comp_exit * cooling_effectiveness
    thermal.component_temps['combustor'] = {
        'z': z_comb.tolist(),
        'gas_temp': t_profile.tolist(),
        'wall_temp': t_wall.tolist(),
        'z_offset': positions['combustor'],
    }

    # --- Turbine: TIT → turbine exit (work extraction cools gas) ---
    z_turb = np.linspace(0, params.turbine.stage_axial_length_mm, 20)
    t = np.linspace(0, 1, 20)
    t_profile = T_comb_exit + (T_turb_exit - T_comb_exit) * t
    t_wall = t_profile * wall_factor
    thermal.component_temps['turbine'] = {
        'z': z_turb.tolist(),
        'gas_temp': t_profile.tolist(),
        'wall_temp': t_wall.tolist(),
        'z_offset': positions['turbine'],
    }

    # --- Nozzle: turbine exit → exhaust ---
    z_noz = np.linspace(0, params.nozzle.length_mm, 20)
    t = np.linspace(0, 1, 20)
    t_profile = T_turb_exit + (T_noz_exit - T_turb_exit) * (t ** 0.5)
    t_wall = t_profile * wall_factor
    thermal.component_temps['nozzle'] = {
        'z': z_noz.tolist(),
        'gas_temp': t_profile.tolist(),
        'wall_temp': t_wall.tolist(),
        'z_offset': positions['nozzle'],
    }

    # --- Shaft: absorbs heat from all surrounding components ---
    z_shaft = np.linspace(0, asm.total_length_mm, 40)
    # Shaft temperature follows a smoothed version of surrounding gas temps
    shaft_temp = np.interp(z_shaft,
        [0, params.inlet.length_mm,
         positions['compressor'] + params.compressor.axial_length_mm,
         positions['combustor'], positions['combustor'] + params.combustor.length_mm,
         positions['turbine'] + params.turbine.stage_axial_length_mm,
         asm.total_length_mm],
        [T_ambient * 0.8, T_inlet_exit * 0.7, T_comp_exit * 0.6,
         T_comp_exit * 0.65, T_comb_exit * 0.45,
         T_turb_exit * 0.5, T_noz_exit * 0.6])
    thermal.component_temps['shaft'] = {
        'z': z_shaft.tolist(),
        'gas_temp': shaft_temp.tolist(),
        'wall_temp': shaft_temp.tolist(),
        'z_offset': 0,
    }

    # --- Casing: outer shell, absorbs radiated heat ---
    z_cas = np.linspace(0, asm.total_length_mm, 40)
    casing_temp = np.interp(z_cas,
        [0, positions['compressor'], positions['combustor'],
         positions['combustor'] + params.combustor.length_mm * 0.5,
         positions['turbine'], positions['nozzle'], asm.total_length_mm],
        [T_ambient, T_inlet_exit * 0.5, T_comp_exit * 0.4,
         T_comb_exit * 0.25, T_turb_exit * 0.35,
         T_turb_exit * 0.4, T_noz_exit * 0.5])
    thermal.component_temps['casing'] = {
        'z': z_cas.tolist(),
        'gas_temp': casing_temp.tolist(),
        'wall_temp': casing_temp.tolist(),
        'z_offset': 0,
    }

    # Transitions: blend between neighboring components
    thermal.component_temps['transitions'] = {
        'z': [0, asm.total_length_mm],
        'gas_temp': [float(T_comp_exit), float(T_turb_exit)],
        'wall_temp': [float(T_comp_exit * 0.8), float(T_turb_exit * 0.8)],
        'z_offset': 0,
    }

    # Global range
    all_temps = []
    for comp_data in thermal.component_temps.values():
        all_temps.extend(comp_data['wall_temp'])
        all_temps.extend(comp_data['gas_temp'])
    thermal.t_min = float(min(all_temps))
    thermal.t_max = float(max(all_temps))

    # Material limits
    for comp, mat in COMPONENT_MATERIALS.items():
        thermal.material_limits[comp] = mat['max_temp_K']

    return thermal


def run_flow_simulation(params: EngineAssemblyParams,
                        asm, cycle_results) -> FlowField:
    """
    Compute airflow field: velocity vectors, pressure, and Mach number
    along the gas path through the engine.
    """
    flow = FlowField()
    stations = cycle_results.stations
    positions = asm.component_positions
    R_air = 287.0

    # Extract station data
    station_data = [
        ('1_ambient',         0,                                                    1.4),
        ('2_inlet_exit',      positions['inlet'] + params.inlet.length_mm,          1.4),
        ('3_compressor_exit', positions['compressor'] + params.compressor.axial_length_mm, 1.4),
        ('4_combustor_exit',  positions['combustor'] + params.combustor.length_mm,   1.33),
        ('5_turbine_exit',    positions['turbine'] + params.turbine.stage_axial_length_mm, 1.33),
        ('6_nozzle_exit',     positions['nozzle'] + params.nozzle.length_mm,         1.33),
    ]

    # Generate streamlines (multiple radial positions)
    n_radial = 5
    n_axial_per_section = 15
    total_length = asm.total_length_mm

    for r_frac in np.linspace(0.2, 0.8, n_radial):
        streamline = {'points': []}

        for i, (st_key, z_pos, gamma) in enumerate(station_data):
            st = stations[st_key]
            T = st.T_total_K
            P = st.P_total_Pa

            # Velocity from station data or estimate
            if st.velocity_m_s > 0:
                v = st.velocity_m_s
            else:
                # Estimate: v = mdot / (rho * A)
                rho = P / (R_air * T) if T > 0 else 1.2
                # Approximate annular area at this station
                r_outer = _get_radius_at_z(z_pos, params, asm)
                r_inner = r_outer * 0.4
                A = np.pi * (r_outer**2 - r_inner**2) / 1e6  # m²
                v = 0.15 / (rho * max(A, 0.001))  # mdot / (rho * A)
                v = min(v, 400)  # Cap

            mach = v / np.sqrt(gamma * R_air * T) if T > 0 else 0

            if i < len(station_data) - 1:
                next_z = station_data[i + 1][1]
                z_pts = np.linspace(z_pos, next_z, n_axial_per_section)
            else:
                z_pts = [z_pos]

            for z in (z_pts if hasattr(z_pts, '__iter__') else [z_pts]):
                r = _get_radius_at_z(z, params, asm) * r_frac
                # Swirl component (compressor adds, turbine removes)
                theta_v = 0
                if positions['compressor'] <= z <= positions['compressor'] + params.compressor.axial_length_mm:
                    theta_v = v * 0.3  # Compressor swirl
                elif positions['turbine'] <= z <= positions['turbine'] + params.turbine.stage_axial_length_mm:
                    theta_v = -v * 0.2  # Turbine de-swirl

                streamline['points'].append({
                    'x': float(r * 0.01),  # Small radial offset for visual spread
                    'y': float(r * r_frac * 0.005),
                    'z': float(z),
                    'vx': float(theta_v * 0.01),
                    'vy': float(0),
                    'vz': float(v),
                    'speed': float(v),
                    'mach': float(mach),
                    'pressure_kPa': float(P / 1000),
                    'temperature_K': float(T),
                })

        flow.streamlines.append(streamline)

    # Per-component flow summary
    for comp_name, st_in, st_out in [
        ('inlet', '1_ambient', '2_inlet_exit'),
        ('compressor', '2_inlet_exit', '3_compressor_exit'),
        ('combustor', '3_compressor_exit', '4_combustor_exit'),
        ('turbine', '4_combustor_exit', '5_turbine_exit'),
        ('nozzle', '5_turbine_exit', '6_nozzle_exit'),
    ]:
        s_in = stations[st_in]
        s_out = stations[st_out]
        flow.component_flow[comp_name] = {
            'inlet_pressure_kPa': round(s_in.P_total_Pa / 1000, 1),
            'outlet_pressure_kPa': round(s_out.P_total_Pa / 1000, 1),
            'pressure_ratio': round(s_out.P_total_Pa / s_in.P_total_Pa, 3) if s_in.P_total_Pa > 0 else 0,
            'inlet_temp_K': round(s_in.T_total_K, 1),
            'outlet_temp_K': round(s_out.T_total_K, 1),
            'inlet_velocity': round(s_in.velocity_m_s, 1),
            'outlet_velocity': round(s_out.velocity_m_s, 1),
        }

    all_speeds = [p['speed'] for s in flow.streamlines for p in s['points']]
    flow.v_min = float(min(all_speeds)) if all_speeds else 0
    flow.v_max = float(max(all_speeds)) if all_speeds else 1

    return flow


def run_stress_simulation(params: EngineAssemblyParams,
                          asm, cycle_results, thermal: ThermalField) -> StressField:
    """
    Compute mechanical stress distribution:
    - Centrifugal stress on rotating parts (compressor, turbine, shaft)
    - Pressure stress on casings (combustor, casing)
    - Thermal stress from temperature gradients
    """
    stress = StressField()
    positions = asm.component_positions
    rpm = params.compressor.rpm
    omega = 2 * np.pi * rpm / 60  # rad/s

    for comp_name, mat in COMPONENT_MATERIALS.items():
        z_pts = 20
        comp_len = _get_component_length(comp_name, params, asm)
        z = np.linspace(0, comp_len, z_pts)
        z_offset = positions.get(comp_name, 0)

        centrifugal = np.zeros(z_pts)
        pressure_stress = np.zeros(z_pts)
        thermal_stress = np.zeros(z_pts)

        # --- Centrifugal stress (rotating parts) ---
        if comp_name in ('compressor', 'turbine', 'shaft'):
            if comp_name == 'compressor':
                # Stress increases with radius: σ = ρ * ω² * r² / 2
                r_hub = params.compressor.impeller_hub_diameter_mm / 2000  # meters
                r_tip = params.compressor.impeller_tip_diameter_mm / 2000
                t_frac = np.linspace(0, 1, z_pts)
                r = r_hub + (r_tip - r_hub) * (t_frac ** 0.5)
                centrifugal = mat['density'] * omega**2 * r**2 / 2 / 1e6  # MPa
            elif comp_name == 'turbine':
                r_hub = params.turbine.hub_diameter_mm / 2000
                r_tip = params.turbine.tip_diameter_mm / 2000
                r = np.linspace(r_hub, r_tip, z_pts)
                centrifugal = mat['density'] * omega**2 * r**2 / 2 / 1e6
            elif comp_name == 'shaft':
                r = params.shaft_diameter_mm / 2000
                centrifugal = np.full(z_pts, mat['density'] * omega**2 * r**2 / 2 / 1e6)

        # --- Pressure stress (hoop stress for cylindrical shells) ---
        if comp_name in ('combustor', 'casing', 'nozzle', 'inlet'):
            if comp_name == 'combustor':
                # σ_hoop = P * r / t
                P_internal = cycle_results.stations['4_combustor_exit'].P_total_Pa
                r = params.combustor.outer_diameter_mm / 2000
                t = params.combustor.liner_thickness_mm / 1000
                pressure_stress = np.full(z_pts, P_internal * r / t / 1e6)
            elif comp_name == 'casing':
                P_max = cycle_results.stations['3_compressor_exit'].P_total_Pa
                r = 0.06  # Approximate casing radius
                t = params.casing_thickness_mm / 1000
                # Varies along length
                t_frac = np.linspace(0, 1, z_pts)
                P_local = P_max * (1 - 0.3 * t_frac)  # Decreasing toward nozzle
                pressure_stress = P_local * r / t / 1e6
            elif comp_name == 'nozzle':
                P_int = cycle_results.stations['5_turbine_exit'].P_total_Pa
                r = params.nozzle.inlet_diameter_mm / 2000
                t = params.nozzle.wall_thickness_mm / 1000
                pressure_stress = np.full(z_pts, P_int * r / t / 1e6)
            elif comp_name == 'inlet':
                pressure_stress = np.full(z_pts, 2.0)  # Low pressure at inlet

        # --- Thermal stress ---
        # σ_thermal = E * α * ΔT / (1 - ν), simplified
        E_GPa = {'Ti-6Al-4V': 114, 'Inconel 718': 200, 'Inconel 625': 205, 'SS 316L': 193}
        nu = 0.3
        E = E_GPa.get(mat['name'], 200) * 1e3  # MPa
        alpha = mat['alpha']

        if comp_name in thermal.component_temps:
            temps = thermal.component_temps[comp_name]
            wall_temps = np.array(temps['wall_temp'])
            if len(wall_temps) != z_pts:
                wall_temps = np.interp(np.linspace(0, 1, z_pts),
                                       np.linspace(0, 1, len(wall_temps)), wall_temps)
            # Gradient-based thermal stress
            dT = np.abs(np.gradient(wall_temps)) * 5  # Amplify gradient effect
            # Also add absolute thermal stress from deviation from assembly temp
            T_assembly = 293  # 20°C
            dT_abs = np.abs(wall_temps - T_assembly) * 0.05
            thermal_stress = E * alpha * (dT + dT_abs) / (1 - nu) / 1e6  # MPa

        # Combined (von Mises approximation)
        total_stress = np.sqrt(centrifugal**2 + pressure_stress**2 + thermal_stress**2 +
                               centrifugal * pressure_stress)

        yield_margin = (mat['yield_MPa'] - total_stress) / mat['yield_MPa'] * 100

        stress.component_stress[comp_name] = {
            'z': z.tolist(),
            'z_offset': z_offset,
            'centrifugal_MPa': centrifugal.tolist(),
            'pressure_MPa': pressure_stress.tolist(),
            'thermal_MPa': thermal_stress.tolist(),
            'total_MPa': total_stress.tolist(),
            'yield_margin_pct': yield_margin.tolist(),
            'material': mat['name'],
            'yield_MPa': mat['yield_MPa'],
        }

        stress.yield_margins[comp_name] = float(np.min(yield_margin))

    all_stress = []
    for comp_data in stress.component_stress.values():
        all_stress.extend(comp_data['total_MPa'])
    stress.s_min = float(min(all_stress)) if all_stress else 0
    stress.s_max = float(max(all_stress)) if all_stress else 1

    return stress


def run_full_simulation() -> dict:
    """
    Run complete thermal + flow + stress simulation and return
    JSON-serializable results for the 3D viewer.
    """
    params = EngineAssemblyParams()
    asm = assemble_engine(params)

    flight = FlightConditions()
    engine = EngineInputs(
        compressor_pressure_ratio=params.compressor.pressure_ratio,
        combustor_exit_temperature_K=params.turbine.inlet_temperature_K,
        mass_flow_air_kg_s=0.15,
    )

    cycle = solve_brayton_cycle(flight, engine)

    thermal = run_thermal_simulation(params, asm, cycle)
    flow = run_flow_simulation(params, asm, cycle)
    stress = run_stress_simulation(params, asm, cycle, thermal)

    # Cycle summary
    cycle_summary = {
        'thrust_N': round(cycle.thrust_N, 1),
        'thrust_kgf': round(cycle.thrust_N / 9.81, 2),
        'exhaust_velocity': round(cycle.exhaust_velocity_m_s, 1),
        'exhaust_temp_K': round(cycle.exhaust_temperature_K, 0),
        'fuel_flow_g_hr': round(cycle.fuel_flow_kg_s * 3600 * 1000, 1),
        'tsfc_g_kNs': round(cycle.tsfc_g_kN_s, 1),
        'thermal_efficiency_pct': round(cycle.thermal_efficiency * 100, 1),
        'compressor_power_kW': round(cycle.compressor_power_W / 1000, 2),
        'turbine_power_kW': round(cycle.turbine_power_W / 1000, 2),
        'is_valid': cycle.is_valid,
        'warnings': cycle.warnings,
        'stations': {},
    }

    # Station Z positions from assembly
    station_z_positions = {
        '1_ambient': 0,
        '2_inlet_exit': round(asm.component_positions.get('compressor', 30), 1),
        '3_compressor_exit': round(asm.component_positions.get('combustor', 55), 1),
        '4_combustor_exit': round(asm.component_positions.get('turbine', 135), 1),
        '5_turbine_exit': round(asm.component_positions.get('nozzle', 165), 1),
        '6_nozzle_exit': round(asm.total_length_mm, 1),
    }

    for key, st in cycle.stations.items():
        cycle_summary['stations'][key] = {
            'name': st.name,
            'T_total_K': round(st.T_total_K, 1),
            'P_total_kPa': round(st.P_total_Pa / 1000, 2),
            'velocity': round(st.velocity_m_s, 1),
            'mach': round(st.mach, 3),
            'z_mm': station_z_positions.get(key, 0),
        }

    return {
        'thermal': {
            'component_temps': thermal.component_temps,
            't_min': thermal.t_min,
            't_max': thermal.t_max,
            'material_limits': thermal.material_limits,
        },
        'flow': {
            'streamlines': flow.streamlines,
            'component_flow': flow.component_flow,
            'v_min': flow.v_min,
            'v_max': flow.v_max,
        },
        'stress': {
            'component_stress': stress.component_stress,
            's_min': stress.s_min,
            's_max': stress.s_max,
            'yield_margins': stress.yield_margins,
        },
        'cycle': cycle_summary,
    }


def _get_radius_at_z(z_mm, params, asm):
    """Get approximate outer flow-path radius at axial position z."""
    pos = asm.component_positions
    if z_mm <= pos['inlet'] + params.inlet.length_mm:
        t = z_mm / max(params.inlet.length_mm, 1)
        return params.inlet.inlet_diameter_mm / 2 * (1 - t * 0.3)
    elif z_mm <= pos['compressor'] + params.compressor.axial_length_mm:
        return params.compressor.impeller_tip_diameter_mm / 2
    elif z_mm <= pos['combustor'] + params.combustor.length_mm:
        return params.combustor.outer_diameter_mm / 2
    elif z_mm <= pos['turbine'] + params.turbine.stage_axial_length_mm:
        return params.turbine.tip_diameter_mm / 2
    else:
        t = (z_mm - pos['nozzle']) / max(params.nozzle.length_mm, 1)
        t = min(max(t, 0), 1)
        return params.nozzle.inlet_diameter_mm / 2 * (1 - t * 0.35)


def _get_component_length(comp_name, params, asm):
    """Get axial length of a component."""
    lengths = {
        'inlet': params.inlet.length_mm,
        'compressor': params.compressor.axial_length_mm,
        'combustor': params.combustor.length_mm,
        'turbine': params.turbine.stage_axial_length_mm,
        'nozzle': params.nozzle.length_mm,
        'shaft': asm.total_length_mm,
        'casing': asm.total_length_mm,
        'transitions': asm.total_length_mm,
    }
    return lengths.get(comp_name, 50)


if __name__ == "__main__":
    import json
    results = run_full_simulation()
    print(f"Simulation complete:")
    print(f"  Thermal range: {results['thermal']['t_min']:.0f} - {results['thermal']['t_max']:.0f} K")
    print(f"  Flow range: {results['flow']['v_min']:.0f} - {results['flow']['v_max']:.0f} m/s")
    print(f"  Stress range: {results['stress']['s_min']:.1f} - {results['stress']['s_max']:.1f} MPa")
    print(f"  Thrust: {results['cycle']['thrust_N']} N")
    print(f"  Streamlines: {len(results['flow']['streamlines'])}")

    # Check yield margins
    for comp, margin in results['stress']['yield_margins'].items():
        status = '✓' if margin > 0 else '✗ YIELD EXCEEDED'
        print(f"  {comp:15s} yield margin: {margin:.1f}% {status}")
