"""
NovaTurbo — 3D Viewer Web Server

Flask backend that serves STL files and engine data
for the Three.js-based 3D viewer frontend.
"""

import os
import sys
import json
import threading
from datetime import datetime
from flask import Flask, render_template, send_from_directory, jsonify, request
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')
CORS(app)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STL_DIR = os.path.join(PROJECT_ROOT, 'exports', 'stl')
USER_LEARNING_DIR = os.path.join(PROJECT_ROOT, 'data', 'user_learning')
UI_VARIANTS_CSV = os.path.join(USER_LEARNING_DIR, 'ui_variants.csv')
UI_TRAIN_DATASET_CSV = os.path.join(USER_LEARNING_DIR, 'dataset_ui_augmented.csv')
CFD_SAMPLES_CSV = os.path.join(USER_LEARNING_DIR, 'cfd_samples.csv')
CFD_CALIBRATION_JSON = os.path.join(USER_LEARNING_DIR, 'cfd_calibration.json')
UI_TRAIN_LOCK = threading.Lock()
UI_TRAIN_STATUS = {
    'state': 'idle',  # idle | running | done | failed
    'message': 'No training started',
    'started_at': None,
    'finished_at': None,
    'variant_count': 0,
    'last_epochs': 0,
}
UI_CFD_LOCK = threading.Lock()
UI_CFD_STATUS = {
    'state': 'idle',  # idle | running | done | failed
    'message': 'No CFD calibration started',
    'solver': None,
    'started_at': None,
    'finished_at': None,
    'sample_count': 0,
    'processed': 0,
}


COMPONENTS = {
    'inlet':       {'color': '#4FC3F7', 'label': 'Air Inlet',         'order': 0},
    'compressor':  {'color': '#81C784', 'label': 'Compressor',        'order': 1},
    'combustor':   {'color': '#FF8A65', 'label': 'Combustion Chamber', 'order': 2},
    'turbine':     {'color': '#FFD54F', 'label': 'Turbine Stage',     'order': 3},
    'nozzle':      {'color': '#E57373', 'label': 'Exhaust Nozzle',    'order': 4},
    'shaft':       {'color': '#B0BEC5', 'label': 'Main Shaft',        'order': 5},
    'casing':      {'color': '#78909C', 'label': 'Outer Casing',      'order': 6},
    'transitions': {'color': '#9575CD', 'label': 'Transitions',       'order': 7},
    'combustor_lattice': {'color': '#FF6D00', 'label': 'Combustor Lattice', 'order': 8},
    'nozzle_lattice':    {'color': '#FF3D00', 'label': 'Nozzle Lattice', 'order': 9},
}


def _clamp(value, default, low, high):
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float(default)
    return max(low, min(high, v))


def _build_engine_state_from_payload(payload: dict):
    """Build geometry + cycle state from dashboard payload."""
    from src.geometry.assembly import EngineAssemblyParams, assemble_engine
    from src.physics.brayton import FlightConditions, EngineInputs, solve_brayton_cycle

    params = EngineAssemblyParams()

    # Geometry controls
    params.nozzle.exit_diameter_mm = _clamp(
        payload.get('nozzle_exit_mm'),
        params.nozzle.exit_diameter_mm,
        35.0,
        params.nozzle.inlet_diameter_mm - 5.0
    )
    params.nozzle.length_mm = _clamp(
        payload.get('nozzle_length_mm'),
        params.nozzle.length_mm,
        30.0,
        90.0
    )
    params.combustor.outer_diameter_mm = _clamp(
        payload.get('combustor_outer_mm'),
        params.combustor.outer_diameter_mm,
        90.0,
        140.0
    )
    params.combustor.inner_diameter_mm = _clamp(
        payload.get('combustor_inner_mm'),
        params.combustor.inner_diameter_mm,
        45.0,
        params.combustor.outer_diameter_mm - 20.0
    )
    params.combustor.length_mm = _clamp(
        payload.get('combustor_length_mm'),
        params.combustor.length_mm,
        50.0,
        130.0
    )

    # Operating controls
    controls = {
        'pr': _clamp(payload.get('pr'), params.compressor.pressure_ratio, 2.0, 6.0),
        'mdot': _clamp(payload.get('mdot'), params.compressor.mass_flow_kg_s, 0.05, 0.4),
        'tit': _clamp(payload.get('tit'), params.turbine.inlet_temperature_K, 800.0, 1300.0),
        'eta_c': _clamp(payload.get('eta_c'), 0.78, 0.60, 0.92),
        'eta_t': _clamp(payload.get('eta_t'), 0.82, 0.65, 0.92),
        'mach': _clamp(payload.get('mach'), 0.0, 0.0, 0.8),
    }

    params.compressor.pressure_ratio = controls['pr']
    params.compressor.mass_flow_kg_s = controls['mdot']
    params.turbine.inlet_temperature_K = controls['tit']
    params.combustor.outlet_temperature_K = controls['tit']

    asm = assemble_engine(params)

    engine = EngineInputs(
        compressor_pressure_ratio=controls['pr'],
        compressor_isentropic_efficiency=controls['eta_c'],
        combustor_exit_temperature_K=controls['tit'],
        turbine_isentropic_efficiency=controls['eta_t'],
        mass_flow_air_kg_s=controls['mdot'],
    )
    flight = FlightConditions(M_flight=controls['mach'], altitude_m=0.0)
    result = solve_brayton_cycle(flight, engine)

    return params, asm, result, controls


def _build_ui_variant_record(params, asm, result, controls):
    """Map current configuration to surrogate training row format."""
    thrust_to_weight = result.thrust_N / (asm.total_mass_kg * 9.81) if asm.total_mass_kg > 0 else 0.0
    return {
        # Inputs (must match src.ai.surrogate.INPUT_FEATURES)
        'compressor_pressure_ratio': params.compressor.pressure_ratio,
        'compressor_efficiency': controls['eta_c'],
        'compressor_diameter_mm': params.compressor.impeller_tip_diameter_mm,
        'compressor_blade_count': params.compressor.blade_count,
        'combustor_length_mm': params.combustor.length_mm,
        'combustor_outer_diameter_mm': params.combustor.outer_diameter_mm,
        'combustor_inner_diameter_mm': params.combustor.inner_diameter_mm,
        'combustor_liner_thickness_mm': params.combustor.liner_thickness_mm,
        'combustor_num_injectors': params.combustor.num_fuel_injectors,
        'combustor_air_fuel_ratio': params.combustor.air_fuel_ratio,
        'turbine_inlet_temp_K': controls['tit'],
        'turbine_efficiency': controls['eta_t'],
        'turbine_blade_count': params.turbine.blade_count,
        'turbine_hub_tip_ratio': params.turbine.hub_tip_ratio,
        'nozzle_exit_diameter_mm': params.nozzle.exit_diameter_mm,
        'mass_flow_kg_s': controls['mdot'],
        'rpm': params.compressor.rpm,
        # Outputs (must match src.ai.surrogate.OUTPUT_FEATURES)
        'thrust_N': result.thrust_N,
        'specific_thrust': result.specific_thrust_N_s_kg,
        'fuel_flow_kg_s': result.fuel_flow_kg_s,
        'tsfc_kg_N_s': result.tsfc_kg_N_s,
        'exhaust_velocity_m_s': result.exhaust_velocity_m_s,
        'exhaust_temp_K': result.exhaust_temperature_K,
        'thermal_efficiency': result.thermal_efficiency,
        'total_mass_kg': asm.total_mass_kg,
        'thrust_to_weight': thrust_to_weight,
        # Metadata
        'is_valid': bool(result.is_valid and result.thrust_N > 0),
        'n_warnings': len(result.warnings),
        'source': 'ui',
    }


def _get_ui_variant_count():
    if not os.path.exists(UI_VARIANTS_CSV):
        return 0
    import pandas as pd
    try:
        return len(pd.read_csv(UI_VARIANTS_CSV))
    except Exception:
        return 0


def _load_cfd_model_if_available():
    if not os.path.exists(CFD_CALIBRATION_JSON):
        return None
    from src.physics.cfd_calibration import load_calibration_model
    return load_calibration_model(CFD_CALIBRATION_JSON)


def _run_cfd_calibration_job(solver: str, max_cases: int, timeout_s: int):
    import pandas as pd
    from src.physics.cfd_calibration import (
        run_solver_case,
        fit_calibration_from_samples,
        save_calibration_model,
    )

    with UI_CFD_LOCK:
        UI_CFD_STATUS['state'] = 'running'
        UI_CFD_STATUS['message'] = f'Running {solver} calibration cases...'
        UI_CFD_STATUS['solver'] = solver
        UI_CFD_STATUS['started_at'] = datetime.utcnow().isoformat() + 'Z'
        UI_CFD_STATUS['finished_at'] = None
        UI_CFD_STATUS['processed'] = 0

    try:
        if not os.path.exists(UI_VARIANTS_CSV):
            raise RuntimeError("No UI variants found. Save variant samples before CFD calibration.")

        ui_df = pd.read_csv(UI_VARIANTS_CSV)
        if ui_df.empty:
            raise RuntimeError("UI variants file is empty.")

        max_cases = int(max(1, min(max_cases, len(ui_df))))
        subset = ui_df.tail(max_cases).reset_index(drop=True)
        samples = []

        for idx, row in subset.iterrows():
            cfd_metrics = run_solver_case(row.to_dict(), solver=solver, timeout_s=timeout_s)
            samples.append({
                'solver': solver,
                'compressor_pressure_ratio': row.get('compressor_pressure_ratio', 0.0),
                'turbine_inlet_temp_K': row.get('turbine_inlet_temp_K', 0.0),
                'mass_flow_kg_s': row.get('mass_flow_kg_s', 0.0),
                'nozzle_exit_diameter_mm': row.get('nozzle_exit_diameter_mm', 0.0),
                'combustor_outer_diameter_mm': row.get('combustor_outer_diameter_mm', 0.0),
                'combustor_inner_diameter_mm': row.get('combustor_inner_diameter_mm', 0.0),
                'combustor_length_mm': row.get('combustor_length_mm', 0.0),
                'baseline_thrust_N': row.get('thrust_N', 0.0),
                'baseline_tsfc_kg_N_s': row.get('tsfc_kg_N_s', 0.0),
                'cfd_thrust_N': cfd_metrics['thrust_N'],
                'cfd_tsfc_kg_N_s': cfd_metrics['tsfc_kg_N_s'],
            })
            with UI_CFD_LOCK:
                UI_CFD_STATUS['processed'] = idx + 1

        new_df = pd.DataFrame(samples)
        if os.path.exists(CFD_SAMPLES_CSV):
            existing = pd.read_csv(CFD_SAMPLES_CSV)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        dedupe_cols = [
            'solver', 'compressor_pressure_ratio', 'turbine_inlet_temp_K',
            'mass_flow_kg_s', 'nozzle_exit_diameter_mm',
            'combustor_outer_diameter_mm', 'combustor_inner_diameter_mm',
            'combustor_length_mm'
        ]
        combined = combined.drop_duplicates(subset=dedupe_cols, keep='last')

        os.makedirs(USER_LEARNING_DIR, exist_ok=True)
        combined.to_csv(CFD_SAMPLES_CSV, index=False)

        model = fit_calibration_from_samples(combined, solver=solver)
        save_calibration_model(model, CFD_CALIBRATION_JSON)

        with UI_CFD_LOCK:
            UI_CFD_STATUS['state'] = 'done'
            UI_CFD_STATUS['message'] = f'CFD calibration ready ({model.sample_count} samples)'
            UI_CFD_STATUS['finished_at'] = datetime.utcnow().isoformat() + 'Z'
            UI_CFD_STATUS['sample_count'] = int(model.sample_count)
    except Exception as e:
        with UI_CFD_LOCK:
            UI_CFD_STATUS['state'] = 'failed'
            UI_CFD_STATUS['message'] = str(e)
            UI_CFD_STATUS['finished_at'] = datetime.utcnow().isoformat() + 'Z'
            UI_CFD_STATUS['sample_count'] = 0


def _run_ui_training_job(epochs: int, use_cfd: bool = False):
    import pandas as pd
    from src.ai.surrogate import train_surrogate, SurrogateConfig, INPUT_FEATURES, OUTPUT_FEATURES
    from src.physics.cfd_calibration import apply_calibration_to_dataframe

    with UI_TRAIN_LOCK:
        UI_TRAIN_STATUS['state'] = 'running'
        UI_TRAIN_STATUS['message'] = 'Training started from UI samples'
        UI_TRAIN_STATUS['started_at'] = datetime.utcnow().isoformat() + 'Z'
        UI_TRAIN_STATUS['finished_at'] = None
        UI_TRAIN_STATUS['last_epochs'] = int(epochs)

    try:
        if not os.path.exists(UI_VARIANTS_CSV):
            raise RuntimeError("No UI samples found. Save at least one variant first.")

        ui_df = pd.read_csv(UI_VARIANTS_CSV)
        if ui_df.empty:
            raise RuntimeError("UI samples file is empty.")

        base_candidates = [
            os.path.join(PROJECT_ROOT, 'data', 'generated', 'dataset_10000.csv'),
            os.path.join(PROJECT_ROOT, 'data', 'generated', 'dataset_500.csv'),
        ]
        base_df = None
        for path in base_candidates:
            if os.path.exists(path):
                base_df = pd.read_csv(path)
                break
        if base_df is None:
            raise RuntimeError("Base dataset not found. Generate dataset first.")

        if 'valid' in base_df.columns and 'is_valid' not in base_df.columns:
            base_df['is_valid'] = base_df['valid']

        required_cols = INPUT_FEATURES + OUTPUT_FEATURES + ['is_valid']
        for col in required_cols:
            if col not in base_df.columns:
                base_df[col] = 0.0 if col != 'is_valid' else True
            if col not in ui_df.columns:
                ui_df[col] = 0.0 if col != 'is_valid' else True

        # Weight UI data more strongly by duplicating it.
        ui_weight = 8
        combined = pd.concat(
            [base_df[required_cols]] + [ui_df[required_cols]] * ui_weight,
            ignore_index=True
        )

        cfd_applied = False
        if use_cfd:
            cfd_model = _load_cfd_model_if_available()
            if cfd_model is not None:
                combined = apply_calibration_to_dataframe(combined, cfd_model)
                cfd_applied = True
            # If no CFD calibration exists, silently fall back to physics-only

        os.makedirs(USER_LEARNING_DIR, exist_ok=True)
        combined.to_csv(UI_TRAIN_DATASET_CSV, index=False)

        cfg = SurrogateConfig(epochs=int(epochs), patience=max(10, int(epochs) // 5))
        _, history = train_surrogate(
            UI_TRAIN_DATASET_CSV,
            config=cfg,
            save_dir=os.path.join(PROJECT_ROOT, 'data', 'trained_models'),
            verbose=False
        )

        with UI_TRAIN_LOCK:
            UI_TRAIN_STATUS['state'] = 'done'
            mode = "CFD-calibrated" if cfd_applied else "physics-only"
            UI_TRAIN_STATUS['message'] = f"Training complete ({mode}, epochs run: {len(history.get('train_loss', []))})"
            UI_TRAIN_STATUS['finished_at'] = datetime.utcnow().isoformat() + 'Z'
            UI_TRAIN_STATUS['variant_count'] = len(ui_df)
    except Exception as e:
        with UI_TRAIN_LOCK:
            UI_TRAIN_STATUS['state'] = 'failed'
            UI_TRAIN_STATUS['message'] = str(e)
            UI_TRAIN_STATUS['finished_at'] = datetime.utcnow().isoformat() + 'Z'
            UI_TRAIN_STATUS['variant_count'] = _get_ui_variant_count()


def _get_engine_metrics():
    """Get engine performance and geometry metrics."""
    try:
        from src.geometry.assembly import EngineAssemblyParams, assemble_engine, print_engine_summary
        params = EngineAssemblyParams()
        asm = assemble_engine(params)
        return {
            'name': params.name,
            'total_length_mm': round(asm.total_length_mm, 1),
            'max_diameter_mm': round(asm.max_diameter_mm, 1),
            'total_mass_g': round(asm.total_mass_kg * 1000, 1),
            'mass_breakdown': {k: round(v * 1000, 1) for k, v in asm.mass_breakdown.items()},
            'positions': {k: round(v, 1) for k, v in asm.component_positions.items()},
            'specs': {
                'pressure_ratio': params.compressor.pressure_ratio,
                'rpm': int(params.compressor.rpm),
                'tit_k': int(params.turbine.inlet_temperature_K),
                'tit_c': int(params.turbine.inlet_temperature_K - 273.15),
                'shaft_diameter_mm': params.shaft_diameter_mm,
            }
        }
    except Exception as e:
        return {'error': str(e)}


def _get_performance_data():
    """Get AI optimization results if available."""
    try:
        import pandas as pd
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'data', 'generated', 'dataset_10000.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            valid = df[df['valid'] == True] if 'valid' in df.columns else df
            return {
                'total_designs': len(df),
                'valid_designs': len(valid),
                'thrust_range': [round(valid['thrust_N'].min(), 1), round(valid['thrust_N'].max(), 1)],
                'tw_range': [round(valid['thrust_to_weight'].min(), 1), round(valid['thrust_to_weight'].max(), 1)],
                'efficiency_range': [round(valid['thermal_efficiency'].min() * 100, 1),
                                     round(valid['thermal_efficiency'].max() * 100, 1)],
                'mean_thrust': round(valid['thrust_N'].mean(), 1),
                'mean_tw': round(valid['thrust_to_weight'].mean(), 1),
            }
    except Exception:
        pass
    return None


@app.route('/')
def index():
    return render_template('viewer.html')


@app.route('/api/components')
def get_components():
    """Return list of available STL components with metadata."""
    available = []
    for name, meta in sorted(COMPONENTS.items(), key=lambda x: x[1]['order']):
        stl_path = os.path.join(STL_DIR, f'{name}.stl')
        if os.path.exists(stl_path):
            size_kb = os.path.getsize(stl_path) / 1024
            available.append({
                'id': name,
                'label': meta['label'],
                'color': meta['color'],
                'file': f'/stl/{name}.stl',
                'size_kb': round(size_kb, 1),
                'order': meta['order'],
            })
    return jsonify(available)


@app.route('/api/engine')
def get_engine_data():
    """Return engine metrics and performance data."""
    metrics = _get_engine_metrics()
    perf = _get_performance_data()
    return jsonify({
        'metrics': metrics,
        'performance': perf,
    })


@app.route('/api/simulation')
def get_simulation():
    """Run full simulation and return thermal + flow + stress data."""
    try:
        from src.physics.simulation import run_full_simulation
        return jsonify(run_full_simulation())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/brayton/compute')
def compute_brayton():
    """Run Brayton cycle with custom parameters. Returns station data for charts."""
    from src.physics.brayton import FlightConditions, EngineInputs, solve_brayton_cycle
    import math

    flight = FlightConditions(
        M_flight=float(request.args.get('mach', 0)),
        altitude_m=float(request.args.get('alt', 0)),
    )
    engine = EngineInputs(
        compressor_pressure_ratio=float(request.args.get('pr', 3.5)),
        compressor_isentropic_efficiency=float(request.args.get('eta_c', 0.78)),
        combustor_exit_temperature_K=float(request.args.get('tit', 1100)),
        turbine_isentropic_efficiency=float(request.args.get('eta_t', 0.82)),
        mass_flow_air_kg_s=float(request.args.get('mdot', 0.15)),
    )
    r = solve_brayton_cycle(flight, engine)

    # Build station arrays for charts
    station_keys = ['1_ambient', '2_inlet_exit', '3_compressor_exit',
                    '4_combustor_exit', '5_turbine_exit', '6_nozzle_exit']
    labels = ['Ambient', 'Inlet', 'Compressor', 'Combustor', 'Turbine', 'Nozzle']
    temps, pressures = [], []
    # Build T-s diagram data (entropy computation)
    ts_points = []
    s_cumulative = 0.0
    prev_T = None
    for i, sk in enumerate(station_keys):
        st = r.stations.get(sk)
        if not st:
            continue
        T = round(st.T_total_K, 1)
        P = round(st.P_total_Pa / 1000, 1)  # kPa
        temps.append(T)
        pressures.append(P)
        # Approximate entropy: ds = cp * ln(T2/T1) - R * ln(P2/P1)
        if prev_T is not None:
            prev_st = r.stations.get(station_keys[i - 1])
            cp = engine.cp_cold if i <= 2 else engine.cp_hot
            if prev_st and prev_st.T_total_K > 0 and prev_st.P_total_Pa > 0:
                ds = cp * math.log(st.T_total_K / prev_st.T_total_K) - \
                     engine.R_air * math.log(st.P_total_Pa / prev_st.P_total_Pa)
                s_cumulative += ds
        prev_T = st.T_total_K
        ts_points.append({'s': round(s_cumulative, 2), 'T': T, 'label': labels[i]})

    return jsonify({
        'labels': labels,
        'temperatures': temps,
        'pressures': pressures,
        'ts_diagram': ts_points,
        'performance': {
            'thrust_N': round(r.thrust_N, 1),
            'tsfc_g_kNs': round(r.tsfc_g_kN_s, 1),
            'thermal_eff': round(r.thermal_efficiency * 100, 1),
            'exhaust_vel': round(r.exhaust_velocity_m_s, 1),
            'exhaust_temp_K': round(r.exhaust_temperature_K, 1),
            'fuel_flow_g_hr': round(r.fuel_flow_kg_s * 3.6e6, 1),
            'compressor_power_kW': round(r.compressor_power_W / 1000, 2),
            'turbine_power_kW': round(r.turbine_power_W / 1000, 2),
            'specific_thrust': round(r.specific_thrust_N_s_kg, 1),
            'air_fuel_ratio': round(r.air_fuel_ratio, 1),
        },
        'warnings': r.warnings,
        'valid': r.is_valid,
    })


@app.route('/api/pareto')
def get_pareto():
    """Return Pareto-optimal designs from dataset for scatter plots."""
    import pandas as pd
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data', 'generated', 'dataset_10000.csv')
    if not os.path.exists(csv_path):
        return jsonify({'error': 'No dataset found'}), 404

    df = pd.read_csv(csv_path)
    valid = df[df['valid'] == True] if 'valid' in df.columns else df

    # Sample up to 500 points for the scatter plot
    sample = valid.sample(n=min(500, len(valid)), random_state=42)

    points = []
    for _, row in sample.iterrows():
        points.append({
            'thrust': round(float(row['thrust_N']), 1),
            'mass': round(float(row['total_mass_kg']), 3),
            'tw': round(float(row['thrust_to_weight']), 2),
            'eff': round(float(row['thermal_efficiency']) * 100, 1),
            'tsfc': round(float(row.get('tsfc_kg_N_s', 0)) * 1e6, 1),
            'pr': round(float(row.get('compressor_pressure_ratio', 0)), 2),
        })

    return jsonify({
        'points': points,
        'total_designs': len(valid),
        'stats': {
            'thrust_range': [round(float(valid['thrust_N'].min()), 1),
                             round(float(valid['thrust_N'].max()), 1)],
            'mass_range': [round(float(valid['total_mass_kg'].min()), 3),
                           round(float(valid['total_mass_kg'].max()), 3)],
            'tw_range': [round(float(valid['thrust_to_weight'].min()), 2),
                         round(float(valid['thrust_to_weight'].max()), 2)],
        }
    })


@app.route('/api/model/regenerate', methods=['POST'])
def regenerate_model():
    """
    Regenerate engine STL files from dashboard parameters.
    This enables live geometry tuning (nozzle/combustor) in the web UI.
    """
    try:
        from src.export.stl_export import export_full_engine
        from src.export.lattice_export import export_engine_lattice

        payload = request.get_json(silent=True) or {}
        params, asm, result, _controls = _build_engine_state_from_payload(payload)

        # Rebuild engine STL files (default paths consumed by viewer.js)
        export_full_engine(output_dir=STL_DIR, n_theta=72, verbose=False, assembly_params=params)

        # Keep default lattice in sync with geometry (non-fatal if lattice export fails)
        try:
            export_engine_lattice(
                output_dir=STL_DIR,
                verbose=False,
                variation='v1_gyroid_standard',
                assembly_params=params
            )
        except Exception:
            pass

        return jsonify({
            'success': True,
            'geometry': {
                'length_mm': round(asm.total_length_mm, 1),
                'max_diameter_mm': round(asm.max_diameter_mm, 1),
                'mass_g': round(asm.total_mass_kg * 1000, 1),
                'nozzle_exit_mm': round(params.nozzle.exit_diameter_mm, 1),
                'nozzle_length_mm': round(params.nozzle.length_mm, 1),
                'combustor_outer_mm': round(params.combustor.outer_diameter_mm, 1),
                'combustor_inner_mm': round(params.combustor.inner_diameter_mm, 1),
                'combustor_length_mm': round(params.combustor.length_mm, 1),
            },
            'performance': {
                'thrust_N': round(result.thrust_N, 1),
                'tsfc_g_kNs': round(result.tsfc_g_kN_s, 1),
                'thermal_eff': round(result.thermal_efficiency * 100, 1),
            },
            'warnings': result.warnings,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/learning/add_variant', methods=['POST'])
def add_learning_variant():
    """Save the current UI-adjusted design as a new training sample."""
    try:
        import pandas as pd
        payload = request.get_json(silent=True) or {}
        params, asm, result, controls = _build_engine_state_from_payload(payload)
        row = _build_ui_variant_record(params, asm, result, controls)

        os.makedirs(USER_LEARNING_DIR, exist_ok=True)
        row_df = pd.DataFrame([row])
        if os.path.exists(UI_VARIANTS_CSV):
            existing = pd.read_csv(UI_VARIANTS_CSV)
            combined = pd.concat([existing, row_df], ignore_index=True)
        else:
            combined = row_df
        combined.to_csv(UI_VARIANTS_CSV, index=False)

        count = len(combined)
        with UI_TRAIN_LOCK:
            UI_TRAIN_STATUS['variant_count'] = count

        return jsonify({
            'success': True,
            'variant_count': count,
            'sample': {
                'thrust_N': round(row['thrust_N'], 1),
                'tsfc_g_kNs': round(row['tsfc_kg_N_s'] * 1e6, 1),
                'mass_g': round(row['total_mass_kg'] * 1000, 1),
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/learning/train', methods=['POST'])
def train_from_ui_variants():
    """Start asynchronous surrogate retraining using UI-saved variants."""
    payload = request.get_json(silent=True) or {}
    epochs = int(_clamp(payload.get('epochs', 80), 80, 20, 300))
    use_cfd = str(payload.get('use_cfd', False)).strip().lower() in ('1', 'true', 'yes', 'on')

    with UI_TRAIN_LOCK:
        if UI_TRAIN_STATUS.get('state') == 'running':
            return jsonify({'error': 'Training already running'}), 409
        UI_TRAIN_STATUS['state'] = 'running'
        mode = "CFD-calibrated" if use_cfd else "physics-only"
        UI_TRAIN_STATUS['message'] = f"Queued {mode} training job ({epochs} epochs)"
        UI_TRAIN_STATUS['started_at'] = datetime.utcnow().isoformat() + 'Z'
        UI_TRAIN_STATUS['finished_at'] = None
        UI_TRAIN_STATUS['last_epochs'] = epochs
        UI_TRAIN_STATUS['variant_count'] = _get_ui_variant_count()

    t = threading.Thread(target=_run_ui_training_job, args=(epochs, use_cfd), daemon=True)
    t.start()

    return jsonify({
        'success': True,
        'state': 'running',
        'epochs': epochs,
        'use_cfd': use_cfd,
        'variant_count': _get_ui_variant_count()
    })


@app.route('/api/learning/status')
def learning_status():
    """Return current UI-learning status and number of saved samples."""
    with UI_TRAIN_LOCK:
        status = dict(UI_TRAIN_STATUS)
    with UI_CFD_LOCK:
        cfd_status = dict(UI_CFD_STATUS)
    status['variant_count'] = _get_ui_variant_count()
    status['dataset_path'] = UI_TRAIN_DATASET_CSV if os.path.exists(UI_TRAIN_DATASET_CSV) else None
    status['cfd'] = cfd_status
    status['has_cfd_calibration'] = os.path.exists(CFD_CALIBRATION_JSON)
    return jsonify(status)


@app.route('/api/learning/cfd/start', methods=['POST'])
def start_cfd_calibration():
    """Start asynchronous CFD calibration from saved UI variants."""
    payload = request.get_json(silent=True) or {}
    solver = str(payload.get('solver', 'openfoam')).strip().lower()
    if solver not in ('openfoam', 'su2'):
        return jsonify({'error': "solver must be 'openfoam' or 'su2'"}), 400

    max_cases = int(_clamp(payload.get('max_cases', 5), 5, 1, 200))
    timeout_s = int(_clamp(payload.get('timeout_s', 900), 900, 60, 3600))

    with UI_CFD_LOCK:
        if UI_CFD_STATUS.get('state') == 'running':
            return jsonify({'error': 'CFD calibration already running'}), 409
        UI_CFD_STATUS['state'] = 'running'
        UI_CFD_STATUS['message'] = f'Queued {solver} CFD calibration'
        UI_CFD_STATUS['solver'] = solver
        UI_CFD_STATUS['started_at'] = datetime.utcnow().isoformat() + 'Z'
        UI_CFD_STATUS['finished_at'] = None
        UI_CFD_STATUS['processed'] = 0

    t = threading.Thread(
        target=_run_cfd_calibration_job,
        args=(solver, max_cases, timeout_s),
        daemon=True
    )
    t.start()

    return jsonify({
        'success': True,
        'state': 'running',
        'solver': solver,
        'max_cases': max_cases,
        'timeout_s': timeout_s,
    })


@app.route('/api/learning/cfd/status')
def cfd_learning_status():
    """Return CFD calibration status."""
    with UI_CFD_LOCK:
        status = dict(UI_CFD_STATUS)
    status['has_cfd_calibration'] = os.path.exists(CFD_CALIBRATION_JSON)
    status['calibration_path'] = CFD_CALIBRATION_JSON if os.path.exists(CFD_CALIBRATION_JSON) else None
    return jsonify(status)


@app.route('/api/inverse_design', methods=['POST'])
def inverse_design():
    """Inverse-design search: target thrust/TSFC -> geometry parameters."""
    import numpy as np
    from src.ai.dataset import DesignSpaceBounds, evaluate_design
    from src.ai.surrogate import load_surrogate
    from src.physics.cfd_calibration import apply_calibration_to_metrics

    payload = request.get_json(silent=True) or {}
    target_thrust = float(_clamp(payload.get('target_thrust_N', 100.0), 100.0, 1.0, 500.0))
    target_tsfc_g = float(_clamp(payload.get('target_tsfc_g_kNs', 35.0), 35.0, 1.0, 500.0))
    target_tsfc = target_tsfc_g / 1e6
    n_samples = int(_clamp(payload.get('n_samples', 2000), 2000, 200, 20000))
    top_k = int(_clamp(payload.get('top_k', 5), 5, 1, 20))

    bounds = DesignSpaceBounds()
    rng = np.random.default_rng(int(payload.get('seed', 42)))

    model = None
    model_path = os.path.join(PROJECT_ROOT, 'data', 'trained_models', 'best_surrogate.pt')
    if os.path.exists(model_path):
        model = load_surrogate(model_path)

    cfd_model = _load_cfd_model_if_available()

    def sample_params():
        sampled = {}
        for field_name in bounds.__dataclass_fields__:
            lo, hi = getattr(bounds, field_name)
            v = float(lo + (hi - lo) * rng.random())
            if field_name in ('compressor_blade_count', 'combustor_num_injectors', 'turbine_blade_count'):
                v = int(round(v))
            sampled[field_name] = v
        return sampled

    candidates = []
    for _ in range(n_samples):
        params = sample_params()

        if model is not None:
            perf = model.predict(params)
            perf['mass_flow_kg_s'] = params['mass_flow_kg_s']
        else:
            import pandas as pd
            perf = evaluate_design(pd.Series(params))

        perf = apply_calibration_to_metrics(perf, cfd_model)
        thrust = float(perf.get('thrust_N', 0.0))
        tsfc = float(perf.get('tsfc_kg_N_s', 0.0))
        score = abs(thrust - target_thrust) / max(target_thrust, 1.0) + \
                abs(tsfc - target_tsfc) / max(target_tsfc, 1e-10)

        candidates.append({
            'score': score,
            'params': params,
            'performance': {
                'thrust_N': thrust,
                'tsfc_g_kNs': tsfc * 1e6,
                'thermal_efficiency': float(perf.get('thermal_efficiency', 0.0)) * 100.0,
            }
        })

    candidates.sort(key=lambda c: c['score'])
    top = candidates[:top_k]
    best = top[0] if top else None

    # UI-mapped fields for one-click application
    ui_params = None
    if best is not None:
        p = best['params']
        ui_params = {
            'pr': round(float(p['compressor_pressure_ratio']), 2),
            'tit': round(float(p['turbine_inlet_temp_K']), 1),
            'eta_c': round(float(p['compressor_efficiency']), 3),
            'eta_t': round(float(p['turbine_efficiency']), 3),
            'mdot': round(float(p['mass_flow_kg_s']), 3),
            'nozzle_exit_mm': round(float(p['nozzle_exit_diameter_mm']), 1),
            'combustor_outer_mm': round(float(p['combustor_outer_diameter_mm']), 1),
            'combustor_inner_mm': round(float(p['combustor_inner_diameter_mm']), 1),
            'combustor_length_mm': round(float(p['combustor_length_mm']), 1),
        }

    return jsonify({
        'success': True,
        'target': {
            'thrust_N': target_thrust,
            'tsfc_g_kNs': target_tsfc_g,
        },
        'used_model': 'surrogate' if model is not None else 'physics',
        'used_cfd_calibration': cfd_model is not None,
        'best': best,
        'best_ui_params': ui_params,
        'candidates': top,
    })


@app.route('/api/lattice/info')
def get_lattice_info():
    """Return lattice structure metadata including available variations."""
    from src.export.lattice_export import LATTICE_VARIATIONS

    lattice_files = {}
    for name in ['combustor_lattice', 'nozzle_lattice']:
        stl_path = os.path.join(STL_DIR, f'{name}.stl')
        if os.path.exists(stl_path):
            size_mb = os.path.getsize(stl_path) / (1024 * 1024)
            lattice_files[name] = {
                'exists': True,
                'size_mb': round(size_mb, 1),
                'file': f'/stl/{name}.stl',
            }
        else:
            lattice_files[name] = {'exists': False}

    # Build variations list with availability
    variations = {}
    for key, var in LATTICE_VARIATIONS.items():
        var_dir = os.path.join(STL_DIR, key)
        available = (os.path.exists(os.path.join(var_dir, 'combustor_lattice.stl')) or
                     os.path.exists(os.path.join(var_dir, 'nozzle_lattice.stl')))
        variations[key] = {
            'label': var['label'],
            'description': var['description'],
            'combustor_type': var['combustor']['tpms_type'],
            'nozzle_type': var['nozzle']['tpms_type'],
            'combustor_cell': var['combustor']['cell_size_mm'],
            'nozzle_cell': var['nozzle']['cell_size_mm'],
            'available': available,
        }

    return jsonify({
        'lattice_files': lattice_files,
        'variations': variations,
        'description': 'TPMS (Triply Periodic Minimal Surface) lattice infill',
        'benefits': [
            'Up to 60% weight reduction in liner walls',
            'Maximized surface area for heat transfer',
            'Self-supporting geometry for metal 3D printing',
            'Graded density: thicker near hot zones',
        ]
    })


@app.route('/api/lattice/generate')
def generate_lattice():
    """Generate lattice STL files if they don't exist."""
    try:
        from src.export.lattice_export import export_engine_lattice
        exported = export_engine_lattice(output_dir=STL_DIR, verbose=False)
        return jsonify({
            'success': True,
            'files': {k: os.path.basename(v) for k, v in exported.items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stl/variation/<variation>/<filename>')
def serve_variation_stl(variation, filename):
    """Serve lattice STL files for a specific variation."""
    var_dir = os.path.join(STL_DIR, variation)
    if not os.path.exists(os.path.join(var_dir, filename)):
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory(var_dir, filename)


@app.route('/stl/<filename>')
def serve_stl(filename):
    """Serve STL files."""
    return send_from_directory(STL_DIR, filename)


if __name__ == '__main__':
    print("\n  NovaTurbo 3D Viewer")
    print(f"  STL directory: {STL_DIR}")
    stl_count = len([f for f in os.listdir(STL_DIR) if f.endswith('.stl')]) if os.path.exists(STL_DIR) else 0
    print(f"  Found {stl_count} STL files")
    print(f"\n  Open: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
