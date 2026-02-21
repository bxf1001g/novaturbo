"""
NovaTurbo — Brayton Cycle Validation Against Published Micro Turbojet Data

Validates our thermodynamic solver against real-world engine test data
from published sources (JetCat P80, P100, P200 and academic references).

Sources:
  - JetCat official datasheets (jetcat.de)
  - Catana et al., "Thermodynamic Analysis of Microjet Engines", MDPI Applied Sciences 2024
  - Vouros et al., MDPI Energies 2025
  - Oklahoma State capstone project (JetCat P100)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from src.physics.brayton import (
    FlightConditions, EngineInputs, CycleResults, solve_brayton_cycle
)


# ─── Published engine reference data ────────────────────────────────────────

@dataclass
class ReferenceEngine:
    """Published performance data for a real micro turbojet."""
    name: str
    source: str

    # Input parameters (what we feed our solver)
    mass_flow_kg_s: float
    compressor_pressure_ratio: float
    compressor_efficiency: float
    turbine_efficiency: float
    combustor_exit_temp_K: float       # TIT
    combustor_pressure_drop: float
    rpm: float

    # Physical dimensions
    diameter_mm: float
    length_mm: float
    weight_kg: float

    # Published performance (ground truth)
    published_thrust_N: float
    published_tsfc_kg_kgf_h: float     # kg/(kgf·h) — standard micro turbojet unit
    published_fuel_flow_ml_min: float
    published_exhaust_temp_C: float    # EGT (after turbine)
    published_exhaust_temp_K: float = 0.0
    published_tsfc_kg_N_s: float = 0.0  # converted

    def __post_init__(self):
        if self.published_exhaust_temp_K == 0.0:
            self.published_exhaust_temp_K = self.published_exhaust_temp_C + 273.15
        # Convert kg/(kgf·h) → kg/(N·s): divide by 9.81 to get kg/(N·h), then /3600 for per-second
        self.published_tsfc_kg_N_s = self.published_tsfc_kg_kgf_h / 9.81 / 3600.0


# Published data from JetCat datasheets + academic papers
REFERENCE_ENGINES: List[ReferenceEngine] = [
    ReferenceEngine(
        name="JetCat P80-SE",
        source="JetCat datasheet + Catana et al. 2024",
        mass_flow_kg_s=0.16,
        compressor_pressure_ratio=3.6,
        compressor_efficiency=0.68,
        turbine_efficiency=0.78,
        combustor_exit_temp_K=1150.0,      # TIT estimate: EGT 700°C + ~180K turbine delta
        combustor_pressure_drop=0.04,
        rpm=125000,
        diameter_mm=112,
        length_mm=315,
        weight_kg=1.36,
        published_thrust_N=80.0,
        published_tsfc_kg_kgf_h=1.2,
        published_fuel_flow_ml_min=275,
        published_exhaust_temp_C=700,
    ),
    ReferenceEngine(
        name="JetCat P100-RX",
        source="JetCat datasheet + OU Capstone 2024",
        mass_flow_kg_s=0.20,
        compressor_pressure_ratio=2.9,
        compressor_efficiency=0.70,
        turbine_efficiency=0.80,
        combustor_exit_temp_K=1100.0,      # TIT estimate: EGT 720°C + ~110K turbine delta
        combustor_pressure_drop=0.04,
        rpm=152000,
        diameter_mm=112,
        length_mm=282,
        weight_kg=1.08,
        published_thrust_N=100.0,
        published_tsfc_kg_kgf_h=1.12,
        published_fuel_flow_ml_min=300,
        published_exhaust_temp_C=720,
    ),
    ReferenceEngine(
        name="JetCat P200-SX",
        source="JetCat manual + Vouros et al. 2025",
        mass_flow_kg_s=0.38,
        compressor_pressure_ratio=3.4,
        compressor_efficiency=0.73,
        turbine_efficiency=0.80,
        combustor_exit_temp_K=1150.0,      # TIT estimate: EGT 750°C + ~130K turbine delta
        combustor_pressure_drop=0.04,
        rpm=112000,
        diameter_mm=130,
        length_mm=395,
        weight_kg=2.65,
        published_thrust_N=222.0,
        published_tsfc_kg_kgf_h=1.15,
        published_fuel_flow_ml_min=600,
        published_exhaust_temp_C=750,
    ),
    ReferenceEngine(
        name="AMT Olympus HP",
        source="AMT Netherlands datasheet",
        mass_flow_kg_s=0.40,
        compressor_pressure_ratio=3.8,
        compressor_efficiency=0.74,
        turbine_efficiency=0.82,
        combustor_exit_temp_K=1200.0,      # TIT estimate: EGT 800°C + ~130K turbine delta
        combustor_pressure_drop=0.05,
        rpm=108000,
        diameter_mm=130,
        length_mm=377,
        weight_kg=2.30,
        published_thrust_N=230.0,
        published_tsfc_kg_kgf_h=1.08,
        published_fuel_flow_ml_min=630,
        published_exhaust_temp_C=800,
    ),
]


@dataclass
class ValidationResult:
    """Comparison of our solver vs published data for one engine."""
    engine_name: str
    source: str

    # Our solver predictions
    predicted_thrust_N: float = 0.0
    predicted_exhaust_temp_K: float = 0.0
    predicted_fuel_flow_kg_s: float = 0.0
    predicted_tsfc_kg_kgf_h: float = 0.0

    # Published ground truth
    published_thrust_N: float = 0.0
    published_exhaust_temp_K: float = 0.0
    published_fuel_flow_ml_min: float = 0.0
    published_tsfc_kg_kgf_h: float = 0.0

    # Error metrics
    thrust_error_pct: float = 0.0
    exhaust_temp_error_pct: float = 0.0
    tsfc_error_pct: float = 0.0
    fuel_flow_error_pct: float = 0.0

    # Solver output
    cycle_results: Optional[CycleResults] = None
    is_valid: bool = True
    notes: List[str] = field(default_factory=list)


def validate_against_reference(ref: ReferenceEngine) -> ValidationResult:
    """Run our Brayton cycle solver with reference engine inputs and compare."""
    vr = ValidationResult(
        engine_name=ref.name,
        source=ref.source,
        published_thrust_N=ref.published_thrust_N,
        published_exhaust_temp_K=ref.published_exhaust_temp_K,
        published_fuel_flow_ml_min=ref.published_fuel_flow_ml_min,
        published_tsfc_kg_kgf_h=ref.published_tsfc_kg_kgf_h,
    )

    flight = FlightConditions(T_ambient_K=288.15, P_ambient_Pa=101325, M_flight=0.0)

    engine = EngineInputs(
        compressor_pressure_ratio=ref.compressor_pressure_ratio,
        compressor_isentropic_efficiency=ref.compressor_efficiency,
        turbine_isentropic_efficiency=ref.turbine_efficiency,
        combustor_exit_temperature_K=ref.combustor_exit_temp_K,
        combustor_pressure_drop_fraction=ref.combustor_pressure_drop,
        mass_flow_air_kg_s=ref.mass_flow_kg_s,
        combustion_efficiency=0.95,
        inlet_pressure_recovery=0.98,
        nozzle_discharge_coefficient=0.98,
        nozzle_isentropic_efficiency=0.95,
    )

    try:
        results = solve_brayton_cycle(flight, engine)
        vr.cycle_results = results
        vr.is_valid = results.is_valid

        # Extract predictions
        vr.predicted_thrust_N = results.thrust_N
        vr.predicted_exhaust_temp_K = results.exhaust_temperature_K
        vr.predicted_fuel_flow_kg_s = results.fuel_flow_kg_s
        # Convert TSFC: our solver gives kg/(N·s) → convert to kg/(kgf·h)
        # kg/(N·s) × 3600 s/h × 9.81 N/kgf = kg/(kgf·h)
        vr.predicted_tsfc_kg_kgf_h = results.tsfc_kg_N_s * 3600.0 * 9.81

        # Compute errors
        if ref.published_thrust_N > 0:
            vr.thrust_error_pct = (
                (vr.predicted_thrust_N - ref.published_thrust_N)
                / ref.published_thrust_N * 100
            )
        if ref.published_exhaust_temp_K > 0:
            vr.exhaust_temp_error_pct = (
                (vr.predicted_exhaust_temp_K - ref.published_exhaust_temp_K)
                / ref.published_exhaust_temp_K * 100
            )
        if ref.published_tsfc_kg_kgf_h > 0:
            vr.tsfc_error_pct = (
                (vr.predicted_tsfc_kg_kgf_h - ref.published_tsfc_kg_kgf_h)
                / ref.published_tsfc_kg_kgf_h * 100
            )

        # Fuel flow: convert our kg/s to ml/min (Jet-A density ~800 kg/m³)
        fuel_ml_min = vr.predicted_fuel_flow_kg_s / 0.8 * 1e3 * 60  # kg/s → ml/min
        if ref.published_fuel_flow_ml_min > 0:
            vr.fuel_flow_error_pct = (
                (fuel_ml_min - ref.published_fuel_flow_ml_min)
                / ref.published_fuel_flow_ml_min * 100
            )

        # Add notes for large deviations
        if abs(vr.thrust_error_pct) > 20:
            vr.notes.append(f"Thrust error {vr.thrust_error_pct:+.1f}% — consider adjusting mass flow or TIT estimate")
        if abs(vr.exhaust_temp_error_pct) > 15:
            vr.notes.append(f"EGT error {vr.exhaust_temp_error_pct:+.1f}% — turbine efficiency or TIT estimate may be off")
        if abs(vr.tsfc_error_pct) > 20:
            vr.notes.append(f"TSFC error {vr.tsfc_error_pct:+.1f}% — combustion model needs calibration")

    except Exception as e:
        vr.is_valid = False
        vr.notes.append(f"Solver failed: {str(e)}")

    return vr


def run_full_validation() -> Dict:
    """Validate against all reference engines and return summary."""
    results = []
    for ref in REFERENCE_ENGINES:
        vr = validate_against_reference(ref)
        results.append(vr)

    # Compute aggregate metrics
    valid_results = [r for r in results if r.is_valid]
    if valid_results:
        avg_thrust_err = np.mean([abs(r.thrust_error_pct) for r in valid_results])
        avg_temp_err = np.mean([abs(r.exhaust_temp_error_pct) for r in valid_results])
        avg_tsfc_err = np.mean([abs(r.tsfc_error_pct) for r in valid_results])
        max_thrust_err = max(abs(r.thrust_error_pct) for r in valid_results)
    else:
        avg_thrust_err = avg_temp_err = avg_tsfc_err = max_thrust_err = float('nan')

    summary = {
        'engine_count': len(REFERENCE_ENGINES),
        'valid_count': len(valid_results),
        'avg_thrust_error_pct': round(avg_thrust_err, 1),
        'avg_exhaust_temp_error_pct': round(avg_temp_err, 1),
        'avg_tsfc_error_pct': round(avg_tsfc_err, 1),
        'max_thrust_error_pct': round(max_thrust_err, 1),
        'grade': _compute_grade(avg_thrust_err, avg_tsfc_err),
        'engines': []
    }

    for vr in results:
        summary['engines'].append({
            'name': vr.engine_name,
            'source': vr.source,
            'valid': vr.is_valid,
            'predicted': {
                'thrust_N': round(vr.predicted_thrust_N, 1),
                'exhaust_temp_K': round(vr.predicted_exhaust_temp_K, 1),
                'tsfc_kg_kgf_h': round(vr.predicted_tsfc_kg_kgf_h, 3),
            },
            'published': {
                'thrust_N': vr.published_thrust_N,
                'exhaust_temp_K': round(vr.published_exhaust_temp_K, 1),
                'tsfc_kg_kgf_h': vr.published_tsfc_kg_kgf_h,
            },
            'errors': {
                'thrust_pct': round(vr.thrust_error_pct, 1),
                'exhaust_temp_pct': round(vr.exhaust_temp_error_pct, 1),
                'tsfc_pct': round(vr.tsfc_error_pct, 1),
                'fuel_flow_pct': round(vr.fuel_flow_error_pct, 1),
            },
            'notes': vr.notes,
        })

    return summary


def _compute_grade(avg_thrust_err: float, avg_tsfc_err: float) -> str:
    """Grade the solver accuracy: A (<5%), B (<10%), C (<20%), D (<30%), F (>30%)."""
    combined = (avg_thrust_err + avg_tsfc_err) / 2
    if combined < 5:
        return 'A'
    elif combined < 10:
        return 'B'
    elif combined < 20:
        return 'C'
    elif combined < 30:
        return 'D'
    else:
        return 'F'


def print_validation_report(summary: Dict):
    """Print formatted validation report to console."""
    print(f"\n{'='*70}")
    print(f"  NovaTurbo Brayton Cycle Validation Report")
    print(f"{'='*70}")
    print(f"  Engines tested: {summary['engine_count']}  |  Valid: {summary['valid_count']}")
    print(f"  Overall Grade: {summary['grade']}")
    print(f"  Avg Thrust Error: {summary['avg_thrust_error_pct']}%")
    print(f"  Avg TSFC Error:   {summary['avg_tsfc_error_pct']}%")
    print(f"  Avg EGT Error:    {summary['avg_exhaust_temp_error_pct']}%")
    print(f"{'─'*70}")

    for eng in summary['engines']:
        status = '✓' if eng['valid'] else '✗'
        print(f"\n  [{status}] {eng['name']} ({eng['source']})")
        print(f"  {'Metric':<20s} {'Predicted':>12s} {'Published':>12s} {'Error':>10s}")
        print(f"  {'─'*54}")
        print(f"  {'Thrust (N)':<20s} {eng['predicted']['thrust_N']:>12.1f} "
              f"{eng['published']['thrust_N']:>12.1f} {eng['errors']['thrust_pct']:>+9.1f}%")
        print(f"  {'EGT (K)':<20s} {eng['predicted']['exhaust_temp_K']:>12.1f} "
              f"{eng['published']['exhaust_temp_K']:>12.1f} {eng['errors']['exhaust_temp_pct']:>+9.1f}%")
        print(f"  {'TSFC (kg/kgf·h)':<20s} {eng['predicted']['tsfc_kg_kgf_h']:>12.3f} "
              f"{eng['published']['tsfc_kg_kgf_h']:>12.3f} {eng['errors']['tsfc_pct']:>+9.1f}%")
        if eng['notes']:
            for note in eng['notes']:
                print(f"  ⚠ {note}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    summary = run_full_validation()
    print_validation_report(summary)
