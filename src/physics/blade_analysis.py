"""
NovaTurbo — Blade Profile Aerodynamic Analysis

XFLR5-inspired analysis for compressor and turbine blade sections:
- Velocity triangles (absolute/relative frames)
- Blade loading (pressure distribution along chord)
- Loss estimation (profile, secondary, tip clearance)
- Diffusion factor & de Haller number (compressor)
- Zweifel coefficient (turbine)
- Work coefficient & flow coefficient

References:
  - Dixon & Hall, "Fluid Mechanics and Thermodynamics of Turbomachinery"
  - Aungier, "Centrifugal Compressors: A Strategy for Aerodynamic Design"
  - Lieblein, "Loss and Stall Analysis of Compressor Cascades"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─── Velocity Triangles ─────────────────────────────────────────────────────

@dataclass
class VelocityTriangle:
    """Velocity triangle at a station (absolute + relative frames)."""
    # Absolute frame
    C_axial: float = 0.0       # Axial velocity (m/s)
    C_tangential: float = 0.0  # Tangential (swirl) velocity (m/s)
    C: float = 0.0             # Absolute velocity magnitude (m/s)
    alpha_deg: float = 0.0     # Absolute flow angle (deg from axial)

    # Relative frame
    W_axial: float = 0.0       # = C_axial (axial component same)
    W_tangential: float = 0.0  # Relative tangential velocity
    W: float = 0.0             # Relative velocity magnitude
    beta_deg: float = 0.0      # Relative flow angle (deg from axial)

    # Blade speed
    U: float = 0.0             # Blade speed at this radius (m/s)


def compute_velocity_triangle(C_axial: float, C_tangential: float,
                               U: float) -> VelocityTriangle:
    """Compute full velocity triangle given axial vel, swirl, and blade speed."""
    vt = VelocityTriangle()
    vt.C_axial = C_axial
    vt.C_tangential = C_tangential
    vt.U = U

    # Absolute
    vt.C = np.sqrt(C_axial**2 + C_tangential**2)
    vt.alpha_deg = np.degrees(np.arctan2(C_tangential, C_axial))

    # Relative: W_theta = C_theta - U
    vt.W_axial = C_axial
    vt.W_tangential = C_tangential - U
    vt.W = np.sqrt(vt.W_axial**2 + vt.W_tangential**2)
    vt.beta_deg = np.degrees(np.arctan2(vt.W_tangential, vt.W_axial))

    return vt


# ─── Compressor Blade Analysis ──────────────────────────────────────────────

@dataclass
class CompressorBladeAnalysis:
    """Aerodynamic analysis of a compressor blade section."""
    # Velocity triangles
    inlet: VelocityTriangle = field(default_factory=VelocityTriangle)
    outlet: VelocityTriangle = field(default_factory=VelocityTriangle)

    # Performance metrics
    diffusion_factor: float = 0.0       # Lieblein D-factor (should be < 0.6)
    de_haller_number: float = 0.0       # W2/W1 (should be > 0.72)
    work_coefficient: float = 0.0       # ΔC_theta * U / U² = Δh / U²
    flow_coefficient: float = 0.0       # C_axial / U
    stage_loading: float = 0.0          # Δh₀ / U²
    degree_of_reaction: float = 0.0     # Fraction of pressure rise in rotor

    # Loss breakdown
    profile_loss_coeff: float = 0.0     # ω_p (blade boundary layer)
    secondary_loss_coeff: float = 0.0   # ω_s (endwall)
    tip_clearance_loss: float = 0.0     # ω_tc (tip leakage)
    total_loss_coeff: float = 0.0       # ω_total = ω_p + ω_s + ω_tc

    # Blade surface pressure distribution
    chord_stations: np.ndarray = field(default_factory=lambda: np.array([]))
    Cp_suction: np.ndarray = field(default_factory=lambda: np.array([]))
    Cp_pressure: np.ndarray = field(default_factory=lambda: np.array([]))

    # Ratings
    stall_risk: str = "low"
    efficiency_estimate: float = 0.0
    notes: List[str] = field(default_factory=list)


def analyze_compressor_blade(
    C_axial: float = 150.0,
    rpm: float = 100000,
    r_mean_mm: float = 40.0,
    beta1_deg: float = 55.0,
    beta2_deg: float = 30.0,
    chord_mm: float = 15.0,
    pitch_mm: float = 20.0,
    blade_thickness_ratio: float = 0.08,
    tip_clearance_mm: float = 0.3,
    blade_height_mm: float = 12.0,
) -> CompressorBladeAnalysis:
    """
    Analyze a compressor blade section at mean radius.

    Parameters:
        C_axial: Axial velocity through compressor (m/s)
        rpm: Rotational speed
        r_mean_mm: Mean blade radius (mm)
        beta1_deg: Relative inlet flow angle (deg from axial)
        beta2_deg: Relative outlet flow angle (deg from axial)
        chord_mm: Blade chord length (mm)
        pitch_mm: Blade pitch/spacing (mm)
        blade_thickness_ratio: Max thickness / chord
        tip_clearance_mm: Tip gap (mm)
        blade_height_mm: Blade span (mm)
    """
    result = CompressorBladeAnalysis()

    # Blade speed at mean radius
    U = (rpm * 2 * np.pi / 60) * (r_mean_mm / 1000.0)

    # Inlet velocity triangle
    # From relative angle β1: W_theta1 = C_axial * tan(β1)
    beta1 = np.radians(beta1_deg)
    beta2 = np.radians(beta2_deg)
    W_theta1 = C_axial * np.tan(beta1)
    C_theta1 = W_theta1 + U
    result.inlet = compute_velocity_triangle(C_axial, C_theta1, U)

    # Outlet velocity triangle
    W_theta2 = C_axial * np.tan(beta2)
    C_theta2 = W_theta2 + U
    result.outlet = compute_velocity_triangle(C_axial, C_theta2, U)

    # Solidity
    sigma = chord_mm / pitch_mm

    # ── Performance Metrics ──
    W1 = result.inlet.W
    W2 = result.outlet.W

    # De Haller number (W2/W1 > 0.72 for acceptable diffusion)
    result.de_haller_number = W2 / W1 if W1 > 0 else 0

    # Lieblein diffusion factor
    delta_W_theta = abs(W_theta1 - W_theta2)
    result.diffusion_factor = 1 - W2 / W1 + delta_W_theta / (2 * sigma * W1) if W1 > 0 else 0

    # Work and flow coefficients
    delta_C_theta = C_theta2 - C_theta1
    result.work_coefficient = delta_C_theta * U / U**2 if U > 0 else 0
    result.flow_coefficient = C_axial / U if U > 0 else 0

    # Stage loading
    delta_h0 = U * delta_C_theta  # Euler work (J/kg)
    result.stage_loading = delta_h0 / U**2 if U > 0 else 0

    # Degree of reaction
    result.degree_of_reaction = 1 - delta_C_theta / (2 * U) if U > 0 else 0.5

    # ── Loss Estimation (Lieblein-based correlations) ──

    # Profile loss: ω_p = 2θ/c × (σ cos²β₂)/(cos βm)
    # Simplified Lieblein momentum thickness correlation
    beta_m = np.arctan(0.5 * (np.tan(beta1) + np.tan(beta2)))
    D = result.diffusion_factor

    # Momentum thickness ratio from D-factor (Lieblein correlation)
    if D < 0.6:
        theta_c = 0.004 * (1 + 3.1 * D**2 + 0.4 * D**3)
    else:
        theta_c = 0.004 * (1 + 3.1 * D**2 + 0.4 * D**3) * (1 + 2 * (D - 0.6))

    result.profile_loss_coeff = 2 * theta_c * sigma * np.cos(beta2)**2 / np.cos(beta_m)

    # Secondary loss (Ainley-Mathieson style)
    AR = blade_height_mm / chord_mm  # aspect ratio
    result.secondary_loss_coeff = 0.018 * sigma / AR * (np.cos(beta2) / np.cos(beta1))**2

    # Tip clearance loss
    tc_ratio = tip_clearance_mm / blade_height_mm
    result.tip_clearance_loss = 0.29 * tc_ratio * np.sqrt(abs(np.tan(beta1) - np.tan(beta2)))

    result.total_loss_coeff = (result.profile_loss_coeff +
                                result.secondary_loss_coeff +
                                result.tip_clearance_loss)

    # Efficiency estimate from total loss
    result.efficiency_estimate = 1 - result.total_loss_coeff * 1.5  # Rough conversion

    # ── Blade Surface Pressure Distribution ──
    n_pts = 50
    s = np.linspace(0, 1, n_pts)  # chord fraction
    result.chord_stations = s

    # Pressure side: monotonic deceleration
    result.Cp_pressure = 1 - (1 - 0.3 * s) * (1 - D * np.sin(np.pi * s)**0.5)

    # Suction side: acceleration then diffusion (prone to separation)
    peak_loc = 0.3  # peak suction at 30% chord
    suction_peak = -(0.5 + D * 1.5)
    result.Cp_suction = np.where(
        s < peak_loc,
        1 + (suction_peak - 1) * (s / peak_loc)**0.7,
        suction_peak + (1.0 - suction_peak) * np.clip((s - peak_loc) / (1 - peak_loc), 0, 1)**1.5
    )

    # ── Stall Risk Assessment ──
    if result.de_haller_number < 0.72:
        result.stall_risk = "high"
        result.notes.append(f"De Haller {result.de_haller_number:.2f} < 0.72 — high stall risk!")
    elif result.de_haller_number < 0.78:
        result.stall_risk = "moderate"
        result.notes.append(f"De Haller {result.de_haller_number:.2f} — moderate stall margin")
    else:
        result.stall_risk = "low"

    if result.diffusion_factor > 0.6:
        result.notes.append(f"D-factor {result.diffusion_factor:.2f} > 0.6 — boundary layer separation likely")
    elif result.diffusion_factor > 0.45:
        result.notes.append(f"D-factor {result.diffusion_factor:.2f} — near design limit")

    if result.degree_of_reaction < 0.3:
        result.notes.append(f"Reaction {result.degree_of_reaction:.2f} — impulse-heavy, low efficiency")
    elif result.degree_of_reaction > 0.7:
        result.notes.append(f"Reaction {result.degree_of_reaction:.2f} — rotor does most work, check stall")

    return result


# ─── Turbine Blade Analysis ─────────────────────────────────────────────────

@dataclass
class TurbineBladeAnalysis:
    """Aerodynamic analysis of a turbine blade section."""
    # Velocity triangles
    ngv_inlet: VelocityTriangle = field(default_factory=VelocityTriangle)
    ngv_outlet: VelocityTriangle = field(default_factory=VelocityTriangle)
    rotor_inlet: VelocityTriangle = field(default_factory=VelocityTriangle)
    rotor_outlet: VelocityTriangle = field(default_factory=VelocityTriangle)

    # Performance metrics
    zweifel_coefficient: float = 0.0    # Blade loading (0.7-1.0 optimal)
    work_coefficient: float = 0.0       # Δh₀ / U²
    flow_coefficient: float = 0.0       # C_axial / U
    degree_of_reaction: float = 0.0
    stage_loading: float = 0.0

    # Loss breakdown
    ngv_loss_coeff: float = 0.0
    rotor_profile_loss: float = 0.0
    rotor_secondary_loss: float = 0.0
    rotor_tip_clearance_loss: float = 0.0
    total_loss_coeff: float = 0.0

    # Blade loading diagram
    chord_stations: np.ndarray = field(default_factory=lambda: np.array([]))
    Cp_suction: np.ndarray = field(default_factory=lambda: np.array([]))
    Cp_pressure: np.ndarray = field(default_factory=lambda: np.array([]))

    # Ratings
    efficiency_estimate: float = 0.0
    notes: List[str] = field(default_factory=list)


def analyze_turbine_blade(
    C_axial: float = 200.0,
    rpm: float = 100000,
    r_mean_mm: float = 36.25,
    alpha2_deg: float = 70.0,       # NGV exit absolute angle
    beta3_deg: float = -60.0,       # Rotor exit relative angle
    ngv_chord_mm: float = 14.0,
    rotor_chord_mm: float = 12.0,
    pitch_mm: float = 15.0,
    blade_thickness_ratio: float = 0.12,
    tip_clearance_mm: float = 0.3,
    blade_height_mm: float = 12.5,
    T_inlet_K: float = 1100.0,
) -> TurbineBladeAnalysis:
    """
    Analyze a turbine stage (NGV + rotor) at mean radius.

    Parameters:
        C_axial: Axial velocity through turbine (m/s)
        rpm: Rotational speed
        r_mean_mm: Mean blade radius (mm)
        alpha2_deg: NGV exit absolute flow angle (from axial)
        beta3_deg: Rotor exit relative flow angle (from axial, negative = opposite swirl)
        ngv_chord_mm: NGV chord
        rotor_chord_mm: Rotor blade chord
        pitch_mm: Rotor blade pitch
        blade_thickness_ratio: Rotor max thickness / chord
        tip_clearance_mm: Tip gap
        blade_height_mm: Blade span
        T_inlet_K: Turbine inlet temperature
    """
    result = TurbineBladeAnalysis()

    U = (rpm * 2 * np.pi / 60) * (r_mean_mm / 1000.0)

    # ── NGV: axial inlet, swirl exit ──
    # Station 1: NGV inlet (axial, no swirl)
    result.ngv_inlet = compute_velocity_triangle(C_axial, 0, 0)  # stator, no U

    # Station 2: NGV exit
    alpha2 = np.radians(alpha2_deg)
    C_theta2 = C_axial * np.tan(alpha2)
    result.ngv_outlet = compute_velocity_triangle(C_axial, C_theta2, U)

    # ── Rotor inlet = NGV outlet (in relative frame) ──
    result.rotor_inlet = result.ngv_outlet

    # Station 3: Rotor exit
    beta3 = np.radians(beta3_deg)
    W_theta3 = C_axial * np.tan(beta3)
    C_theta3 = W_theta3 + U
    result.rotor_outlet = compute_velocity_triangle(C_axial, C_theta3, U)

    # ── Performance Metrics ──
    delta_C_theta = C_theta2 - C_theta3  # Change in swirl
    delta_h0 = U * delta_C_theta  # Euler work extracted (J/kg)

    result.work_coefficient = delta_h0 / U**2 if U > 0 else 0
    result.flow_coefficient = C_axial / U if U > 0 else 0
    result.stage_loading = result.work_coefficient

    # Degree of reaction
    result.degree_of_reaction = 1 - (C_theta2 + C_theta3) / (2 * U) if U > 0 else 0.5

    # Zweifel coefficient (blade loading)
    sigma = rotor_chord_mm / pitch_mm
    beta2_rel = np.radians(result.rotor_inlet.beta_deg)
    beta3_rel = np.radians(result.rotor_outlet.beta_deg)
    result.zweifel_coefficient = (
        2 * sigma * np.cos(beta3_rel)**2 *
        (np.tan(beta2_rel) - np.tan(beta3_rel)) / np.cos(beta3_rel)
    ) if np.cos(beta3_rel) != 0 else 0
    result.zweifel_coefficient = abs(result.zweifel_coefficient)

    # ── Loss Estimation ──

    # NGV loss (Ainley-Mathieson)
    alpha_m = np.arctan(0.5 * (0 + np.tan(alpha2)))
    result.ngv_loss_coeff = 0.05 * (1 + (alpha2_deg / 80)**2)  # Simplified

    # Rotor profile loss
    D_rotor = abs(np.tan(beta2_rel) - np.tan(beta3_rel)) * np.cos(beta3_rel) / (2 * sigma)
    if D_rotor < 0.6:
        theta_c_r = 0.004 * (1 + 3.1 * D_rotor**2)
    else:
        theta_c_r = 0.004 * (1 + 3.1 * D_rotor**2) * (1 + 2 * (D_rotor - 0.6))
    result.rotor_profile_loss = 2 * theta_c_r * sigma

    # Rotor secondary loss
    AR = blade_height_mm / rotor_chord_mm
    result.rotor_secondary_loss = 0.018 * sigma / AR

    # Tip clearance loss
    tc_ratio = tip_clearance_mm / blade_height_mm
    result.rotor_tip_clearance_loss = 0.93 * tc_ratio * result.stage_loading

    result.total_loss_coeff = (result.ngv_loss_coeff +
                                result.rotor_profile_loss +
                                result.rotor_secondary_loss +
                                result.rotor_tip_clearance_loss)

    result.efficiency_estimate = 1 - result.total_loss_coeff * 0.8

    # ── Blade Loading Diagram ──
    n_pts = 50
    s = np.linspace(0, 1, n_pts)
    result.chord_stations = s

    # Turbine blade: acceleration on suction side, high loading
    Z = min(result.zweifel_coefficient, 2.0)
    result.Cp_pressure = 1 - 0.3 * s - 0.2 * np.sin(np.pi * s)
    result.Cp_suction = 1 - Z * (1.2 * np.sin(np.pi * s)**0.8 + 0.3 * s)

    # ── Notes ──
    if result.zweifel_coefficient < 0.6:
        result.notes.append(f"Zweifel {result.zweifel_coefficient:.2f} < 0.6 — under-loaded, add blades or increase turning")
    elif result.zweifel_coefficient > 1.1:
        result.notes.append(f"Zweifel {result.zweifel_coefficient:.2f} > 1.1 — over-loaded, boundary layer separation risk")
    else:
        result.notes.append(f"Zweifel {result.zweifel_coefficient:.2f} — good blade loading range")

    if result.degree_of_reaction < 0.2:
        result.notes.append(f"Reaction {result.degree_of_reaction:.2f} — nearly impulse design")
    elif result.degree_of_reaction > 0.6:
        result.notes.append(f"Reaction {result.degree_of_reaction:.2f} — high reaction, check rotor inlet conditions")

    return result


# ─── Combined Report ─────────────────────────────────────────────────────────

def full_blade_analysis(
    rpm: float = 100000,
    C_axial_compressor: float = 150.0,
    C_axial_turbine: float = 200.0,
    compressor_r_mean_mm: float = 40.0,
    turbine_r_mean_mm: float = 36.25,
) -> Dict:
    """Run full blade analysis for both compressor and turbine, return JSON-friendly dict."""

    comp = analyze_compressor_blade(
        C_axial=C_axial_compressor, rpm=rpm, r_mean_mm=compressor_r_mean_mm
    )
    turb = analyze_turbine_blade(
        C_axial=C_axial_turbine, rpm=rpm, r_mean_mm=turbine_r_mean_mm
    )

    return {
        'compressor': {
            'velocity_triangles': {
                'inlet': _vt_to_dict(comp.inlet),
                'outlet': _vt_to_dict(comp.outlet),
            },
            'metrics': {
                'diffusion_factor': round(comp.diffusion_factor, 3),
                'de_haller_number': round(comp.de_haller_number, 3),
                'work_coefficient': round(comp.work_coefficient, 3),
                'flow_coefficient': round(comp.flow_coefficient, 3),
                'degree_of_reaction': round(comp.degree_of_reaction, 3),
                'stage_loading': round(comp.stage_loading, 3),
            },
            'losses': {
                'profile': round(comp.profile_loss_coeff, 4),
                'secondary': round(comp.secondary_loss_coeff, 4),
                'tip_clearance': round(comp.tip_clearance_loss, 4),
                'total': round(comp.total_loss_coeff, 4),
            },
            'efficiency_estimate': round(comp.efficiency_estimate, 3),
            'stall_risk': comp.stall_risk,
            'pressure_distribution': {
                'chord': comp.chord_stations.tolist(),
                'Cp_suction': comp.Cp_suction.tolist(),
                'Cp_pressure': comp.Cp_pressure.tolist(),
            },
            'notes': comp.notes,
        },
        'turbine': {
            'velocity_triangles': {
                'ngv_inlet': _vt_to_dict(turb.ngv_inlet),
                'ngv_outlet': _vt_to_dict(turb.ngv_outlet),
                'rotor_inlet': _vt_to_dict(turb.rotor_inlet),
                'rotor_outlet': _vt_to_dict(turb.rotor_outlet),
            },
            'metrics': {
                'zweifel_coefficient': round(turb.zweifel_coefficient, 3),
                'work_coefficient': round(turb.work_coefficient, 3),
                'flow_coefficient': round(turb.flow_coefficient, 3),
                'degree_of_reaction': round(turb.degree_of_reaction, 3),
                'stage_loading': round(turb.stage_loading, 3),
            },
            'losses': {
                'ngv': round(turb.ngv_loss_coeff, 4),
                'rotor_profile': round(turb.rotor_profile_loss, 4),
                'rotor_secondary': round(turb.rotor_secondary_loss, 4),
                'rotor_tip_clearance': round(turb.rotor_tip_clearance_loss, 4),
                'total': round(turb.total_loss_coeff, 4),
            },
            'efficiency_estimate': round(turb.efficiency_estimate, 3),
            'pressure_distribution': {
                'chord': turb.chord_stations.tolist(),
                'Cp_suction': turb.Cp_suction.tolist(),
                'Cp_pressure': turb.Cp_pressure.tolist(),
            },
            'notes': turb.notes,
        }
    }


def _vt_to_dict(vt: VelocityTriangle) -> Dict:
    return {
        'C_axial': round(vt.C_axial, 1),
        'C_tangential': round(vt.C_tangential, 1),
        'C': round(vt.C, 1),
        'alpha_deg': round(vt.alpha_deg, 1),
        'W_axial': round(vt.W_axial, 1),
        'W_tangential': round(vt.W_tangential, 1),
        'W': round(vt.W, 1),
        'beta_deg': round(vt.beta_deg, 1),
        'U': round(vt.U, 1),
    }


def print_blade_report(data: Dict):
    """Print blade analysis report to console."""
    print(f"\n{'='*60}")
    print(f"  NovaTurbo Blade Profile Analysis")
    print(f"{'='*60}")

    # Compressor
    c = data['compressor']
    print(f"\n  ── COMPRESSOR ──")
    print(f"  Diffusion Factor:    {c['metrics']['diffusion_factor']:.3f}")
    print(f"  De Haller Number:    {c['metrics']['de_haller_number']:.3f}")
    print(f"  Work Coefficient:    {c['metrics']['work_coefficient']:.3f}")
    print(f"  Flow Coefficient:    {c['metrics']['flow_coefficient']:.3f}")
    print(f"  Degree of Reaction:  {c['metrics']['degree_of_reaction']:.3f}")
    print(f"  Efficiency Estimate: {c['efficiency_estimate']*100:.1f}%")
    print(f"  Stall Risk:          {c['stall_risk']}")
    print(f"  Total Loss Coeff:    {c['losses']['total']:.4f}")
    for n in c['notes']:
        print(f"  ⚠ {n}")

    # Turbine
    t = data['turbine']
    print(f"\n  ── TURBINE ──")
    print(f"  Zweifel Coefficient: {t['metrics']['zweifel_coefficient']:.3f}")
    print(f"  Work Coefficient:    {t['metrics']['work_coefficient']:.3f}")
    print(f"  Flow Coefficient:    {t['metrics']['flow_coefficient']:.3f}")
    print(f"  Degree of Reaction:  {t['metrics']['degree_of_reaction']:.3f}")
    print(f"  Efficiency Estimate: {t['efficiency_estimate']*100:.1f}%")
    print(f"  Total Loss Coeff:    {t['losses']['total']:.4f}")
    for n in t['notes']:
        print(f"  ⚠ {n}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    data = full_blade_analysis()
    print_blade_report(data)
