"""
NovaTurbo — Axial Turbine Stage Parametric Geometry Model

Generates a single-stage axial turbine geometry:
- Nozzle Guide Vanes (NGV)
- Rotor blades
- Hub and tip contours

Reference: Dixon & Hall, "Fluid Mechanics and Thermodynamics of Turbomachinery"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class TurbineParams:
    """Input parameters for axial turbine stage geometry."""
    # Overall sizing
    tip_diameter_mm: float = 85.0
    hub_diameter_mm: float = 60.0
    blade_height_mm: float = 12.5       # (tip - hub) / 2
    axial_chord_mm: float = 12.0
    stage_axial_length_mm: float = 30.0

    # NGV (Nozzle Guide Vanes / Stator)
    ngv_count: int = 13                  # Prime number
    ngv_chord_mm: float = 14.0
    ngv_inlet_angle_deg: float = 0.0     # Axial inlet
    ngv_outlet_angle_deg: float = 70.0   # High turning
    ngv_thickness_ratio: float = 0.15    # Max thickness / chord
    ngv_axial_position_mm: float = 0.0

    # Rotor blades
    blade_count: int = 17                # Prime number (avoid resonance with NGV)
    blade_chord_mm: float = 12.0
    blade_inlet_angle_deg: float = 40.0  # Relative frame
    blade_outlet_angle_deg: float = -60.0  # Relative frame (negative = opposite direction)
    blade_thickness_ratio: float = 0.12
    blade_twist_deg: float = 15.0        # Twist from hub to tip
    blade_lean_deg: float = 0.0          # Tangential lean
    tip_clearance_mm: float = 0.3
    blade_axial_position_mm: float = 16.0  # After NGV

    # Hub/tip contours
    hub_tip_ratio: float = 0.7          # r_hub / r_tip
    hub_flare_angle_deg: float = 5.0    # Hub radius increase through stage
    tip_flare_angle_deg: float = 0.0    # Casing flare

    # Operating conditions
    rpm: float = 100000
    inlet_temperature_K: float = 1100.0


@dataclass
class TurbineGeometry:
    """Computed turbine geometry outputs."""
    # Hub and casing contours (meridional)
    hub_contour: np.ndarray = field(default_factory=lambda: np.array([]))
    casing_contour: np.ndarray = field(default_factory=lambda: np.array([]))

    # Blade profiles (camberline + thickness at multiple spans)
    ngv_profiles: list = field(default_factory=list)
    rotor_profiles: list = field(default_factory=list)

    # 3D blade points
    ngv_points: np.ndarray = field(default_factory=lambda: np.array([]))
    rotor_points: np.ndarray = field(default_factory=lambda: np.array([]))

    # Key metrics
    tip_speed_m_s: float = 0.0
    blade_spacing_mm: float = 0.0
    solidity: float = 0.0              # chord / spacing
    aspect_ratio: float = 0.0


def _generate_blade_profile(chord_mm, inlet_angle_deg, outlet_angle_deg,
                             thickness_ratio, n_pts=40):
    """
    Generate a 2D blade profile (camberline + thickness distribution).
    Uses a circular-arc camberline with NACA thickness distribution.
    """
    s = np.linspace(0, 1, n_pts)

    # Camberline angles
    theta_in = np.radians(inlet_angle_deg)
    theta_out = np.radians(outlet_angle_deg)
    theta = theta_in + (theta_out - theta_in) * s

    # Camberline coordinates (blade-aligned)
    x_cam = chord_mm * s
    y_cam = chord_mm * np.cumsum(np.tan(theta - theta.mean())) / n_pts

    # NACA-style thickness distribution: t(s) = t_max * (a0*sqrt(s) - a1*s - a2*s² + a3*s³ - a4*s⁴)
    t_max = thickness_ratio * chord_mm
    t_dist = t_max * (2.969 * np.sqrt(s + 1e-10) - 1.260 * s -
                       3.516 * s**2 + 2.843 * s**3 - 1.036 * s**4)
    t_dist = np.clip(t_dist, 0, t_max)

    # Suction and pressure surfaces
    nx = -np.gradient(y_cam)
    ny = np.gradient(x_cam)
    norm = np.sqrt(nx**2 + ny**2) + 1e-10
    nx /= norm
    ny /= norm

    x_suction = x_cam + 0.5 * t_dist * nx
    y_suction = y_cam + 0.5 * t_dist * ny
    x_pressure = x_cam - 0.5 * t_dist * nx
    y_pressure = y_cam - 0.5 * t_dist * ny

    return {
        'camberline_x': x_cam,
        'camberline_y': y_cam,
        'suction_x': x_suction,
        'suction_y': y_suction,
        'pressure_x': x_pressure,
        'pressure_y': y_pressure,
        'thickness': t_dist,
        'chord_mm': chord_mm
    }


def compute_turbine_geometry(params: TurbineParams) -> TurbineGeometry:
    """
    Generate axial turbine stage geometry from parametric inputs.
    """
    geo = TurbineGeometry()

    r_tip = params.tip_diameter_mm / 2.0
    r_hub = params.hub_diameter_mm / 2.0
    blade_h = r_tip - r_hub
    L = params.stage_axial_length_mm

    # Key metrics
    geo.tip_speed_m_s = (params.rpm * 2 * np.pi / 60) * (r_tip / 1000.0)
    geo.blade_spacing_mm = np.pi * params.tip_diameter_mm / params.blade_count
    geo.solidity = params.blade_chord_mm / geo.blade_spacing_mm
    geo.aspect_ratio = blade_h / params.blade_chord_mm

    # --- Hub and casing meridional contours ---
    n_pts = 40
    z = np.linspace(0, L, n_pts)

    hub_flare = np.tan(np.radians(params.hub_flare_angle_deg))
    tip_flare = np.tan(np.radians(params.tip_flare_angle_deg))

    hub_r = r_hub + hub_flare * z
    casing_r = r_tip + tip_flare * z

    geo.hub_contour = np.column_stack([z, hub_r])
    geo.casing_contour = np.column_stack([z, casing_r])

    # --- NGV profiles at multiple spans ---
    for span_frac in np.linspace(0, 1, 7):
        r = r_hub + span_frac * blade_h
        profile = _generate_blade_profile(
            params.ngv_chord_mm,
            params.ngv_inlet_angle_deg,
            params.ngv_outlet_angle_deg,
            params.ngv_thickness_ratio
        )
        profile['span'] = span_frac
        profile['radius_mm'] = r
        geo.ngv_profiles.append(profile)

    # --- Rotor blade profiles (with twist) ---
    for span_frac in np.linspace(0, 1, 7):
        r = r_hub + span_frac * blade_h

        # Apply twist: hub has more turning, tip has less
        twist_offset = params.blade_twist_deg * (span_frac - 0.5)
        inlet_angle = params.blade_inlet_angle_deg + twist_offset
        outlet_angle = params.blade_outlet_angle_deg - twist_offset

        profile = _generate_blade_profile(
            params.blade_chord_mm,
            inlet_angle,
            outlet_angle,
            params.blade_thickness_ratio
        )
        profile['span'] = span_frac
        profile['radius_mm'] = r
        geo.rotor_profiles.append(profile)

    # --- 3D blade point clouds (for visualization / STL) ---
    # NGVs
    ngv_points = []
    for i in range(params.ngv_count):
        theta_offset = 2 * np.pi * i / params.ngv_count
        for profile in geo.ngv_profiles:
            r = profile['radius_mm']
            for side in ['suction', 'pressure']:
                x_2d = profile[f'{side}_x']
                y_2d = profile[f'{side}_y']
                theta = theta_offset + y_2d / r
                pts = np.column_stack([
                    r * np.cos(theta),
                    r * np.sin(theta),
                    params.ngv_axial_position_mm + x_2d
                ])
                ngv_points.append(pts)

    if ngv_points:
        geo.ngv_points = np.vstack(ngv_points)

    # Rotor blades
    rotor_points = []
    for i in range(params.blade_count):
        theta_offset = 2 * np.pi * i / params.blade_count
        for profile in geo.rotor_profiles:
            r = profile['radius_mm']
            for side in ['suction', 'pressure']:
                x_2d = profile[f'{side}_x']
                y_2d = profile[f'{side}_y']
                theta = theta_offset + y_2d / r
                pts = np.column_stack([
                    r * np.cos(theta),
                    r * np.sin(theta),
                    params.blade_axial_position_mm + x_2d
                ])
                rotor_points.append(pts)

    if rotor_points:
        geo.rotor_points = np.vstack(rotor_points)

    return geo


def get_turbine_mass_kg(params: TurbineParams, material_density: float = 8190.0) -> float:
    """
    Estimate turbine stage mass (disc + blades + NGV).
    Default material: Inconel 718 (8190 kg/m³)
    """
    r_tip = params.tip_diameter_mm / 2000.0
    r_hub = params.hub_diameter_mm / 2000.0
    blade_h = r_tip - r_hub

    # Turbine disc
    disc_thickness = params.axial_chord_mm / 1000.0
    disc_vol = np.pi * (r_hub**2) * disc_thickness * 0.6  # 60% solid

    # Rotor blades
    blade_vol = params.blade_count * blade_h * (params.blade_chord_mm / 1000.0) * \
        (params.blade_thickness_ratio * params.blade_chord_mm / 1000.0)

    # NGV blades
    ngv_vol = params.ngv_count * blade_h * (params.ngv_chord_mm / 1000.0) * \
        (params.ngv_thickness_ratio * params.ngv_chord_mm / 1000.0)

    total_vol = disc_vol + blade_vol + ngv_vol
    return total_vol * material_density


if __name__ == "__main__":
    params = TurbineParams()
    geo = compute_turbine_geometry(params)

    print(f"=== NovaTurbo Axial Turbine Stage ===")
    print(f"Tip diameter: {params.tip_diameter_mm} mm")
    print(f"Hub diameter: {params.hub_diameter_mm} mm")
    print(f"Blade height: {params.tip_diameter_mm/2 - params.hub_diameter_mm/2:.1f} mm")
    print(f"Tip speed: {geo.tip_speed_m_s:.1f} m/s")
    print(f"NGV count: {params.ngv_count}, Rotor count: {params.blade_count}")
    print(f"Solidity: {geo.solidity:.2f}")
    print(f"Aspect ratio: {geo.aspect_ratio:.2f}")
    print(f"NGV points: {len(geo.ngv_points)}")
    print(f"Rotor points: {len(geo.rotor_points)}")
    print(f"Estimated mass: {get_turbine_mass_kg(params)*1000:.1f} g")
