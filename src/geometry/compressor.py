"""
NovaTurbo — Centrifugal Compressor Parametric Geometry Model

Generates a centrifugal (radial) compressor geometry defined by:
- Impeller: inducer eye, blade angles, tip diameter
- Diffuser: vaneless or vaned section
- Shroud contour

Reference: Aungier, "Centrifugal Compressors: A Strategy for Aerodynamic Design"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompressorParams:
    """Input parameters for centrifugal compressor geometry."""
    # Impeller
    impeller_tip_diameter_mm: float = 80.0
    impeller_hub_diameter_mm: float = 20.0
    inducer_tip_diameter_mm: float = 50.0
    blade_count: int = 12
    splitter_blade_count: int = 12      # Splitter blades (half-length)
    inducer_blade_angle_deg: float = 30.0
    exducer_blade_angle_deg: float = 60.0
    blade_thickness_mm: float = 1.0
    backsweep_angle_deg: float = 30.0   # Backward sweep for stability
    axial_length_mm: float = 25.0       # Impeller axial depth

    # Diffuser
    diffuser_type: str = "vaneless"     # "vaneless" or "vaned"
    diffuser_radius_ratio: float = 1.5  # Diffuser outlet / impeller tip radius
    diffuser_width_mm: float = 5.0
    diffuser_vane_count: int = 17       # If vaned
    diffuser_vane_angle_deg: float = 25.0

    # Shroud
    shroud_clearance_mm: float = 0.3
    inlet_duct_length_mm: float = 15.0

    # Operating conditions (for geometry sizing)
    rpm: float = 100000
    pressure_ratio: float = 3.5
    mass_flow_kg_s: float = 0.15


@dataclass
class CompressorGeometry:
    """Computed compressor geometry outputs."""
    # Impeller profile points (meridional plane)
    hub_contour: np.ndarray = field(default_factory=lambda: np.array([]))
    shroud_contour: np.ndarray = field(default_factory=lambda: np.array([]))

    # Blade profiles at multiple span positions
    blade_profiles: list = field(default_factory=list)

    # Key dimensions
    tip_speed_m_s: float = 0.0
    outlet_width_mm: float = 0.0
    diffuser_outlet_radius_mm: float = 0.0

    # Derived geometry arrays
    impeller_points: np.ndarray = field(default_factory=lambda: np.array([]))
    diffuser_points: np.ndarray = field(default_factory=lambda: np.array([]))


def compute_compressor_geometry(params: CompressorParams) -> CompressorGeometry:
    """
    Generate centrifugal compressor geometry from parametric inputs.
    Returns meridional contours, blade profiles, and key dimensions.
    """
    geo = CompressorGeometry()

    r_tip = params.impeller_tip_diameter_mm / 2.0
    r_hub = params.impeller_hub_diameter_mm / 2.0
    r_inducer = params.inducer_tip_diameter_mm / 2.0
    ax_len = params.axial_length_mm

    # Tip speed
    geo.tip_speed_m_s = (params.rpm * 2 * np.pi / 60) * (r_tip / 1000.0)

    # --- Meridional contour (hub and shroud) ---
    n_pts = 50

    # Hub contour: from inlet to impeller exit (curved)
    t = np.linspace(0, 1, n_pts)
    # Hub starts at (0, r_hub) and curves to (ax_len, r_tip)
    hub_z = t * ax_len
    hub_r = r_hub + (r_tip - r_hub) * (3 * t**2 - 2 * t**3)  # Smooth cubic
    geo.hub_contour = np.column_stack([hub_z, hub_r])

    # Shroud contour: from inlet to impeller exit
    shroud_z = np.concatenate([
        np.linspace(-params.inlet_duct_length_mm, 0, 10),
        t * ax_len
    ])
    shroud_r_inlet = np.full(10, r_inducer + params.shroud_clearance_mm)
    shroud_r_impeller = (r_inducer + params.shroud_clearance_mm) + \
        (r_tip + params.shroud_clearance_mm - r_inducer - params.shroud_clearance_mm) * \
        (3 * t**2 - 2 * t**3)
    shroud_r = np.concatenate([shroud_r_inlet, shroud_r_impeller])
    geo.shroud_contour = np.column_stack([shroud_z, shroud_r])

    # --- Blade profiles (simplified 2D camberline) ---
    # Generate blade camberline from inducer to exducer
    beta_1 = np.radians(params.inducer_blade_angle_deg)
    beta_2 = np.radians(params.exducer_blade_angle_deg)
    backsweep = np.radians(params.backsweep_angle_deg)

    n_blade_pts = 30
    s = np.linspace(0, 1, n_blade_pts)

    for span_frac in [0.0, 0.5, 1.0]:  # Hub, mid, tip
        r_start = r_hub + span_frac * (r_inducer - r_hub)
        r_end = r_tip

        # Radial positions along blade
        r_blade = r_start + (r_end - r_start) * s

        # Blade angle varies from beta_1 to (beta_2 - backsweep)
        beta = beta_1 + (beta_2 - backsweep - beta_1) * s

        # Theta (circumferential angle) from integration of tan(beta)/r
        dtheta = np.tan(beta) / r_blade
        theta = np.cumsum(dtheta) * (r_end - r_start) / n_blade_pts

        # Convert to x, y coordinates
        x_blade = r_blade * np.cos(theta)
        y_blade = r_blade * np.sin(theta)

        geo.blade_profiles.append({
            'span': span_frac,
            'r': r_blade,
            'theta': theta,
            'x': x_blade,
            'y': y_blade,
            'beta': np.degrees(beta)
        })

    # --- Diffuser geometry ---
    geo.diffuser_outlet_radius_mm = r_tip * params.diffuser_radius_ratio
    geo.outlet_width_mm = params.diffuser_width_mm

    # Diffuser channel points
    n_diff = 20
    r_diff = np.linspace(r_tip, geo.diffuser_outlet_radius_mm, n_diff)
    geo.diffuser_points = np.column_stack([
        np.full(n_diff, ax_len),  # All at impeller exit axial position
        r_diff
    ])

    # --- Full impeller 3D points (for STL export) ---
    theta_blades = np.linspace(0, 2 * np.pi, params.blade_count, endpoint=False)
    all_points = []
    for theta_offset in theta_blades:
        for profile in geo.blade_profiles:
            pts = np.column_stack([
                profile['r'] * np.cos(profile['theta'] + theta_offset),
                profile['r'] * np.sin(profile['theta'] + theta_offset),
                np.linspace(0, ax_len, len(profile['r']))
            ])
            all_points.append(pts)

    if all_points:
        geo.impeller_points = np.vstack(all_points)

    return geo


def get_compressor_mass_kg(params: CompressorParams, material_density: float = 4430.0) -> float:
    """
    Estimate compressor mass (simplified volume calculation).
    Default material: Ti-6Al-4V (4430 kg/m³)
    """
    r_tip = params.impeller_tip_diameter_mm / 2000.0  # Convert to meters
    r_hub = params.impeller_hub_diameter_mm / 2000.0
    ax_len = params.axial_length_mm / 1000.0

    # Approximate impeller as annular disc with blades
    disc_volume = np.pi * (r_tip**2 - r_hub**2) * ax_len * 0.3  # 30% solid fraction
    blade_volume = (params.blade_count + params.splitter_blade_count) * \
        (r_tip - r_hub) * ax_len * (params.blade_thickness_mm / 1000.0) * 0.5

    total_volume = disc_volume + blade_volume
    return total_volume * material_density


if __name__ == "__main__":
    # Test with default parameters
    params = CompressorParams()
    geo = compute_compressor_geometry(params)

    print(f"=== NovaTurbo Centrifugal Compressor ===")
    print(f"Tip diameter: {params.impeller_tip_diameter_mm} mm")
    print(f"Tip speed: {geo.tip_speed_m_s:.1f} m/s")
    print(f"Blade count: {params.blade_count} + {params.splitter_blade_count} splitters")
    print(f"Diffuser outlet radius: {geo.diffuser_outlet_radius_mm:.1f} mm")
    print(f"Hub contour points: {len(geo.hub_contour)}")
    print(f"Blade profiles: {len(geo.blade_profiles)} spans")
    print(f"Total impeller points: {len(geo.impeller_points)}")
    print(f"Estimated mass: {get_compressor_mass_kg(params)*1000:.1f} g")
