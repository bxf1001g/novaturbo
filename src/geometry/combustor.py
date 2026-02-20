"""
NovaTurbo — Annular Combustion Chamber Parametric Geometry Model

Generates an annular combustor geometry defined by:
- Outer & inner liner (casing)
- Fuel injector positions
- Dilution holes for mixing
- Cooling holes (film cooling)
- Dome / swirler geometry

Reference: Lefebvre & Ballal, "Gas Turbine Combustion" (3rd Edition)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CombustorParams:
    """Input parameters for annular combustion chamber geometry."""
    # Overall dimensions
    outer_diameter_mm: float = 110.0
    inner_diameter_mm: float = 60.0
    length_mm: float = 80.0
    liner_thickness_mm: float = 1.2
    casing_thickness_mm: float = 1.5

    # Dome / primary zone
    dome_recess_mm: float = 5.0         # Dome set-back from liner start
    dome_type: str = "flat"             # "flat" or "spherical"

    # Fuel injectors
    num_fuel_injectors: int = 6
    injector_diameter_mm: float = 3.0
    injector_spray_angle_deg: float = 60.0
    swirler_vane_count: int = 8
    swirler_vane_angle_deg: float = 45.0
    injector_axial_position_fraction: float = 0.05  # Near dome

    # Primary zone holes (air admission)
    num_primary_holes: int = 6
    primary_hole_diameter_mm: float = 6.0
    primary_hole_axial_fraction: float = 0.25   # 25% of length

    # Dilution holes
    num_dilution_holes: int = 12
    dilution_hole_diameter_mm: float = 8.0
    dilution_hole_axial_fraction: float = 0.55  # 55% of length

    # Cooling holes (film cooling rows)
    num_cooling_rows: int = 4
    holes_per_row: int = 24
    cooling_hole_diameter_mm: float = 1.5
    cooling_hole_angle_deg: float = 30.0    # Angled to surface for film cooling

    # Liner zones (fraction of total length)
    primary_zone_fraction: float = 0.30     # 0 to 30%
    intermediate_zone_fraction: float = 0.25  # 30% to 55%
    dilution_zone_fraction: float = 0.45    # 55% to 100%

    # Operating conditions (for sizing reference)
    inlet_temperature_K: float = 450.0      # After compressor
    outlet_temperature_K: float = 1100.0    # Turbine inlet
    air_fuel_ratio: float = 60.0
    pressure_Pa: float = 350000.0           # After compressor


@dataclass
class CombustorGeometry:
    """Computed combustor geometry outputs."""
    # Liner contours (2D meridional — axial vs radius)
    outer_liner_outer: np.ndarray = field(default_factory=lambda: np.array([]))
    outer_liner_inner: np.ndarray = field(default_factory=lambda: np.array([]))
    inner_liner_outer: np.ndarray = field(default_factory=lambda: np.array([]))
    inner_liner_inner: np.ndarray = field(default_factory=lambda: np.array([]))

    # Hole positions [axial_mm, circumferential_angle_rad, radius_mm]
    fuel_injector_positions: list = field(default_factory=list)
    primary_hole_positions: list = field(default_factory=list)
    dilution_hole_positions: list = field(default_factory=list)
    cooling_hole_positions: list = field(default_factory=list)

    # Derived
    annulus_area_mm2: float = 0.0
    volume_mm3: float = 0.0
    surface_area_mm2: float = 0.0
    loading_parameter: float = 0.0      # Combustor loading (kg/(s·atm^1.8·m³))


def compute_combustor_geometry(params: CombustorParams) -> CombustorGeometry:
    """
    Generate annular combustion chamber geometry from parametric inputs.
    """
    geo = CombustorGeometry()

    r_outer = params.outer_diameter_mm / 2.0
    r_inner = params.inner_diameter_mm / 2.0
    L = params.length_mm
    t_liner = params.liner_thickness_mm

    # --- Liner contours (straight annular liners) ---
    n_pts = 60
    z = np.linspace(0, L, n_pts)

    # Outer liner (slight divergence for pressure stability)
    diverge = 0.02  # 2% radius increase over length
    outer_r = r_outer + diverge * r_outer * (z / L)

    geo.outer_liner_outer = np.column_stack([z, outer_r + params.casing_thickness_mm])
    geo.outer_liner_inner = np.column_stack([z, outer_r])
    geo.inner_liner_outer = np.column_stack([z, np.full(n_pts, r_inner)])
    geo.inner_liner_inner = np.column_stack([z, np.full(n_pts, r_inner - t_liner)])

    # --- Annulus area & volume ---
    r_mid_outer = (r_outer + r_outer * (1 + diverge)) / 2.0
    geo.annulus_area_mm2 = np.pi * (r_mid_outer**2 - r_inner**2)
    geo.volume_mm3 = geo.annulus_area_mm2 * L
    geo.surface_area_mm2 = 2 * np.pi * (r_mid_outer + r_inner) * L

    # --- Fuel injector positions (on dome face, evenly spaced) ---
    r_mid = (r_outer + r_inner) / 2.0
    for i in range(params.num_fuel_injectors):
        theta = 2 * np.pi * i / params.num_fuel_injectors
        geo.fuel_injector_positions.append({
            'index': i,
            'axial_mm': params.injector_axial_position_fraction * L,
            'theta_rad': theta,
            'radius_mm': r_mid,
            'diameter_mm': params.injector_diameter_mm,
            'x': r_mid * np.cos(theta),
            'y': r_mid * np.sin(theta),
            'z': params.injector_axial_position_fraction * L
        })

    # --- Primary holes (on outer liner) ---
    for i in range(params.num_primary_holes):
        theta = 2 * np.pi * i / params.num_primary_holes
        # Offset from injectors by half-spacing
        theta += np.pi / params.num_primary_holes
        geo.primary_hole_positions.append({
            'index': i,
            'axial_mm': params.primary_hole_axial_fraction * L,
            'theta_rad': theta,
            'radius_mm': r_outer,
            'diameter_mm': params.primary_hole_diameter_mm,
            'surface': 'outer_liner'
        })

    # --- Dilution holes (larger, on both liners) ---
    for i in range(params.num_dilution_holes):
        theta = 2 * np.pi * i / params.num_dilution_holes
        surface = 'outer_liner' if i % 2 == 0 else 'inner_liner'
        r = r_outer if surface == 'outer_liner' else r_inner
        geo.dilution_hole_positions.append({
            'index': i,
            'axial_mm': params.dilution_hole_axial_fraction * L,
            'theta_rad': theta,
            'radius_mm': r,
            'diameter_mm': params.dilution_hole_diameter_mm,
            'surface': surface
        })

    # --- Cooling holes (film cooling rows along liner) ---
    cooling_row_positions = np.linspace(0.15, 0.85, params.num_cooling_rows)
    for row_idx, axial_frac in enumerate(cooling_row_positions):
        for hole_idx in range(params.holes_per_row):
            theta = 2 * np.pi * hole_idx / params.holes_per_row
            # Alternate rows on outer and inner liner
            surface = 'outer_liner' if row_idx % 2 == 0 else 'inner_liner'
            r = r_outer if surface == 'outer_liner' else r_inner
            geo.cooling_hole_positions.append({
                'row': row_idx,
                'index': hole_idx,
                'axial_mm': axial_frac * L,
                'theta_rad': theta,
                'radius_mm': r,
                'diameter_mm': params.cooling_hole_diameter_mm,
                'angle_deg': params.cooling_hole_angle_deg,
                'surface': surface
            })

    return geo


def get_combustor_mass_kg(params: CombustorParams, material_density: float = 8190.0) -> float:
    """
    Estimate combustor mass (liner + casing shell volume).
    Default material: Inconel 718 (8190 kg/m³)
    """
    r_o = params.outer_diameter_mm / 2000.0
    r_i = params.inner_diameter_mm / 2000.0
    L = params.length_mm / 1000.0
    t_l = params.liner_thickness_mm / 1000.0
    t_c = params.casing_thickness_mm / 1000.0

    # Outer liner shell
    outer_vol = 2 * np.pi * r_o * t_l * L
    # Inner liner shell
    inner_vol = 2 * np.pi * r_i * t_l * L
    # Outer casing
    casing_vol = 2 * np.pi * (r_o + t_l) * t_c * L
    # Dome plate (annular)
    dome_vol = np.pi * (r_o**2 - r_i**2) * t_l

    # Subtract holes
    n_holes = (params.num_primary_holes + params.num_dilution_holes +
               params.num_cooling_rows * params.holes_per_row)
    avg_hole_d = 3.0 / 1000.0  # Average hole diameter in m
    hole_vol = n_holes * np.pi * (avg_hole_d/2)**2 * t_l

    total_vol = outer_vol + inner_vol + casing_vol + dome_vol - hole_vol
    return total_vol * material_density


def compute_combustor_loading(params: CombustorParams, mass_flow_kg_s: float) -> float:
    """
    Calculate combustor loading parameter.
    Low loading = stable combustion, high loading = risk of blowout.
    Typical range: 1-10 kg/(s·atm^1.8·m³)
    """
    volume_m3 = compute_combustor_geometry(params).volume_mm3 * 1e-9
    pressure_atm = params.pressure_Pa / 101325.0
    loading = mass_flow_kg_s / (pressure_atm**1.8 * volume_m3)
    return loading


if __name__ == "__main__":
    params = CombustorParams()
    geo = compute_combustor_geometry(params)

    print(f"=== NovaTurbo Annular Combustor ===")
    print(f"Dimensions: Ø{params.outer_diameter_mm}/Ø{params.inner_diameter_mm} × {params.length_mm} mm")
    print(f"Annulus area: {geo.annulus_area_mm2:.0f} mm²")
    print(f"Volume: {geo.volume_mm3:.0f} mm³ ({geo.volume_mm3*1e-9*1e6:.1f} cm³)")
    print(f"Surface area: {geo.surface_area_mm2:.0f} mm²")
    print(f"Fuel injectors: {len(geo.fuel_injector_positions)}")
    print(f"Primary holes: {len(geo.primary_hole_positions)}")
    print(f"Dilution holes: {len(geo.dilution_hole_positions)}")
    print(f"Cooling holes: {len(geo.cooling_hole_positions)}")
    print(f"Estimated mass: {get_combustor_mass_kg(params)*1000:.1f} g")
    print(f"Combustor loading: {compute_combustor_loading(params, 0.15):.2f} kg/(s·atm^1.8·m³)")
