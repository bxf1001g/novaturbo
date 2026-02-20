"""
NovaTurbo — Convergent Nozzle Parametric Geometry Model

Generates a convergent exhaust nozzle geometry for micro turbojet.
For subsonic exhaust (typical of micro turbojets), a simple convergent
nozzle is sufficient. Convergent-divergent (C-D) option is included
for future supersonic designs.

Reference: Anderson, "Modern Compressible Flow"
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class NozzleParams:
    """Input parameters for nozzle geometry."""
    # Geometry
    inlet_diameter_mm: float = 85.0     # Matches turbine exit
    exit_diameter_mm: float = 55.0      # Throat (convergent nozzle)
    length_mm: float = 50.0
    wall_thickness_mm: float = 1.5

    # Contour shape
    contour_type: str = "conic"         # "conic", "bell", or "rao"
    contraction_half_angle_deg: float = 15.0  # For conic nozzle

    # Convergent-Divergent option (for supersonic, future use)
    is_convergent_divergent: bool = False
    divergent_exit_diameter_mm: float = 65.0  # Only if C-D
    divergent_length_mm: float = 30.0
    divergent_half_angle_deg: float = 8.0

    # Nozzle lip
    lip_thickness_mm: float = 2.0
    lip_radius_mm: float = 3.0         # Rounded exit lip

    # Mounting
    flange_outer_diameter_mm: float = 95.0
    flange_thickness_mm: float = 3.0
    bolt_circle_diameter_mm: float = 90.0
    num_bolts: int = 6

    # Operating conditions
    discharge_coefficient: float = 0.98


@dataclass
class NozzleGeometry:
    """Computed nozzle geometry outputs."""
    # Inner contour (flow path)
    inner_contour: np.ndarray = field(default_factory=lambda: np.array([]))
    # Outer contour (external surface)
    outer_contour: np.ndarray = field(default_factory=lambda: np.array([]))

    # Key areas
    inlet_area_mm2: float = 0.0
    throat_area_mm2: float = 0.0
    exit_area_mm2: float = 0.0
    area_ratio: float = 0.0

    # 3D surface points
    surface_points: np.ndarray = field(default_factory=lambda: np.array([]))


def _bell_contour(r_inlet, r_exit, length, n_pts=50):
    """Generate a bell-shaped (cosine) nozzle contour."""
    z = np.linspace(0, length, n_pts)
    t = z / length
    # Cosine contraction: smooth acceleration
    r = r_inlet - (r_inlet - r_exit) * (1 - np.cos(np.pi * t)) / 2
    return z, r


def _conic_contour(r_inlet, r_exit, length, n_pts=50):
    """Generate a straight conic nozzle contour."""
    z = np.linspace(0, length, n_pts)
    r = r_inlet + (r_exit - r_inlet) * (z / length)
    return z, r


def _rao_contour(r_inlet, r_exit, length, n_pts=50):
    """
    Generate a Rao-optimized (thrust-optimized) contour.
    Uses a cubic Bezier approximation of the Rao method.
    """
    z = np.linspace(0, length, n_pts)
    t = z / length
    # Rao-like cubic: fast initial contraction, gentle near throat
    r = r_inlet + (r_exit - r_inlet) * (3*t**2 - 2*t**3)
    return z, r


def compute_nozzle_geometry(params: NozzleParams) -> NozzleGeometry:
    """
    Generate nozzle geometry from parametric inputs.
    """
    geo = NozzleGeometry()

    r_inlet = params.inlet_diameter_mm / 2.0
    r_exit = params.exit_diameter_mm / 2.0
    L = params.length_mm
    t = params.wall_thickness_mm

    # --- Inner contour (flow path) ---
    contour_funcs = {
        'conic': _conic_contour,
        'bell': _bell_contour,
        'rao': _rao_contour
    }
    contour_fn = contour_funcs.get(params.contour_type, _bell_contour)
    z_inner, r_inner = contour_fn(r_inlet, r_exit, L)

    # Add convergent-divergent section if enabled
    if params.is_convergent_divergent:
        r_div_exit = params.divergent_exit_diameter_mm / 2.0
        z_div = np.linspace(L, L + params.divergent_length_mm, 30)[1:]
        t_div = (z_div - L) / params.divergent_length_mm
        r_div = r_exit + (r_div_exit - r_exit) * (3*t_div**2 - 2*t_div**3)
        z_inner = np.concatenate([z_inner, z_div])
        r_inner = np.concatenate([r_inner, r_div])

    geo.inner_contour = np.column_stack([z_inner, r_inner])

    # --- Outer contour ---
    r_outer = r_inner + t
    # Smooth outer surface (less contraction than inner)
    r_outer_smooth = r_inlet + t + (r_exit + t - r_inlet - t) * \
        (1 - np.cos(np.pi * np.linspace(0, 1, len(z_inner)))) / 2
    # Blend: mostly follow inner wall but smoother
    r_outer = 0.7 * (r_inner + t) + 0.3 * r_outer_smooth
    geo.outer_contour = np.column_stack([z_inner, r_outer])

    # --- Areas ---
    geo.inlet_area_mm2 = np.pi * r_inlet**2
    geo.throat_area_mm2 = np.pi * r_exit**2
    if params.is_convergent_divergent:
        geo.exit_area_mm2 = np.pi * (params.divergent_exit_diameter_mm / 2.0)**2
    else:
        geo.exit_area_mm2 = geo.throat_area_mm2
    geo.area_ratio = geo.inlet_area_mm2 / geo.throat_area_mm2

    # --- 3D surface points (revolution of inner contour) ---
    n_theta = 36
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    surface_pts = []
    for i in range(len(z_inner)):
        for th in theta:
            surface_pts.append([
                r_inner[i] * np.cos(th),
                r_inner[i] * np.sin(th),
                z_inner[i]
            ])
    geo.surface_points = np.array(surface_pts)

    return geo


def get_nozzle_mass_kg(params: NozzleParams, material_density: float = 8190.0) -> float:
    """
    Estimate nozzle mass (shell of revolution).
    Default material: Inconel 718 (8190 kg/m³)
    """
    r_in = params.inlet_diameter_mm / 2000.0
    r_ex = params.exit_diameter_mm / 2000.0
    L = params.length_mm / 1000.0
    t = params.wall_thickness_mm / 1000.0

    # Average radius
    r_avg = (r_in + r_ex) / 2.0
    # Cone shell volume
    shell_vol = np.pi * (r_avg * 2) * t * L

    # Flange
    r_flange = params.flange_outer_diameter_mm / 2000.0
    flange_vol = np.pi * (r_flange**2 - r_in**2) * (params.flange_thickness_mm / 1000.0)

    return (shell_vol + flange_vol) * material_density


if __name__ == "__main__":
    params = NozzleParams()
    geo = compute_nozzle_geometry(params)

    print(f"=== NovaTurbo Convergent Nozzle ===")
    print(f"Inlet: Ø{params.inlet_diameter_mm} mm → Exit: Ø{params.exit_diameter_mm} mm")
    print(f"Length: {params.length_mm} mm")
    print(f"Contour type: {params.contour_type}")
    print(f"Inlet area: {geo.inlet_area_mm2:.0f} mm²")
    print(f"Throat area: {geo.throat_area_mm2:.0f} mm²")
    print(f"Area ratio: {geo.area_ratio:.2f}")
    print(f"Contour points: {len(geo.inner_contour)}")
    print(f"Surface points: {len(geo.surface_points)}")
    print(f"Estimated mass: {get_nozzle_mass_kg(params)*1000:.1f} g")
