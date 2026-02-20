"""
NovaTurbo — Air Inlet Parametric Geometry Model

Simple pitot/bell-mouth inlet for micro turbojet.
Designed for low-speed VTOL operation (subsonic).
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class InletParams:
    """Input parameters for air inlet geometry."""
    inlet_diameter_mm: float = 80.0
    outlet_diameter_mm: float = 50.0      # Matches compressor inducer
    length_mm: float = 40.0
    lip_radius_mm: float = 8.0            # Rounded intake lip
    wall_thickness_mm: float = 1.5
    
    inlet_type: str = "bellmouth"         # "bellmouth", "pitot", "flush"
    screen_mesh: bool = True              # FOD protection screen
    screen_mesh_size_mm: float = 2.0

    # Pressure recovery
    pressure_recovery: float = 0.98


@dataclass
class InletGeometry:
    """Computed inlet geometry outputs."""
    inner_contour: np.ndarray = field(default_factory=lambda: np.array([]))
    outer_contour: np.ndarray = field(default_factory=lambda: np.array([]))
    capture_area_mm2: float = 0.0
    throat_area_mm2: float = 0.0


def compute_inlet_geometry(params: InletParams) -> InletGeometry:
    """Generate inlet geometry from parametric inputs."""
    geo = InletGeometry()

    r_in = params.inlet_diameter_mm / 2.0
    r_out = params.outlet_diameter_mm / 2.0
    L = params.length_mm
    t = params.wall_thickness_mm

    n_pts = 40
    z = np.linspace(0, L, n_pts)
    s = z / L

    if params.inlet_type == "bellmouth":
        # Bell-mouth: smooth elliptical contraction
        r_inner = r_in - (r_in - r_out) * (1 - np.sqrt(1 - s**2 + 1e-10))
        # Clamp near outlet
        r_inner = np.clip(r_inner, r_out, r_in)
    elif params.inlet_type == "pitot":
        # Simple conic contraction
        r_inner = r_in + (r_out - r_in) * s
    else:
        # Flush inlet (sharp lip)
        r_inner = r_in + (r_out - r_in) * (3*s**2 - 2*s**3)

    geo.inner_contour = np.column_stack([z, r_inner])
    geo.outer_contour = np.column_stack([z, r_inner + t])

    geo.capture_area_mm2 = np.pi * r_in**2
    geo.throat_area_mm2 = np.pi * r_out**2

    return geo


def get_inlet_mass_kg(params: InletParams, material_density: float = 4430.0) -> float:
    """Estimate inlet mass. Default: Ti-6Al-4V."""
    r_avg = (params.inlet_diameter_mm + params.outlet_diameter_mm) / 4000.0
    L = params.length_mm / 1000.0
    t = params.wall_thickness_mm / 1000.0
    return 2 * np.pi * r_avg * t * L * material_density


if __name__ == "__main__":
    params = InletParams()
    geo = compute_inlet_geometry(params)
    print(f"=== NovaTurbo Air Inlet ===")
    print(f"Type: {params.inlet_type}")
    print(f"Ø{params.inlet_diameter_mm} → Ø{params.outlet_diameter_mm} mm, L={params.length_mm} mm")
    print(f"Capture area: {geo.capture_area_mm2:.0f} mm²")
    print(f"Throat area: {geo.throat_area_mm2:.0f} mm²")
    print(f"Estimated mass: {get_inlet_mass_kg(params)*1000:.1f} g")
