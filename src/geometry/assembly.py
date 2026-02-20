"""
NovaTurbo — Full Engine Assembly Module

Combines all engine components into a complete micro turbojet assembly.
Handles axial positioning, interface matching, and total engine metrics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

from .inlet import InletParams, InletGeometry, compute_inlet_geometry, get_inlet_mass_kg
from .compressor import CompressorParams, CompressorGeometry, compute_compressor_geometry, get_compressor_mass_kg
from .combustor import CombustorParams, CombustorGeometry, compute_combustor_geometry, get_combustor_mass_kg
from .turbine import TurbineParams, TurbineGeometry, compute_turbine_geometry, get_turbine_mass_kg
from .nozzle import NozzleParams, NozzleGeometry, compute_nozzle_geometry, get_nozzle_mass_kg


@dataclass
class EngineAssemblyParams:
    """Parameters for the complete engine assembly."""
    name: str = "NovaTurbo-100"

    # Component parameters
    inlet: InletParams = field(default_factory=InletParams)
    compressor: CompressorParams = field(default_factory=CompressorParams)
    combustor: CombustorParams = field(default_factory=CombustorParams)
    turbine: TurbineParams = field(default_factory=TurbineParams)
    nozzle: NozzleParams = field(default_factory=NozzleParams)

    # Inter-component gaps (0 for flush fit)
    inlet_compressor_gap_mm: float = 0.0
    compressor_combustor_gap_mm: float = 0.0
    combustor_turbine_gap_mm: float = 0.0
    turbine_nozzle_gap_mm: float = 0.0

    # Outer casing
    casing_thickness_mm: float = 2.0
    casing_clearance_mm: float = 1.0      # Clearance between components and casing

    # Shaft
    shaft_diameter_mm: float = 10.0
    shaft_material: str = "ss316l"


@dataclass
class EngineAssembly:
    """Complete engine assembly with all components positioned."""
    # Component geometries
    inlet_geo: Optional[InletGeometry] = None
    compressor_geo: Optional[CompressorGeometry] = None
    combustor_geo: Optional[CombustorGeometry] = None
    turbine_geo: Optional[TurbineGeometry] = None
    nozzle_geo: Optional[NozzleGeometry] = None

    # Axial positions (start of each component)
    component_positions: Dict[str, float] = field(default_factory=dict)

    # Total engine metrics
    total_length_mm: float = 0.0
    max_diameter_mm: float = 0.0
    total_mass_kg: float = 0.0
    mass_breakdown: Dict[str, float] = field(default_factory=dict)


def assemble_engine(params: EngineAssemblyParams) -> EngineAssembly:
    """
    Assemble complete engine from component parameters.
    Computes all geometries and positions them axially.
    """
    asm = EngineAssembly()

    # Compute all component geometries
    asm.inlet_geo = compute_inlet_geometry(params.inlet)
    asm.compressor_geo = compute_compressor_geometry(params.compressor)
    asm.combustor_geo = compute_combustor_geometry(params.combustor)
    asm.turbine_geo = compute_turbine_geometry(params.turbine)
    asm.nozzle_geo = compute_nozzle_geometry(params.nozzle)

    # Axial positioning (front to back)
    z = 0.0

    asm.component_positions['inlet'] = z
    z += params.inlet.length_mm + params.inlet_compressor_gap_mm

    asm.component_positions['compressor'] = z
    z += params.compressor.axial_length_mm + params.compressor_combustor_gap_mm

    asm.component_positions['combustor'] = z
    z += params.combustor.length_mm + params.combustor_turbine_gap_mm

    asm.component_positions['turbine'] = z
    z += params.turbine.stage_axial_length_mm + params.turbine_nozzle_gap_mm

    asm.component_positions['nozzle'] = z
    z += params.nozzle.length_mm

    asm.total_length_mm = z

    # Maximum diameter
    asm.max_diameter_mm = max(
        params.inlet.inlet_diameter_mm,
        params.compressor.impeller_tip_diameter_mm * params.compressor.diffuser_radius_ratio * 2,
        params.combustor.outer_diameter_mm + 2 * params.combustor.casing_thickness_mm,
        params.turbine.tip_diameter_mm,
        params.nozzle.inlet_diameter_mm
    )

    # Mass breakdown
    asm.mass_breakdown = {
        'inlet': get_inlet_mass_kg(params.inlet),
        'compressor': get_compressor_mass_kg(params.compressor),
        'combustor': get_combustor_mass_kg(params.combustor),
        'turbine': get_turbine_mass_kg(params.turbine),
        'nozzle': get_nozzle_mass_kg(params.nozzle),
    }

    # Shaft mass estimate
    shaft_r = params.shaft_diameter_mm / 2000.0
    shaft_l = asm.total_length_mm / 1000.0
    shaft_density = 7990.0  # SS316L
    asm.mass_breakdown['shaft'] = np.pi * shaft_r**2 * shaft_l * shaft_density

    # Misc (bearings, fasteners, seals) — rough 15% adder
    subtotal = sum(asm.mass_breakdown.values())
    asm.mass_breakdown['misc'] = subtotal * 0.15

    asm.total_mass_kg = sum(asm.mass_breakdown.values())

    return asm


def print_engine_summary(asm: EngineAssembly, params: EngineAssemblyParams):
    """Print a detailed engine assembly summary."""
    print(f"\n{'='*55}")
    print(f"  NovaTurbo Engine Assembly: {params.name}")
    print(f"{'='*55}")
    print(f"\n  Total length:   {asm.total_length_mm:.1f} mm")
    print(f"  Max diameter:   {asm.max_diameter_mm:.1f} mm")
    print(f"  Total mass:     {asm.total_mass_kg*1000:.0f} g ({asm.total_mass_kg:.3f} kg)")

    print(f"\n  Component Positions (axial):")
    for comp, pos in asm.component_positions.items():
        print(f"    {comp:15s} starts at z = {pos:.1f} mm")

    print(f"\n  Mass Breakdown:")
    for comp, mass in asm.mass_breakdown.items():
        pct = mass / asm.total_mass_kg * 100
        bar = '#' * int(pct / 3)
        print(f"    {comp:15s} {mass*1000:7.1f} g  ({pct:5.1f}%) {bar}")

    print(f"\n  Key Operating Parameters:")
    print(f"    Compressor PR:  {params.compressor.pressure_ratio}")
    print(f"    RPM:            {params.compressor.rpm:.0f}")
    print(f"    TIT:            {params.turbine.inlet_temperature_K:.0f} K "
          f"({params.turbine.inlet_temperature_K - 273.15:.0f} °C)")
    print(f"    Tip speed:      {asm.compressor_geo.tip_speed_m_s:.1f} m/s (compressor)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    params = EngineAssemblyParams()
    asm = assemble_engine(params)
    print_engine_summary(asm, params)
