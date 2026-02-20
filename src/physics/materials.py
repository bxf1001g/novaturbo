"""
NovaTurbo — Material Properties Lookup

Loads material data from config/materials.yaml and provides
lookup functions for thermal/mechanical property validation.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class MaterialProperties:
    """Material properties for engine components."""
    name: str = ""
    category: str = ""
    density_kg_m3: float = 0.0
    melting_point_K: float = 0.0
    max_service_temp_K: float = 0.0
    max_short_term_temp_K: float = 0.0
    yield_strength_MPa: float = 0.0
    ultimate_strength_MPa: float = 0.0
    thermal_conductivity_W_mK: float = 0.0
    specific_heat_J_kgK: float = 0.0
    thermal_expansion_1e6_K: float = 0.0
    youngs_modulus_GPa: float = 0.0
    printable_dmls: bool = False
    min_wall_thickness_mm: float = 0.4
    cost_per_kg_usd: float = 0.0


# Default material database (in case YAML not available)
DEFAULT_MATERIALS = {
    'inconel_718': MaterialProperties(
        name="Inconel 718", category="nickel_superalloy",
        density_kg_m3=8190, melting_point_K=1609,
        max_service_temp_K=973, max_short_term_temp_K=1253,
        yield_strength_MPa=1034, ultimate_strength_MPa=1241,
        thermal_conductivity_W_mK=11.4, specific_heat_J_kgK=435,
        thermal_expansion_1e6_K=13.0, youngs_modulus_GPa=200,
        printable_dmls=True, min_wall_thickness_mm=0.4, cost_per_kg_usd=45
    ),
    'inconel_625': MaterialProperties(
        name="Inconel 625", category="nickel_superalloy",
        density_kg_m3=8440, melting_point_K=1623,
        max_service_temp_K=1253, max_short_term_temp_K=1373,
        yield_strength_MPa=758, ultimate_strength_MPa=965,
        thermal_conductivity_W_mK=9.8, specific_heat_J_kgK=410,
        thermal_expansion_1e6_K=12.8, youngs_modulus_GPa=206,
        printable_dmls=True, min_wall_thickness_mm=0.4, cost_per_kg_usd=55
    ),
    'ti6al4v': MaterialProperties(
        name="Ti-6Al-4V", category="titanium_alloy",
        density_kg_m3=4430, melting_point_K=1933,
        max_service_temp_K=673, max_short_term_temp_K=773,
        yield_strength_MPa=880, ultimate_strength_MPa=950,
        thermal_conductivity_W_mK=6.7, specific_heat_J_kgK=526,
        thermal_expansion_1e6_K=8.6, youngs_modulus_GPa=114,
        printable_dmls=True, min_wall_thickness_mm=0.3, cost_per_kg_usd=120
    ),
    'ss316l': MaterialProperties(
        name="Stainless Steel 316L", category="stainless_steel",
        density_kg_m3=7990, melting_point_K=1673,
        max_service_temp_K=1143, max_short_term_temp_K=1223,
        yield_strength_MPa=205, ultimate_strength_MPa=515,
        thermal_conductivity_W_mK=16.3, specific_heat_J_kgK=500,
        thermal_expansion_1e6_K=16.0, youngs_modulus_GPa=193,
        printable_dmls=True, min_wall_thickness_mm=0.3, cost_per_kg_usd=8
    ),
}


def load_materials(config_path: Optional[str] = None) -> Dict[str, MaterialProperties]:
    """Load materials from YAML config or return defaults."""
    if config_path and yaml:
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            materials = {}
            for key, props in data.get('materials', {}).items():
                mat = MaterialProperties()
                for field_name, value in props.items():
                    if hasattr(mat, field_name):
                        setattr(mat, field_name, value)
                materials[key] = mat
            return materials
        except Exception:
            pass
    return DEFAULT_MATERIALS.copy()


def check_thermal_limits(material_key: str, operating_temp_K: float,
                         materials: Optional[Dict] = None) -> dict:
    """
    Check if operating temperature is within material limits.
    Returns dict with status and margin.
    """
    if materials is None:
        materials = DEFAULT_MATERIALS

    mat = materials.get(material_key)
    if mat is None:
        return {'valid': False, 'error': f'Unknown material: {material_key}'}

    margin_continuous = mat.max_service_temp_K - operating_temp_K
    margin_short_term = mat.max_short_term_temp_K - operating_temp_K

    return {
        'valid': operating_temp_K <= mat.max_service_temp_K,
        'material': mat.name,
        'operating_temp_K': operating_temp_K,
        'max_continuous_K': mat.max_service_temp_K,
        'max_short_term_K': mat.max_short_term_temp_K,
        'margin_continuous_K': margin_continuous,
        'margin_short_term_K': margin_short_term,
        'status': 'OK' if margin_continuous > 0 else
                  'SHORT_TERM_ONLY' if margin_short_term > 0 else
                  'EXCEEDS_LIMITS'
    }


def recommend_material(operating_temp_K: float, component: str,
                       materials: Optional[Dict] = None) -> str:
    """Recommend the best material for given temperature and component type."""
    if materials is None:
        materials = DEFAULT_MATERIALS

    candidates = []
    for key, mat in materials.items():
        if mat.max_service_temp_K >= operating_temp_K and mat.printable_dmls:
            candidates.append((key, mat))

    if not candidates:
        return 'inconel_625'  # Fallback to highest temp material

    # Sort by cost (cheapest suitable material)
    candidates.sort(key=lambda x: x[1].cost_per_kg_usd)
    return candidates[0][0]


if __name__ == "__main__":
    materials = load_materials()
    print("=== NovaTurbo Material Database ===\n")
    for key, mat in materials.items():
        print(f"  {mat.name}")
        print(f"    Max service temp: {mat.max_service_temp_K} K ({mat.max_service_temp_K-273:.0f} °C)")
        print(f"    Density: {mat.density_kg_m3} kg/m³")
        print(f"    Yield strength: {mat.yield_strength_MPa} MPa")
        print(f"    DMLS printable: {'Yes' if mat.printable_dmls else 'No'}")
        print(f"    Cost: ${mat.cost_per_kg_usd}/kg")
        print()

    # Check thermal limits for combustor at TIT
    check = check_thermal_limits('inconel_718', 1100)
    print(f"Inconel 718 at 1100K: {check['status']} (margin: {check['margin_continuous_K']:.0f}K)")

    rec = recommend_material(1100, 'combustor')
    print(f"Recommended material for 1100K combustor: {materials[rec].name}")
