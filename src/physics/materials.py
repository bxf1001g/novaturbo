"""
NovaTurbo — Material Properties & Thermal Barrier Coating System

Loads material data from config/materials.yaml and provides
lookup functions for thermal/mechanical property validation.

Includes TBC (Thermal Barrier Coating) system with bio-inspired
and conventional coating materials for hot-section protection.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List

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


@dataclass
class TBCCoating:
    """Thermal Barrier Coating definition."""
    name: str = ""
    category: str = ""  # "conventional", "bio_inspired", "ceramic"
    thermal_conductivity_W_mK: float = 1.0
    thickness_mm: float = 0.3
    max_surface_temp_K: float = 1500
    emissivity: float = 0.9
    solar_reflectance: float = 0.1
    density_kg_m3: float = 5000
    cost_per_m2_usd: float = 50
    bond_coat: str = "NiCrAlY"
    description: str = ""


# === TBC Coating Database ===
TBC_COATINGS = {
    # --- Conventional Industrial TBCs ---
    'ysz_standard': TBCCoating(
        name="YSZ (7% Yttria-Stabilized Zirconia)",
        category="conventional",
        thermal_conductivity_W_mK=2.0,
        thickness_mm=0.3,
        max_surface_temp_K=1473,
        emissivity=0.5,
        density_kg_m3=5900,
        cost_per_m2_usd=80,
        bond_coat="NiCrAlY",
        description="Industry standard TBC for gas turbines since 1970s"
    ),
    'ysz_ebpvd': TBCCoating(
        name="YSZ EB-PVD (Columnar)",
        category="conventional",
        thermal_conductivity_W_mK=1.5,
        thickness_mm=0.25,
        max_surface_temp_K=1473,
        emissivity=0.45,
        density_kg_m3=5900,
        cost_per_m2_usd=200,
        bond_coat="NiCrAlY",
        description="Electron-beam PVD columnar YSZ, better strain tolerance"
    ),
    'gadolinium_zirconate': TBCCoating(
        name="Gadolinium Zirconate (Gd2Zr2O7)",
        category="ceramic",
        thermal_conductivity_W_mK=1.6,
        thickness_mm=0.3,
        max_surface_temp_K=1573,
        emissivity=0.55,
        density_kg_m3=6100,
        cost_per_m2_usd=150,
        bond_coat="NiCoCrAlY",
        description="Next-gen TBC, lower conductivity than YSZ, higher temp capability"
    ),

    # --- Bio-Inspired Coatings ---
    'diatomite_composite': TBCCoating(
        name="Diatomite-Silica Composite",
        category="bio_inspired",
        thermal_conductivity_W_mK=0.06,
        thickness_mm=0.5,
        max_surface_temp_K=1373,
        emissivity=0.92,
        density_kg_m3=400,
        cost_per_m2_usd=25,
        bond_coat="alumina_interlayer",
        description="Diatom-inspired porous silica; 33x lower conductivity than YSZ"
    ),
    'prismatic_chitin': TBCCoating(
        name="Prismatic Chitin-Ceramic Hybrid",
        category="bio_inspired",
        thermal_conductivity_W_mK=0.03,
        thickness_mm=0.4,
        max_surface_temp_K=1273,
        emissivity=0.95,
        solar_reflectance=0.90,
        density_kg_m3=350,
        cost_per_m2_usd=40,
        bond_coat="alumina_interlayer",
        description="Silver ant-inspired prismatic micro-array; extreme IR reflection"
    ),
    'nacre_layered': TBCCoating(
        name="Nacre-Inspired Layered Ceramic",
        category="bio_inspired",
        thermal_conductivity_W_mK=0.8,
        thickness_mm=0.35,
        max_surface_temp_K=1523,
        emissivity=0.7,
        density_kg_m3=3200,
        cost_per_m2_usd=120,
        bond_coat="NiCrAlY",
        description="Mother-of-pearl structure; alternating ceramic/polymer layers for crack resistance"
    ),
    'aerogel_ceramic': TBCCoating(
        name="Ceramic Aerogel Coating",
        category="bio_inspired",
        thermal_conductivity_W_mK=0.015,
        thickness_mm=0.8,
        max_surface_temp_K=1173,
        emissivity=0.85,
        density_kg_m3=200,
        cost_per_m2_usd=300,
        bond_coat="alumina_interlayer",
        description="Ultra-low density ceramic aerogel; best insulation but lower max temp"
    ),
}


@dataclass
class TBCResult:
    """Result of TBC thermal analysis for a component."""
    component: str = ""
    substrate_material: str = ""
    coating: str = ""
    coating_name: str = ""
    gas_temp_K: float = 0.0
    wall_temp_no_tbc_K: float = 0.0
    coating_surface_temp_K: float = 0.0
    substrate_temp_K: float = 0.0
    temp_drop_across_tbc_K: float = 0.0
    margin_no_tbc_K: float = 0.0
    margin_with_tbc_K: float = 0.0
    improvement_K: float = 0.0
    coating_within_limits: bool = True
    substrate_within_limits: bool = True
    mass_added_g: float = 0.0


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


def _find_materials_config() -> Optional[str]:
    """Auto-detect config/materials.yaml relative to the project root."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(this_dir))
    candidate = os.path.join(project_root, 'config', 'materials.yaml')
    return candidate if os.path.exists(candidate) else None


def load_materials(config_path: Optional[str] = None) -> Dict[str, MaterialProperties]:
    """Load materials from YAML config or return defaults.

    If *config_path* is omitted the function automatically searches for
    ``config/materials.yaml`` relative to the project root so callers don't
    need to know where the file lives.
    """
    if config_path is None:
        config_path = _find_materials_config()

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


# ============================================================
# Thermal Barrier Coating (TBC) Analysis System
# ============================================================

def compute_tbc_temp_drop(gas_temp_K: float, wall_temp_no_tbc_K: float,
                          coating: TBCCoating, substrate_k_W_mK: float,
                          wall_thickness_mm: float = 2.0,
                          h_gas: float = 500.0) -> dict:
    """
    1-D steady-state heat transfer through TBC + substrate.

    Models the thermal resistance network:
      T_gas → [h_gas convection] → T_coating_surface → [TBC conduction] →
      T_substrate_outer → [substrate conduction] → T_substrate_inner

    The TBC acts as a thermal resistor in series, reducing the temperature
    that the metal substrate actually sees.

    Parameters:
        gas_temp_K:       Hot gas total temperature
        wall_temp_no_tbc_K: Wall temperature without any coating
        coating:          TBCCoating object
        substrate_k_W_mK: Substrate metal thermal conductivity
        wall_thickness_mm: Substrate wall thickness
        h_gas:            Gas-side heat transfer coefficient (W/m²K)

    Returns dict with temperatures at each interface.
    """
    t_c = coating.thickness_mm / 1000.0  # coating thickness [m]
    t_s = wall_thickness_mm / 1000.0     # substrate thickness [m]
    k_c = coating.thermal_conductivity_W_mK
    k_s = substrate_k_W_mK

    # Thermal resistance network (per unit area)
    R_conv = 1.0 / h_gas               # gas-side convection
    R_tbc = t_c / k_c                   # TBC conduction
    R_sub = t_s / k_s                   # substrate conduction
    R_total = R_conv + R_tbc + R_sub

    # Heat flux (assume cold side is cooled to ~60% of gas temp, typical)
    T_cool = gas_temp_K * 0.4  # internal cooling air temperature
    q = (gas_temp_K - T_cool) / R_total  # W/m²

    # Temperature at each interface
    T_coating_surface = gas_temp_K - q * R_conv
    T_substrate_outer = T_coating_surface - q * R_tbc
    T_substrate_inner = T_substrate_outer - q * R_sub

    temp_drop = T_coating_surface - T_substrate_outer

    return {
        'heat_flux_W_m2': q,
        'T_coating_surface_K': T_coating_surface,
        'T_substrate_outer_K': T_substrate_outer,
        'T_substrate_inner_K': T_substrate_inner,
        'temp_drop_across_tbc_K': temp_drop,
        'R_conv': R_conv,
        'R_tbc': R_tbc,
        'R_sub': R_sub,
        'R_total': R_total,
        'tbc_fraction_of_resistance': R_tbc / R_total,
    }


def analyze_tbc_for_component(component: str, gas_temp_K: float,
                              wall_temp_no_tbc_K: float,
                              substrate_material_key: str = 'inconel_718',
                              coating_key: str = 'ysz_standard',
                              surface_area_cm2: float = 50.0,
                              materials: Optional[Dict] = None) -> TBCResult:
    """
    Full TBC analysis for a single engine component.
    Computes temperature reduction, margin improvement, and mass penalty.
    """
    if materials is None:
        materials = DEFAULT_MATERIALS

    substrate = materials.get(substrate_material_key, DEFAULT_MATERIALS['inconel_718'])
    coating = TBC_COATINGS.get(coating_key, TBC_COATINGS['ysz_standard'])

    # Run 1-D heat transfer
    ht = compute_tbc_temp_drop(
        gas_temp_K, wall_temp_no_tbc_K, coating,
        substrate.thermal_conductivity_W_mK
    )

    # Mass added by coating
    area_m2 = surface_area_cm2 / 1e4
    mass_added_g = coating.density_kg_m3 * area_m2 * (coating.thickness_mm / 1000.0) * 1000

    result = TBCResult(
        component=component,
        substrate_material=substrate.name,
        coating=coating_key,
        coating_name=coating.name,
        gas_temp_K=gas_temp_K,
        wall_temp_no_tbc_K=wall_temp_no_tbc_K,
        coating_surface_temp_K=ht['T_coating_surface_K'],
        substrate_temp_K=ht['T_substrate_outer_K'],
        temp_drop_across_tbc_K=ht['temp_drop_across_tbc_K'],
        margin_no_tbc_K=substrate.max_service_temp_K - wall_temp_no_tbc_K,
        margin_with_tbc_K=substrate.max_service_temp_K - ht['T_substrate_outer_K'],
        improvement_K=wall_temp_no_tbc_K - ht['T_substrate_outer_K'],
        coating_within_limits=ht['T_coating_surface_K'] <= coating.max_surface_temp_K,
        substrate_within_limits=ht['T_substrate_outer_K'] <= substrate.max_service_temp_K,
        mass_added_g=mass_added_g,
    )
    return result


# Approximate hot-section surface areas for a micro turbojet (cm²)
COMPONENT_SURFACE_AREAS_CM2 = {
    'combustor': 180.0,    # inner + outer liner
    'turbine': 45.0,       # NGV + rotor blade surfaces
    'nozzle': 60.0,        # convergent nozzle interior
}


def run_full_tbc_analysis(coating_key: str = 'diatomite_composite',
                          custom_assignments: Optional[Dict[str, str]] = None) -> dict:
    """
    Run TBC analysis across all hot-section components using the full
    simulation pipeline to get baseline temperatures.

    Parameters:
        coating_key: Which TBC coating to analyze (from TBC_COATINGS)
        custom_assignments: Optional dict of {component: coating_key} overrides

    Returns comprehensive analysis with before/after comparison.
    """
    from src.physics.simulation import run_full_simulation, COMPONENT_MATERIALS

    # Run baseline simulation
    sim = run_full_simulation()
    thermal = sim['thermal']

    # Default: apply coating to combustor, turbine, nozzle
    hot_components = ['combustor', 'turbine', 'nozzle']
    assignments = {comp: coating_key for comp in hot_components}
    if custom_assignments:
        assignments.update(custom_assignments)

    results = []
    total_mass_g = 0.0
    total_improvement_K = 0.0

    for comp in hot_components:
        temps = thermal['component_temps'][comp]
        gas_max = max(temps['gas_temp'])
        wall_max = max(temps['wall_temp'])
        mat = COMPONENT_MATERIALS[comp]

        # Map component material name to key
        mat_key_map = {
            'Inconel 718': 'inconel_718',
            'Inconel 625': 'inconel_625',
            'Ti-6Al-4V': 'ti6al4v',
            'SS 316L': 'ss316l',
        }
        mat_key = mat_key_map.get(mat['name'], 'inconel_718')
        coat_key = assignments.get(comp, coating_key)
        area = COMPONENT_SURFACE_AREAS_CM2.get(comp, 50.0)

        r = analyze_tbc_for_component(
            comp, gas_max, wall_max, mat_key, coat_key, area
        )
        results.append(r)
        total_mass_g += r.mass_added_g
        total_improvement_K += r.improvement_K

    # Build summary
    coating = TBC_COATINGS.get(coating_key, TBC_COATINGS['ysz_standard'])
    summary = {
        'coating_used': coating.name,
        'coating_key': coating_key,
        'coating_category': coating.category,
        'coating_conductivity_W_mK': coating.thermal_conductivity_W_mK,
        'coating_thickness_mm': coating.thickness_mm,
        'total_mass_added_g': round(total_mass_g, 2),
        'avg_temp_improvement_K': round(total_improvement_K / len(results), 1),
        'components': [],
        'all_within_limits': all(r.substrate_within_limits and r.coating_within_limits for r in results),
    }

    for r in results:
        summary['components'].append({
            'component': r.component,
            'substrate': r.substrate_material,
            'coating': r.coating_name,
            'gas_temp_K': round(r.gas_temp_K, 1),
            'wall_no_tbc_K': round(r.wall_temp_no_tbc_K, 1),
            'coating_surface_K': round(r.coating_surface_temp_K, 1),
            'substrate_temp_K': round(r.substrate_temp_K, 1),
            'temp_drop_K': round(r.temp_drop_across_tbc_K, 1),
            'margin_before_K': round(r.margin_no_tbc_K, 1),
            'margin_after_K': round(r.margin_with_tbc_K, 1),
            'improvement_K': round(r.improvement_K, 1),
            'mass_added_g': round(r.mass_added_g, 2),
            'coating_ok': r.coating_within_limits,
            'substrate_ok': r.substrate_within_limits,
        })

    return summary


def compare_all_coatings() -> dict:
    """Compare all TBC coatings across hot-section components."""
    from src.physics.simulation import run_full_simulation, COMPONENT_MATERIALS

    sim = run_full_simulation()
    thermal = sim['thermal']

    comparison = {}
    for coat_key, coating in TBC_COATINGS.items():
        analysis = run_full_tbc_analysis(coat_key)
        comparison[coat_key] = {
            'name': coating.name,
            'category': coating.category,
            'k_W_mK': coating.thermal_conductivity_W_mK,
            'thickness_mm': coating.thickness_mm,
            'max_surface_K': coating.max_surface_temp_K,
            'avg_improvement_K': analysis['avg_temp_improvement_K'],
            'total_mass_g': analysis['total_mass_added_g'],
            'all_ok': analysis['all_within_limits'],
            'components': analysis['components'],
        }

    return comparison


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

    # --- TBC Analysis ---
    print("\n" + "=" * 70)
    print("THERMAL BARRIER COATING ANALYSIS")
    print("=" * 70)

    print("\n--- Available Coatings ---")
    for key, coat in TBC_COATINGS.items():
        print(f"  {coat.name}")
        print(f"    k = {coat.thermal_conductivity_W_mK} W/mK | "
              f"t = {coat.thickness_mm}mm | "
              f"Max: {coat.max_surface_temp_K}K | "
              f"Category: {coat.category}")

    print("\n--- Full Comparison ---\n")
    comparison = compare_all_coatings()
    print(f"{'Coating':<40} | {'k(W/mK)':>8} | {'Avg ΔT':>7} | {'Mass(g)':>8} | {'Status':<6}")
    print("-" * 80)
    for key, data in comparison.items():
        status = "✓ OK" if data['all_ok'] else "⚠ HOT"
        print(f"{data['name']:<40} | {data['k_W_mK']:>8.3f} | {data['avg_improvement_K']:>6.1f}K | {data['total_mass_g']:>7.2f}g | {status}")

    # Detailed analysis for best bio-inspired option
    print("\n--- Detailed: Diatomite Composite ---\n")
    analysis = run_full_tbc_analysis('diatomite_composite')
    for c in analysis['components']:
        print(f"  {c['component']:12s} | Wall: {c['wall_no_tbc_K']:.0f}K → {c['substrate_temp_K']:.0f}K "
              f"(↓{c['improvement_K']:.0f}K) | Margin: {c['margin_before_K']:+.0f}K → {c['margin_after_K']:+.0f}K "
              f"| +{c['mass_added_g']:.1f}g")
    print(f"\n  Total mass added: {analysis['total_mass_added_g']:.1f}g")
    print(f"  Average improvement: {analysis['avg_temp_improvement_K']:.1f}K")
