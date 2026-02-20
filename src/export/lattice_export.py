"""
NovaTurbo — Lattice STL Export

Exports TPMS lattice structures as STL files for 3D printing
and web viewer visualization. Supports multiple design variations.
"""

import numpy as np
import os
from typing import Optional

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


# Design variations: different TPMS types, densities, cell sizes
LATTICE_VARIATIONS = {
    'v1_gyroid_standard': {
        'label': 'V1 — Gyroid Standard',
        'description': 'Classic gyroid lattice — balanced heat transfer & strength',
        'combustor': {'tpms_type': 'GYROID', 'cell_size_mm': 12.0, 'wall_thickness': 0.5},
        'nozzle': {'tpms_type': 'GYROID', 'cell_size_mm': 10.0, 'wall_thickness': 0.5},
    },
    'v2_gyroid_dense': {
        'label': 'V2 — Gyroid Dense',
        'description': 'Dense gyroid — maximum heat exchange, heavier',
        'combustor': {'tpms_type': 'GYROID', 'cell_size_mm': 10.0, 'wall_thickness': 0.5},
        'nozzle': {'tpms_type': 'GYROID', 'cell_size_mm': 8.0, 'wall_thickness': 0.5},
    },
    'v3_schwarz_p': {
        'label': 'V3 — Schwarz-P',
        'description': 'Schwarz Primitive — straight-through flow channels',
        'combustor': {'tpms_type': 'SCHWARZ_P', 'cell_size_mm': 12.0, 'wall_thickness': 0.5},
        'nozzle': {'tpms_type': 'SCHWARZ_P', 'cell_size_mm': 10.0, 'wall_thickness': 0.5},
    },
    'v4_diamond': {
        'label': 'V4 — Diamond',
        'description': 'Diamond surface — highest stiffness-to-weight ratio',
        'combustor': {'tpms_type': 'DIAMOND', 'cell_size_mm': 12.0, 'wall_thickness': 0.5},
        'nozzle': {'tpms_type': 'DIAMOND', 'cell_size_mm': 10.0, 'wall_thickness': 0.5},
    },
}


def export_lattice_stl(verts: np.ndarray, faces: np.ndarray,
                        filepath: str, verbose: bool = True,
                        min_face_ratio: float = 0.01) -> str:
    """Export lattice vertices/faces to STL file.
    Filters out small disconnected fragments, keeping only
    connected components with > min_face_ratio of total faces.
    """
    if not HAS_TRIMESH:
        raise ImportError("trimesh required for STL export")

    if len(verts) == 0:
        if verbose:
            print("  Warning: No lattice geometry to export")
        return ""

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.fix_normals()

    # Remove degenerate faces
    mask = mesh.nondegenerate_faces()
    mesh.update_faces(mask)

    # Keep only large connected components (removes boundary fragments)
    parts = mesh.split()
    if len(parts) > 1:
        min_faces = max(int(len(mesh.faces) * min_face_ratio), 100)
        large_parts = [p for p in parts if len(p.faces) >= min_faces]
        if large_parts:
            mesh = trimesh.util.concatenate(large_parts)
            if verbose:
                print(f"  Filtered: {len(parts)} components -> {len(large_parts)}")

    mesh.export(filepath)

    if verbose:
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  -> {len(mesh.vertices):,} verts, "
              f"{len(mesh.faces):,} faces, {size_kb:.0f} KB")

    return filepath


def export_engine_lattice(output_dir: str = "exports/stl",
                           verbose: bool = True,
                           variation: str = 'v1_gyroid_standard',
                           assembly_params=None) -> dict:
    """
    Generate and export TPMS lattice structures for a specific variation.
    """
    if not HAS_TRIMESH:
        print("ERROR: trimesh required")
        return {}

    from src.geometry.lattice import (
        LatticeParams, TPMSType,
        generate_combustor_lattice, generate_nozzle_lattice
    )
    from src.geometry.assembly import EngineAssemblyParams, assemble_engine

    os.makedirs(output_dir, exist_ok=True)

    params = assembly_params if assembly_params is not None else EngineAssemblyParams()
    asm = assemble_engine(params)
    positions = asm.component_positions

    var_config = LATTICE_VARIATIONS.get(variation, LATTICE_VARIATIONS['v1_gyroid_standard'])
    exported = {}

    if verbose:
        print(f"\n  NovaTurbo Lattice Export — {var_config['label']}")
        print(f"  {var_config['description']}\n")

    # Grid resolution (voxels/cell) — balanced for browser-friendly mesh size
    grid_res = 7

    # Combustor lattice
    cc = var_config['combustor']
    if verbose:
        print(f"  Combustor: {cc['tpms_type']} cell={cc['cell_size_mm']}mm t={cc['wall_thickness']}")
    lp = LatticeParams(
        tpms_type=TPMSType[cc['tpms_type']],
        cell_size_mm=cc['cell_size_mm'],
        wall_thickness=cc['wall_thickness'],
        resolution=grid_res,
    )
    verts, faces = generate_combustor_lattice(
        combustor_params=params.combustor,
        z_offset=positions['combustor'],
        lattice_params=lp,
    )
    if len(verts) > 0:
        path = os.path.join(output_dir, "combustor_lattice.stl")
        export_lattice_stl(verts, faces, path, verbose)
        exported['combustor_lattice'] = path

    # Nozzle lattice
    nc = var_config['nozzle']
    if verbose:
        print(f"  Nozzle: {nc['tpms_type']} cell={nc['cell_size_mm']}mm t={nc['wall_thickness']}")
    lp2 = LatticeParams(
        tpms_type=TPMSType[nc['tpms_type']],
        cell_size_mm=nc['cell_size_mm'],
        wall_thickness=nc['wall_thickness'],
        resolution=grid_res,
    )
    verts, faces = generate_nozzle_lattice(
        nozzle_params=params.nozzle,
        z_offset=positions['nozzle'],
        lattice_params=lp2,
    )
    if len(verts) > 0:
        path = os.path.join(output_dir, "nozzle_lattice.stl")
        export_lattice_stl(verts, faces, path, verbose)
        exported['nozzle_lattice'] = path

    if verbose:
        print(f"\n  Exported: {len(exported)} lattice files")

    return exported


def export_all_variations(output_dir: str = "exports/stl",
                           verbose: bool = True) -> dict:
    """Generate and export all 4 lattice design variations."""
    all_exported = {}

    for var_key, var_config in LATTICE_VARIATIONS.items():
        var_dir = os.path.join(output_dir, var_key)
        if verbose:
            print(f"\n{'='*60}")
        exported = export_engine_lattice(var_dir, verbose, var_key)
        all_exported[var_key] = exported

    return all_exported


if __name__ == "__main__":
    export_all_variations(verbose=True)
