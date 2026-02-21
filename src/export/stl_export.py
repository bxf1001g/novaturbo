"""
NovaTurbo — STL Export Module

Generates watertight STL meshes for each engine component
using revolution of 2D profiles (no CadQuery dependency needed).

Exports individual component STLs + full engine assembly
ready for Fusion 360 import or direct 3D printing.
"""

import numpy as np
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def _revolve_profile(z_coords: np.ndarray, r_coords: np.ndarray,
                     n_theta: int = 72, angle: float = 2 * np.pi) -> 'trimesh.Trimesh':
    """
    Create a solid of revolution from a 2D profile (z, r).
    Revolves around the Z axis.
    Returns a trimesh.Trimesh object.
    """
    if not HAS_TRIMESH:
        raise ImportError("trimesh is required for STL export. Install with: pip install trimesh")

    n_pts = len(z_coords)
    theta = np.linspace(0, angle, n_theta + 1)[:-1]

    # Generate vertices
    vertices = []
    for i in range(n_pts):
        for j in range(n_theta):
            x = r_coords[i] * np.cos(theta[j])
            y = r_coords[i] * np.sin(theta[j])
            z = z_coords[i]
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Generate faces (quads split into triangles)
    faces = []
    for i in range(n_pts - 1):
        for j in range(n_theta):
            j_next = (j + 1) % n_theta
            # Current ring indices
            v00 = i * n_theta + j
            v01 = i * n_theta + j_next
            v10 = (i + 1) * n_theta + j
            v11 = (i + 1) * n_theta + j_next

            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])

    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    return mesh


def _create_shell(z_outer: np.ndarray, r_outer: np.ndarray,
                  z_inner: np.ndarray, r_inner: np.ndarray,
                  n_theta: int = 72, min_wall_mm: float = 0.8) -> 'trimesh.Trimesh':
    """
    Create a hollow shell (like a combustor liner) by combining
    outer and inner revolution surfaces with end caps.
    Enforces minimum wall thickness to prevent zero-thickness geometry.
    """
    # Enforce minimum wall thickness everywhere
    r_outer = np.array(r_outer, dtype=float)
    r_inner = np.array(r_inner, dtype=float)
    wall = r_outer - np.interp(z_outer, z_inner, r_inner)
    thin = wall < min_wall_mm
    if np.any(thin):
        r_inner_interp = np.interp(z_outer, z_inner, r_inner)
        r_inner_interp[thin] = r_outer[thin] - min_wall_mm
        r_inner_interp = np.maximum(r_inner_interp, 0.1)  # Prevent negative radii
        r_inner = np.interp(z_inner, z_outer, r_inner_interp)

    outer = _revolve_profile(z_outer, r_outer, n_theta)
    inner = _revolve_profile(z_inner, r_inner, n_theta)

    # Flip inner normals to face inward
    inner.invert()

    # Create end caps (annular discs)
    caps = []
    for z_pos, r_out, r_in in [(z_outer[0], r_outer[0], r_inner[0]),
                                 (z_outer[-1], r_outer[-1], r_inner[-1])]:
        cap = _create_annular_disc(z_pos, r_in, r_out, n_theta)
        caps.append(cap)

    # Combine all
    combined = trimesh.util.concatenate([outer, inner] + caps)
    return combined


def _create_annular_disc(z_pos: float, r_inner: float, r_outer: float,
                         n_theta: int = 72) -> 'trimesh.Trimesh':
    """Create a flat annular disc (end cap) at a given Z position."""
    theta = np.linspace(0, 2 * np.pi, n_theta + 1)[:-1]

    vertices = []
    # Inner ring
    for t in theta:
        vertices.append([r_inner * np.cos(t), r_inner * np.sin(t), z_pos])
    # Outer ring
    for t in theta:
        vertices.append([r_outer * np.cos(t), r_outer * np.sin(t), z_pos])

    vertices = np.array(vertices)

    faces = []
    for j in range(n_theta):
        j_next = (j + 1) % n_theta
        v_in = j
        v_in_next = j_next
        v_out = n_theta + j
        v_out_next = n_theta + j_next
        faces.append([v_in, v_out, v_in_next])
        faces.append([v_in_next, v_out, v_out_next])

    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    return mesh


def _create_cylinder(z_start: float, z_end: float, radius: float,
                     n_theta: int = 72) -> 'trimesh.Trimesh':
    """Create a solid cylinder along the Z axis."""
    z = np.array([z_start, z_end])
    r = np.array([radius, radius])
    shell = _revolve_profile(z, r, n_theta)

    # Add end caps (solid discs)
    caps = []
    for z_pos in [z_start, z_end]:
        theta = np.linspace(0, 2 * np.pi, n_theta + 1)[:-1]
        verts = [[0, 0, z_pos]]
        for t in theta:
            verts.append([radius * np.cos(t), radius * np.sin(t), z_pos])
        verts = np.array(verts)
        faces = []
        for j in range(n_theta):
            j_next = (j + 1) % n_theta
            faces.append([0, j + 1, j_next + 1])
        caps.append(trimesh.Trimesh(vertices=verts, faces=np.array(faces)))

    return trimesh.util.concatenate([shell] + caps)


def export_inlet_stl(inlet_geo, params, z_offset: float = 0.0,
                     n_theta: int = 72) -> 'trimesh.Trimesh':
    """Generate STL mesh for the air inlet."""
    z = inlet_geo.inner_contour[:, 0] + z_offset
    r_inner = inlet_geo.inner_contour[:, 1]
    r_outer = inlet_geo.outer_contour[:, 1]
    return _create_shell(z, r_outer, z, r_inner, n_theta)


def export_compressor_stl(comp_geo, params, z_offset: float = 0.0,
                          n_theta: int = 72) -> 'trimesh.Trimesh':
    """Generate STL mesh for the compressor (hub + shroud shell)."""
    # Hub body (solid of revolution)
    z_hub = comp_geo.hub_contour[:, 0] + z_offset
    r_hub = comp_geo.hub_contour[:, 1]

    # Shroud
    z_shroud = comp_geo.shroud_contour[:, 0] + z_offset
    r_shroud = comp_geo.shroud_contour[:, 1]

    # Create shell between shroud (outer) and hub (inner)
    # Interpolate to same Z points
    n_pts = 50
    z_common = np.linspace(z_offset, z_offset + params.axial_length_mm, n_pts)
    r_hub_interp = np.interp(z_common, z_hub, r_hub)
    r_shroud_interp = np.interp(z_common, z_shroud, r_shroud)

    shell = _create_shell(z_common, r_shroud_interp, z_common, r_hub_interp, n_theta)

    # Add simplified blade geometry (thin radial fins)
    blades = []
    for i in range(params.blade_count):
        theta = 2 * np.pi * i / params.blade_count
        blade_thickness = params.blade_thickness_mm

        for z_idx in range(0, n_pts - 1, 3):
            r_in = r_hub_interp[z_idx]
            r_out = r_shroud_interp[z_idx] - params.shroud_clearance_mm
            z1 = z_common[z_idx]
            z2 = z_common[min(z_idx + 3, n_pts - 1)]
            ht = blade_thickness / 2000.0  # Half-thickness in mm for offset

            # Blade as a thin box
            verts = np.array([
                [r_in * np.cos(theta - ht/r_in), r_in * np.sin(theta - ht/r_in), z1],
                [r_in * np.cos(theta + ht/r_in), r_in * np.sin(theta + ht/r_in), z1],
                [r_out * np.cos(theta + ht/r_out), r_out * np.sin(theta + ht/r_out), z1],
                [r_out * np.cos(theta - ht/r_out), r_out * np.sin(theta - ht/r_out), z1],
                [r_in * np.cos(theta - ht/r_in), r_in * np.sin(theta - ht/r_in), z2],
                [r_in * np.cos(theta + ht/r_in), r_in * np.sin(theta + ht/r_in), z2],
                [r_out * np.cos(theta + ht/r_out), r_out * np.sin(theta + ht/r_out), z2],
                [r_out * np.cos(theta - ht/r_out), r_out * np.sin(theta - ht/r_out), z2],
            ])
            box_faces = np.array([
                [0,1,2],[0,2,3],[4,6,5],[4,7,6],
                [0,4,5],[0,5,1],[2,6,7],[2,7,3],
                [0,3,7],[0,7,4],[1,5,6],[1,6,2]
            ])
            blades.append(trimesh.Trimesh(vertices=verts, faces=box_faces))

    if blades:
        all_parts = [shell] + blades
        return trimesh.util.concatenate(all_parts)
    return shell


def export_combustor_stl(comb_geo, params, z_offset: float = 0.0,
                         n_theta: int = 72) -> 'trimesh.Trimesh':
    """Generate STL mesh for the combustion chamber."""
    z = comb_geo.outer_liner_outer[:, 0] + z_offset
    r_outer_out = comb_geo.outer_liner_outer[:, 1]
    r_outer_in = comb_geo.outer_liner_inner[:, 1]
    r_inner_out = comb_geo.inner_liner_outer[:, 1]
    r_inner_in = comb_geo.inner_liner_inner[:, 1]

    # Outer liner shell
    outer_shell = _create_shell(z, r_outer_out, z, r_outer_in, n_theta)
    # Inner liner shell
    inner_shell = _create_shell(z, r_inner_out, z, r_inner_in, n_theta)

    return trimesh.util.concatenate([outer_shell, inner_shell])


def export_turbine_stl(turb_geo, params, z_offset: float = 0.0,
                       n_theta: int = 72) -> 'trimesh.Trimesh':
    """Generate STL mesh for the turbine stage (hub + casing + NGV + rotor blades)."""
    z = turb_geo.hub_contour[:, 0] + z_offset
    r_hub = turb_geo.hub_contour[:, 1]
    r_casing = turb_geo.casing_contour[:, 1]

    # Annular channel
    shell = _create_shell(z, r_casing, z, r_hub, n_theta)

    # Hub disc (solid)
    disc = _create_cylinder(
        z_offset, z_offset + params.stage_axial_length_mm,
        params.hub_diameter_mm / 2.0, n_theta
    )

    parts = [shell, disc]

    # NGV stator vanes
    r_hub_val = params.hub_diameter_mm / 2.0
    r_tip_val = params.tip_diameter_mm / 2.0
    blade_h = r_tip_val - r_hub_val
    ht_ngv = (params.ngv_thickness_ratio * params.ngv_chord_mm) / 2.0

    for i in range(params.ngv_count):
        theta = 2 * np.pi * i / params.ngv_count
        z_le = z_offset + params.ngv_axial_position_mm
        z_te = z_le + params.ngv_chord_mm
        r_in = r_hub_val
        r_out = r_tip_val - 0.5

        verts = np.array([
            [r_in * np.cos(theta - ht_ngv/r_in),  r_in * np.sin(theta - ht_ngv/r_in),  z_le],
            [r_in * np.cos(theta + ht_ngv/r_in),  r_in * np.sin(theta + ht_ngv/r_in),  z_le],
            [r_out * np.cos(theta + ht_ngv/r_out), r_out * np.sin(theta + ht_ngv/r_out), z_le],
            [r_out * np.cos(theta - ht_ngv/r_out), r_out * np.sin(theta - ht_ngv/r_out), z_le],
            [r_in * np.cos(theta - ht_ngv/r_in),  r_in * np.sin(theta - ht_ngv/r_in),  z_te],
            [r_in * np.cos(theta + ht_ngv/r_in),  r_in * np.sin(theta + ht_ngv/r_in),  z_te],
            [r_out * np.cos(theta + ht_ngv/r_out), r_out * np.sin(theta + ht_ngv/r_out), z_te],
            [r_out * np.cos(theta - ht_ngv/r_out), r_out * np.sin(theta - ht_ngv/r_out), z_te],
        ])
        faces = np.array([
            [0,1,2],[0,2,3],[4,6,5],[4,7,6],
            [0,4,5],[0,5,1],[2,6,7],[2,7,3],
            [0,3,7],[0,7,4],[1,5,6],[1,6,2]
        ])
        parts.append(trimesh.Trimesh(vertices=verts, faces=faces))

    # Rotor blades
    ht_rot = (params.blade_thickness_ratio * params.blade_chord_mm) / 2.0
    for i in range(params.blade_count):
        theta = 2 * np.pi * i / params.blade_count
        z_le = z_offset + params.blade_axial_position_mm
        z_te = z_le + params.blade_chord_mm
        r_in = r_hub_val
        r_out = r_tip_val - 0.5

        verts = np.array([
            [r_in * np.cos(theta - ht_rot/r_in),  r_in * np.sin(theta - ht_rot/r_in),  z_le],
            [r_in * np.cos(theta + ht_rot/r_in),  r_in * np.sin(theta + ht_rot/r_in),  z_le],
            [r_out * np.cos(theta + ht_rot/r_out), r_out * np.sin(theta + ht_rot/r_out), z_le],
            [r_out * np.cos(theta - ht_rot/r_out), r_out * np.sin(theta - ht_rot/r_out), z_le],
            [r_in * np.cos(theta - ht_rot/r_in),  r_in * np.sin(theta - ht_rot/r_in),  z_te],
            [r_in * np.cos(theta + ht_rot/r_in),  r_in * np.sin(theta + ht_rot/r_in),  z_te],
            [r_out * np.cos(theta + ht_rot/r_out), r_out * np.sin(theta + ht_rot/r_out), z_te],
            [r_out * np.cos(theta - ht_rot/r_out), r_out * np.sin(theta - ht_rot/r_out), z_te],
        ])
        faces = np.array([
            [0,1,2],[0,2,3],[4,6,5],[4,7,6],
            [0,4,5],[0,5,1],[2,6,7],[2,7,3],
            [0,3,7],[0,7,4],[1,5,6],[1,6,2]
        ])
        parts.append(trimesh.Trimesh(vertices=verts, faces=faces))

    return trimesh.util.concatenate(parts)


def export_nozzle_stl(noz_geo, params, z_offset: float = 0.0,
                      n_theta: int = 72) -> 'trimesh.Trimesh':
    """Generate STL mesh for the exhaust nozzle."""
    z_inner = noz_geo.inner_contour[:, 0] + z_offset
    r_inner = noz_geo.inner_contour[:, 1]
    z_outer = noz_geo.outer_contour[:, 0] + z_offset
    r_outer = noz_geo.outer_contour[:, 1]

    return _create_shell(z_inner, r_outer, z_inner, r_inner, n_theta)


def export_full_engine(output_dir: str = "exports/stl",
                       n_theta: int = 72,
                       verbose: bool = True,
                       assembly_params=None) -> dict:
    """
    Export the complete engine assembly as individual STL files
    and one combined assembly STL with outer casing and transitions.

    Returns dict of {component_name: filepath}
    """
    if not HAS_TRIMESH:
        print("ERROR: trimesh is required. Install with: pip install trimesh")
        return {}

    from src.geometry.assembly import EngineAssemblyParams, assemble_engine

    os.makedirs(output_dir, exist_ok=True)

    params = assembly_params if assembly_params is not None else EngineAssemblyParams()
    asm = assemble_engine(params)
    positions = asm.component_positions

    if verbose:
        print(f"\n  NovaTurbo STL Export (Connected Assembly)")
        print(f"  Output: {os.path.abspath(output_dir)}")
        print(f"  Resolution: {n_theta} segments/revolution\n")

    exported = {}
    meshes = []

    # Inlet
    if verbose:
        print(f"  Exporting inlet...", end=" ")
    mesh = export_inlet_stl(asm.inlet_geo, params.inlet, positions['inlet'], n_theta)
    path = os.path.join(output_dir, "inlet.stl")
    mesh.export(path)
    exported['inlet'] = path
    meshes.append(mesh)
    if verbose:
        print(f"✓ {len(mesh.vertices)} verts")

    # Compressor
    if verbose:
        print(f"  Exporting compressor...", end=" ")
    mesh = export_compressor_stl(asm.compressor_geo, params.compressor, positions['compressor'], n_theta)
    path = os.path.join(output_dir, "compressor.stl")
    mesh.export(path)
    exported['compressor'] = path
    meshes.append(mesh)
    if verbose:
        print(f"✓ {len(mesh.vertices)} verts")

    # Combustor
    if verbose:
        print(f"  Exporting combustor...", end=" ")
    mesh = export_combustor_stl(asm.combustor_geo, params.combustor, positions['combustor'], n_theta)
    path = os.path.join(output_dir, "combustor.stl")
    mesh.export(path)
    exported['combustor'] = path
    meshes.append(mesh)
    if verbose:
        print(f"✓ {len(mesh.vertices)} verts")

    # Turbine
    if verbose:
        print(f"  Exporting turbine...", end=" ")
    mesh = export_turbine_stl(asm.turbine_geo, params.turbine, positions['turbine'], n_theta)
    path = os.path.join(output_dir, "turbine.stl")
    mesh.export(path)
    exported['turbine'] = path
    meshes.append(mesh)
    if verbose:
        print(f"✓ {len(mesh.vertices)} verts")

    # Nozzle
    if verbose:
        print(f"  Exporting nozzle...", end=" ")
    mesh = export_nozzle_stl(asm.nozzle_geo, params.nozzle, positions['nozzle'], n_theta)
    path = os.path.join(output_dir, "nozzle.stl")
    mesh.export(path)
    exported['nozzle'] = path
    meshes.append(mesh)
    if verbose:
        print(f"✓ {len(mesh.vertices)} verts")

    # Shaft (simple cylinder through entire engine)
    if verbose:
        print(f"  Exporting shaft...", end=" ")
    shaft = _create_cylinder(0, asm.total_length_mm, params.shaft_diameter_mm / 2.0, n_theta)
    path = os.path.join(output_dir, "shaft.stl")
    shaft.export(path)
    exported['shaft'] = path
    meshes.append(shaft)
    if verbose:
        print(f"✓ {len(shaft.vertices)} verts")

    # Outer casing — continuous shell wrapping the full engine
    if verbose:
        print(f"  Exporting outer casing...", end=" ")
    casing = _create_outer_casing(asm, params, n_theta)
    path = os.path.join(output_dir, "casing.stl")
    casing.export(path)
    exported['casing'] = path
    meshes.append(casing)
    if verbose:
        print(f"✓ {len(casing.vertices)} verts")

    # Transition sections between components
    if verbose:
        print(f"  Exporting transitions...", end=" ")
    transitions = _create_transitions(asm, params, n_theta)
    path = os.path.join(output_dir, "transitions.stl")
    transitions.export(path)
    exported['transitions'] = path
    meshes.append(transitions)
    if verbose:
        print(f"✓ {len(transitions.vertices)} verts")

    # Full assembly
    if verbose:
        print(f"\n  Combining into full assembly...", end=" ")
    assembly = trimesh.util.concatenate(meshes)
    path = os.path.join(output_dir, "full_engine_assembly.stl")
    assembly.export(path)
    exported['assembly'] = path
    if verbose:
        total_verts = len(assembly.vertices)
        total_faces = len(assembly.faces)
        file_size = os.path.getsize(path) / 1024
        print(f"✓")
        print(f"\n  Assembly: {total_verts:,} vertices, {total_faces:,} faces")
        print(f"  File size: {file_size:.0f} KB")

    if verbose:
        print(f"\n  Files exported:")
        for name, fpath in exported.items():
            size = os.path.getsize(fpath) / 1024
            print(f"    {name:20s} → {fpath} ({size:.0f} KB)")

    return exported


def _create_outer_casing(asm, params, n_theta: int = 72) -> 'trimesh.Trimesh':
    """
    Create a continuous outer casing that wraps the entire engine.
    Follows the max outer radius of each component with clearance,
    providing a smooth external shell.
    """
    positions = asm.component_positions
    casing_t = params.casing_thickness_mm
    clearance = params.casing_clearance_mm
    total_L = asm.total_length_mm

    # Build casing profile: z positions and corresponding outer radii
    # Each section follows the component's outer radius + clearance
    z_points = []
    r_points = []

    # Inlet section
    z_start = positions['inlet']
    z_end = z_start + params.inlet.length_mm
    r = params.inlet.inlet_diameter_mm / 2.0 + clearance
    z_points.extend([z_start, z_end])
    r_points.extend([r, r])

    # Compressor section (wider due to diffuser)
    z_start = positions['compressor']
    z_end = z_start + params.compressor.axial_length_mm
    r = params.compressor.impeller_tip_diameter_mm / 2.0 * params.compressor.diffuser_radius_ratio + clearance
    z_points.extend([z_start, z_end])
    r_points.extend([r, r])

    # Combustor section
    z_start = positions['combustor']
    z_end = z_start + params.combustor.length_mm
    r = params.combustor.outer_diameter_mm / 2.0 + params.combustor.casing_thickness_mm + clearance
    z_points.extend([z_start, z_end])
    r_points.extend([r, r])

    # Turbine section
    z_start = positions['turbine']
    z_end = z_start + params.turbine.stage_axial_length_mm
    r = params.turbine.tip_diameter_mm / 2.0 + clearance
    z_points.extend([z_start, z_end])
    r_points.extend([r, r])

    # Nozzle section
    z_start = positions['nozzle']
    z_end = z_start + params.nozzle.length_mm
    r_noz_start = params.nozzle.inlet_diameter_mm / 2.0 + clearance
    r_noz_end = params.nozzle.exit_diameter_mm / 2.0 + clearance
    z_points.extend([z_start, z_end])
    r_points.extend([r_noz_start, r_noz_end])

    z_arr = np.array(z_points)
    r_arr = np.array(r_points)

    # Sort by z and create a smooth interpolated casing profile
    sort_idx = np.argsort(z_arr)
    z_sorted = z_arr[sort_idx]
    r_sorted = r_arr[sort_idx]

    # Remove duplicate z values (keep max radius)
    z_unique = [z_sorted[0]]
    r_unique = [r_sorted[0]]
    for i in range(1, len(z_sorted)):
        if abs(z_sorted[i] - z_unique[-1]) < 0.01:
            r_unique[-1] = max(r_unique[-1], r_sorted[i])
        else:
            z_unique.append(z_sorted[i])
            r_unique.append(r_sorted[i])

    z_unique = np.array(z_unique)
    r_unique = np.array(r_unique)

    # Interpolate to smooth curve
    n_casing_pts = 100
    z_smooth = np.linspace(z_unique[0], z_unique[-1], n_casing_pts)
    r_smooth = np.interp(z_smooth, z_unique, r_unique)

    # Apply smoothing (moving average) for gentle transitions
    kernel_size = 7
    kernel = np.ones(kernel_size) / kernel_size
    r_padded = np.pad(r_smooth, kernel_size // 2, mode='edge')
    r_smooth = np.convolve(r_padded, kernel, mode='valid')[:n_casing_pts]

    # Outer casing surface
    r_outer = r_smooth + casing_t
    r_inner = r_smooth

    return _create_shell(z_smooth, r_outer, z_smooth, r_inner, n_theta)


def _create_transitions(asm, params, n_theta: int = 72) -> 'trimesh.Trimesh':
    """
    Create transition sections that connect components smoothly.
    These are conical/bell-shaped adapters that bridge diameter differences.
    """
    positions = asm.component_positions
    transitions = []

    # 1. Inlet → Compressor transition
    z_start = positions['inlet'] + params.inlet.length_mm
    z_end = positions['compressor']
    if z_end > z_start:
        z_end = z_start  # No gap, but match diameters
    # Even with 0 gap, add a short adapter ring if diameters differ
    r_from_outer = params.inlet.outlet_diameter_mm / 2.0 + params.inlet.wall_thickness_mm
    r_to_outer = params.compressor.inducer_tip_diameter_mm / 2.0 + params.compressor.shroud_clearance_mm + 1.5
    r_from_inner = params.inlet.outlet_diameter_mm / 2.0
    r_to_inner = params.compressor.inducer_tip_diameter_mm / 2.0
    z_pos = positions['compressor']
    trans_len = 3.0  # 3mm short adapter
    z = np.linspace(z_pos - trans_len, z_pos, 10)
    t = np.linspace(0, 1, 10)
    r_o = r_from_outer + (r_to_outer - r_from_outer) * (3*t**2 - 2*t**3)
    r_i = r_from_inner + (r_to_inner - r_from_inner) * (3*t**2 - 2*t**3)
    transitions.append(_create_shell(z, r_o, z, r_i, n_theta))

    # 2. Compressor → Combustor transition (diffuser to combustor annulus)
    z_pos = positions['combustor']
    r_from = params.compressor.impeller_tip_diameter_mm / 2.0 * params.compressor.diffuser_radius_ratio
    r_to = params.combustor.outer_diameter_mm / 2.0
    trans_len = 5.0
    z = np.linspace(z_pos - trans_len, z_pos, 15)
    t = np.linspace(0, 1, 15)
    r_o = r_from + (r_to - r_from) * (3*t**2 - 2*t**3)
    wall_t = 1.5
    r_i = np.maximum(r_o - wall_t, 0.5)
    transitions.append(_create_shell(z, r_o, z, r_i, n_theta))

    # Inner transition (compressor hub to combustor inner liner)
    r_from_i = params.compressor.impeller_hub_diameter_mm / 2.0
    r_to_i = params.combustor.inner_diameter_mm / 2.0
    r_i_inner = r_from_i + (r_to_i - r_from_i) * (3*t**2 - 2*t**3)
    r_i_outer = np.maximum(r_i_inner + wall_t, r_i_inner + 0.8)
    transitions.append(_create_shell(z, r_i_outer, z, r_i_inner, n_theta))

    # 3. Combustor → Turbine transition
    z_pos = positions['turbine']
    r_from = params.combustor.outer_diameter_mm / 2.0
    r_to = params.turbine.tip_diameter_mm / 2.0
    trans_len = 5.0
    z = np.linspace(z_pos - trans_len, z_pos, 15)
    t = np.linspace(0, 1, 15)
    r_o = r_from + (r_to - r_from) * (3*t**2 - 2*t**3)
    r_i = np.maximum(r_o - wall_t, 0.5)
    transitions.append(_create_shell(z, r_o, z, r_i, n_theta))

    # Inner: combustor inner liner to turbine hub
    r_from_i = params.combustor.inner_diameter_mm / 2.0
    r_to_i = params.turbine.hub_diameter_mm / 2.0
    r_i_inner = r_from_i + (r_to_i - r_from_i) * (3*t**2 - 2*t**3)
    r_i_outer = np.maximum(r_i_inner + wall_t, r_i_inner + 0.8)
    transitions.append(_create_shell(z, r_i_outer, z, r_i_inner, n_theta))

    # 4. Turbine → Nozzle transition
    z_pos = positions['nozzle']
    r_from = params.turbine.tip_diameter_mm / 2.0
    r_to = params.nozzle.inlet_diameter_mm / 2.0
    trans_len = 3.0
    z = np.linspace(z_pos - trans_len, z_pos, 10)
    t = np.linspace(0, 1, 10)
    r_o = r_from + (r_to - r_from) * (3*t**2 - 2*t**3)
    r_i = np.maximum(r_o - wall_t, 0.5)
    transitions.append(_create_shell(z, r_o, z, r_i, n_theta))

    return trimesh.util.concatenate(transitions)


if __name__ == "__main__":
    exported = export_full_engine(verbose=True)
