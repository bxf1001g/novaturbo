"""
NovaTurbo — OpenFOAM Cartesian Mesh Case Generator

Generates complete OpenFOAM case directories from NovaTurbo engine STL
geometry. Uses Cartesian meshing via blockMesh + snappyHexMesh to avoid
singularities (as recommended for complex internal geometries).

Case structure generated:
    case_dir/
        constant/
            triSurface/      ← Engine STL files copied here
            turbulenceProperties
            transportProperties
            thermophysicalProperties
        system/
            blockMeshDict     ← Background Cartesian hex mesh
            snappyHexMeshDict ← Surface snapping + refinement
            controlDict
            fvSchemes
            fvSolution
            decomposeParDict
        0/
            U  p  T  k  omega  nut  alphat

Solver: rhoSimpleFoam (steady compressible RANS) or rhoPimpleFoam (transient)
Turbulence: k-omega SST (standard for turbomachinery)
"""

import os
import shutil
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

import numpy as np


@dataclass
class CFDCaseConfig:
    """Configuration for OpenFOAM case generation."""
    # Solver
    solver: str = "rhoSimpleFoam"       # "rhoSimpleFoam" or "rhoPimpleFoam"
    turbulence_model: str = "kOmegaSST"

    # Mesh resolution
    base_cell_size_mm: float = 4.0      # Background hex cell size
    refinement_levels: int = 3          # snappyHexMesh surface refinement
    boundary_layers: int = 3            # Prism layers on walls
    layer_expansion: float = 1.2        # Layer thickness expansion ratio
    first_layer_mm: float = 0.1         # First prism layer thickness

    # Domain (auto-computed from STL bounding box + margins)
    domain_margin_mm: float = 50.0      # Margin around engine geometry
    inlet_extension_mm: float = 80.0    # Extended inlet region for flow dev
    outlet_extension_mm: float = 120.0  # Extended outlet for wake capture

    # Boundary conditions
    inlet_mass_flow_kg_s: float = 0.28  # Compressor face mass flow
    inlet_temperature_K: float = 288.15 # Ambient air temperature
    outlet_pressure_Pa: float = 101325  # Atmospheric back-pressure
    wall_temperature_K: float = 0.0     # 0 = adiabatic walls

    # Operating conditions (for initialisation)
    reference_pressure_Pa: float = 101325
    reference_velocity_m_s: float = 50.0  # Estimate for initial field

    # Solver controls
    n_iterations: int = 2000
    write_interval: int = 500
    n_procs: int = 4                    # Parallel decomposition

    # Mesh quality gate
    auto_remesh_on_fail: bool = True
    max_remesh_attempts: int = 2


def generate_openfoam_case(
    case_dir: str,
    stl_dir: str = "exports/stl",
    config: Optional[CFDCaseConfig] = None,
    engine_params=None,
    verbose: bool = True,
) -> dict:
    """
    Generate a complete OpenFOAM case directory from engine STL files.

    Parameters
    ----------
    case_dir : str
        Output case directory path
    stl_dir : str
        Directory containing engine STL files
    config : CFDCaseConfig
        Meshing and solver configuration
    engine_params : EngineAssemblyParams, optional
        Engine parameters (for computing BCs from thermodynamics)
    verbose : bool
        Print progress

    Returns
    -------
    dict with keys: case_dir, stl_files, config, bounding_box
    """
    if config is None:
        config = CFDCaseConfig()

    if verbose:
        print(f"\n  OpenFOAM Case Generator")
        print(f"  Solver: {config.solver}  |  Turbulence: {config.turbulence_model}")
        print(f"  Base cell: {config.base_cell_size_mm}mm  |  Refinement: L{config.refinement_levels}")
        print(f"  Prism layers: {config.boundary_layers}  |  First layer: {config.first_layer_mm}mm")

    # Create directory structure
    for subdir in [
        "constant/triSurface",
        "system",
        "0",
    ]:
        os.makedirs(os.path.join(case_dir, subdir), exist_ok=True)

    # Copy STL files and compute bounding box
    stl_files, bbox = _copy_stl_files(case_dir, stl_dir, verbose)

    if not stl_files:
        raise FileNotFoundError(f"No STL files found in {stl_dir}")

    # Compute domain bounds from bbox + margins
    domain = _compute_domain(bbox, config)

    # Generate all OpenFOAM dictionaries
    _write_block_mesh_dict(case_dir, domain, config)
    _write_snappy_hex_mesh_dict(case_dir, stl_files, config)
    _write_control_dict(case_dir, config)
    _write_fv_schemes(case_dir, config)
    _write_fv_solution(case_dir, config)
    _write_decompose_par_dict(case_dir, config)
    _write_turbulence_properties(case_dir, config)
    _write_transport_properties(case_dir)
    _write_thermophysical_properties(case_dir)
    _write_boundary_conditions(case_dir, config)

    # Write Allrun script
    _write_allrun(case_dir, config)

    # Save config for reproducibility
    config_path = os.path.join(case_dir, "novaturbo_config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    if verbose:
        print(f"\n  Case generated: {case_dir}")
        print(f"  STL files: {len(stl_files)}")
        print(f"  Domain: [{domain['x_min']:.0f}, {domain['x_max']:.0f}] x "
              f"[{domain['y_min']:.0f}, {domain['y_max']:.0f}] x "
              f"[{domain['z_min']:.0f}, {domain['z_max']:.0f}] mm")
        print(f"  Run with: cd {case_dir} && ./Allrun")

    return {
        'case_dir': case_dir,
        'stl_files': stl_files,
        'config': asdict(config),
        'bounding_box': bbox,
        'domain': domain,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _copy_stl_files(case_dir: str, stl_dir: str, verbose: bool) -> tuple:
    """Copy STL files to constant/triSurface and compute bounding box."""
    tri_dir = os.path.join(case_dir, "constant", "triSurface")
    stl_files = []
    all_mins = []
    all_maxs = []

    if not os.path.isdir(stl_dir):
        return [], {}

    for fname in os.listdir(stl_dir):
        if not fname.endswith(".stl"):
            continue
        src = os.path.join(stl_dir, fname)
        dst = os.path.join(tri_dir, fname)
        shutil.copy2(src, dst)
        stl_files.append(fname)

        # Quick bbox from binary STL
        bbox = _stl_bounding_box(src)
        if bbox:
            all_mins.append(bbox[0])
            all_maxs.append(bbox[1])

        if verbose:
            size_kb = os.path.getsize(src) / 1024
            print(f"    STL: {fname} ({size_kb:.0f} KB)")

    if not all_mins:
        return stl_files, {}

    bbox_min = np.min(all_mins, axis=0)
    bbox_max = np.max(all_maxs, axis=0)
    bbox = {
        'x_min': float(bbox_min[0]), 'x_max': float(bbox_max[0]),
        'y_min': float(bbox_min[1]), 'y_max': float(bbox_max[1]),
        'z_min': float(bbox_min[2]), 'z_max': float(bbox_max[2]),
    }
    return stl_files, bbox


def _stl_bounding_box(filepath: str):
    """Read bounding box from binary STL file."""
    try:
        import trimesh
        mesh = trimesh.load(filepath)
        return mesh.bounds[0], mesh.bounds[1]
    except Exception:
        return None


def _compute_domain(bbox: dict, config: CFDCaseConfig) -> dict:
    """Compute Cartesian domain from geometry bbox + margins."""
    m = config.domain_margin_mm
    # Engine axis is along Z; X/Y are radial
    return {
        'x_min': bbox.get('x_min', -60) - m,
        'x_max': bbox.get('x_max', 60) + m,
        'y_min': bbox.get('y_min', -60) - m,
        'y_max': bbox.get('y_max', 60) + m,
        'z_min': bbox.get('z_min', 0) - config.inlet_extension_mm,
        'z_max': bbox.get('z_max', 240) + config.outlet_extension_mm,
    }


def _of_header(cls: str, obj: str) -> str:
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


# ---- system/ files --------------------------------------------------------

def _write_block_mesh_dict(case_dir: str, domain: dict, config: CFDCaseConfig):
    """Generate Cartesian background mesh (blockMeshDict)."""
    cs = config.base_cell_size_mm
    # Convert mm to metres for OpenFOAM
    s = 0.001
    x0, x1 = domain['x_min'] * s, domain['x_max'] * s
    y0, y1 = domain['y_min'] * s, domain['y_max'] * s
    z0, z1 = domain['z_min'] * s, domain['z_max'] * s

    nx = max(int((domain['x_max'] - domain['x_min']) / cs), 10)
    ny = max(int((domain['y_max'] - domain['y_min']) / cs), 10)
    nz = max(int((domain['z_max'] - domain['z_min']) / cs), 10)

    content = _of_header("dictionary", "blockMeshDict") + f"""
convertToMeters 1;

vertices
(
    ({x0:.6f} {y0:.6f} {z0:.6f})
    ({x1:.6f} {y0:.6f} {z0:.6f})
    ({x1:.6f} {y1:.6f} {z0:.6f})
    ({x0:.6f} {y1:.6f} {z0:.6f})
    ({x0:.6f} {y0:.6f} {z1:.6f})
    ({x1:.6f} {y0:.6f} {z1:.6f})
    ({x1:.6f} {y1:.6f} {z1:.6f})
    ({x0:.6f} {y1:.6f} {z1:.6f})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 3 2 1)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (4 5 6 7)
        );
    }}
    walls
    {{
        type wall;
        faces
        (
            (0 1 5 4)
            (2 3 7 6)
            (1 2 6 5)
            (0 4 7 3)
        );
    }}
);

mergePatchPairs
(
);
"""
    _write_file(case_dir, "system/blockMeshDict", content)


def _write_snappy_hex_mesh_dict(case_dir: str, stl_files: list, config: CFDCaseConfig):
    """Generate snappyHexMeshDict for Cartesian surface-snapping mesh."""
    # Build geometry entries for each STL
    geo_entries = []
    refine_entries = []
    layer_entries = []

    for stl in stl_files:
        name = stl.replace(".stl", "")
        geo_entries.append(f"""        {name}
        {{
            type triSurfaceMesh;
            file "{stl}";
        }}""")

        level = config.refinement_levels
        # Combustor and nozzle get extra refinement
        if "combustor" in name.lower() or "nozzle" in name.lower():
            level = min(level + 1, 6)

        refine_entries.append(f"""            {name}
            {{
                level ({max(level-1,1)} {level});
                patchInfo
                {{
                    type wall;
                }}
            }}""")

        layer_entries.append(f"""        "{name}.*"
        {{
            nSurfaceLayers {config.boundary_layers};
        }}""")

    geo_block = "\n".join(geo_entries)
    refine_block = "\n".join(refine_entries)
    layer_block = "\n".join(layer_entries)

    content = _of_header("dictionary", "snappyHexMeshDict") + f"""
castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
{geo_block}
}}

castellatedMeshControls
{{
    maxLocalCells       500000;
    maxGlobalCells      4000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;
    resolveFeatureAngle 30;
    allowFreeStandingZoneFaces true;

    features
    (
    );

    refinementSurfaces
    {{
{refine_block}
    }}

    refinementRegions
    {{
    }}

    locationInMesh (0 0 0.12);  // Point inside the domain, outside engine
}}

snapControls
{{
    nSmoothPatch            3;
    tolerance               2.0;
    nSolveIter              100;
    nRelaxIter              5;
    nFeatureSnapIter        10;
    implicitFeatureSnap     true;
    explicitFeatureSnap     false;
    multiRegionFeatureSnap  false;
}}

addLayersControls
{{
    relativeSizes           true;
    expansionRatio          {config.layer_expansion};
    firstLayerThickness     {config.first_layer_mm * 0.001:.6f};
    minThickness            0.001;
    nGrow                   0;
    featureAngle            60;
    nRelaxIter              5;
    nSmoothSurfaceNormals   1;
    nSmoothNormals          3;
    nSmoothThickness        10;
    maxFaceThicknessRatio   0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle      90;
    nBufferCellsNoExtrude   0;
    nLayerIter              50;

    layers
    {{
{layer_block}
    }}
}}

meshQualityControls
{{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       -1e30;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;

    nSmoothScale        4;
    errorReduction      0.75;
}}

debug 0;
"""
    _write_file(case_dir, "system/snappyHexMeshDict", content)


def _write_control_dict(case_dir: str, config: CFDCaseConfig):
    content = _of_header("dictionary", "controlDict") + f"""
application     {config.solver};

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {config.n_iterations};
deltaT          1;

writeControl    timeStep;
writeInterval   {config.write_interval};
purgeWrite      2;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    forces
    {{
        type            forces;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   10;
        patches         (\".*\");
        rho             rhoInf;
        rhoInf          1.225;
        CofR            (0 0 0.12);
    }}

    fieldAverage
    {{
        type            fieldAverage;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        fields
        (
            U {{ mean on; prime2Mean on; base time; }}
            p {{ mean on; prime2Mean on; base time; }}
            T {{ mean on; prime2Mean off; base time; }}
        );
    }}
}}
"""
    _write_file(case_dir, "system/controlDict", content)


def _write_fv_schemes(case_dir: str, config: CFDCaseConfig):
    content = _of_header("dictionary", "fvSchemes") + """
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
    grad(p)         Gauss linear;
}

divSchemes
{
    default                         none;
    div(phi,U)                      bounded Gauss linearUpwind grad(U);
    div(phi,K)                      bounded Gauss upwind;
    div(phi,h)                      bounded Gauss linearUpwind default;
    div(phi,k)                      bounded Gauss upwind;
    div(phi,omega)                  bounded Gauss upwind;
    div(phi,epsilon)                bounded Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}
"""
    _write_file(case_dir, "system/fvSchemes", content)


def _write_fv_solution(case_dir: str, config: CFDCaseConfig):
    content = _of_header("dictionary", "fvSolution") + """
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
    }

    "(U|h|k|omega|epsilon)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0.01;
    }

    rho
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-6;
        relTol          0.01;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent          yes;
    residualControl
    {
        p               1e-5;
        U               1e-5;
        h               1e-5;
        "(k|omega|epsilon)" 1e-5;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
        rho             0.5;
    }
    equations
    {
        U               0.7;
        h               0.7;
        k               0.7;
        omega           0.7;
        epsilon         0.7;
    }
}
"""
    _write_file(case_dir, "system/fvSolution", content)


def _write_decompose_par_dict(case_dir: str, config: CFDCaseConfig):
    content = _of_header("dictionary", "decomposeParDict") + f"""
numberOfSubdomains  {config.n_procs};

method          scotch;

distributed     no;
"""
    _write_file(case_dir, "system/decomposeParDict", content)


# ---- constant/ files ------------------------------------------------------

def _write_turbulence_properties(case_dir: str, config: CFDCaseConfig):
    if config.turbulence_model == "kOmegaSST":
        ras_model = "kOmegaSST"
    else:
        ras_model = "kEpsilon"

    content = _of_header("dictionary", "turbulenceProperties") + f"""
simulationType  RAS;

RAS
{{
    RASModel        {ras_model};
    turbulence      on;
    printCoeffs     on;
}}
"""
    _write_file(case_dir, "constant/turbulenceProperties", content)


def _write_transport_properties(case_dir: str):
    content = _of_header("dictionary", "transportProperties") + """
// Not used for compressible solver (thermophysicalProperties used instead)
"""
    _write_file(case_dir, "constant/transportProperties", content)


def _write_thermophysical_properties(case_dir: str):
    content = _of_header("dictionary", "thermophysicalProperties") + """
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       sutherland;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}

mixture
{
    specie
    {
        molWeight       28.97;      // Air
    }
    thermodynamics
    {
        Cp              1005;       // J/(kg*K)
        Hf              0;
    }
    transport
    {
        As              1.458e-06;  // Sutherland coefficient
        Ts              110.4;      // Sutherland temperature
    }
}
"""
    _write_file(case_dir, "constant/thermophysicalProperties", content)


# ---- 0/ boundary conditions -----------------------------------------------

def _write_boundary_conditions(case_dir: str, config: CFDCaseConfig):
    """Write initial/boundary condition files for all fields."""
    u_ref = config.reference_velocity_m_s
    p_ref = config.reference_pressure_Pa
    T_in = config.inlet_temperature_K
    T_wall = config.wall_temperature_K
    p_out = config.outlet_pressure_Pa

    # Turbulence estimates (5% intensity, length scale ~ 0.01m)
    I = 0.05
    k_val = 1.5 * (u_ref * I) ** 2
    omega_val = k_val ** 0.5 / (0.09 ** 0.25 * 0.01)  # Cmu^0.25 * l
    nut_val = k_val / omega_val

    wall_T = "zeroGradient" if T_wall == 0 else f"fixedValue;\n        value           uniform {T_wall}"

    # U (velocity)
    _write_file(case_dir, "0/U", _of_header("volVectorField", "U") + f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 {u_ref});

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform (0 0 {u_ref});
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform (0 0 0);
        value           uniform (0 0 {u_ref});
    }}
    walls
    {{
        type            noSlip;
    }}
    ".*"
    {{
        type            noSlip;
    }}
}}
""")

    # p (pressure)
    _write_file(case_dir, "0/p", _of_header("volScalarField", "p") + f"""
dimensions      [1 -1 -2 0 0 0 0];
internalField   uniform {p_ref};

boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform {p_out};
    }}
    walls
    {{
        type            zeroGradient;
    }}
    ".*"
    {{
        type            zeroGradient;
    }}
}}
""")

    # T (temperature)
    _write_file(case_dir, "0/T", _of_header("volScalarField", "T") + f"""
dimensions      [0 0 0 1 0 0 0];
internalField   uniform {T_in};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {T_in};
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform {T_in};
        value           uniform {T_in};
    }}
    walls
    {{
        type            {wall_T};
    }}
    ".*"
    {{
        type            zeroGradient;
    }}
}}
""")

    # k (turbulent kinetic energy)
    _write_file(case_dir, "0/k", _of_header("volScalarField", "k") + f"""
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k_val:.6f};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {k_val:.6f};
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform {k_val:.6f};
        value           uniform {k_val:.6f};
    }}
    walls
    {{
        type            kqRWallFunction;
        value           uniform {k_val:.6f};
    }}
    ".*"
    {{
        type            kqRWallFunction;
        value           uniform {k_val:.6f};
    }}
}}
""")

    # omega (specific dissipation rate)
    _write_file(case_dir, "0/omega", _of_header("volScalarField", "omega") + f"""
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform {omega_val:.2f};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {omega_val:.2f};
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform {omega_val:.2f};
        value           uniform {omega_val:.2f};
    }}
    walls
    {{
        type            omegaWallFunction;
        value           uniform {omega_val:.2f};
    }}
    ".*"
    {{
        type            omegaWallFunction;
        value           uniform {omega_val:.2f};
    }}
}}
""")

    # nut (turbulent viscosity)
    _write_file(case_dir, "0/nut", _of_header("volScalarField", "nut") + f"""
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform {nut_val:.6f};

boundaryField
{{
    inlet
    {{
        type            calculated;
        value           uniform 0;
    }}
    outlet
    {{
        type            calculated;
        value           uniform 0;
    }}
    walls
    {{
        type            nutkWallFunction;
        value           uniform 0;
    }}
    ".*"
    {{
        type            nutkWallFunction;
        value           uniform 0;
    }}
}}
""")

    # alphat (turbulent thermal diffusivity)
    _write_file(case_dir, "0/alphat", _of_header("volScalarField", "alphat") + f"""
dimensions      [1 -1 -1 0 0 0 0];
internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            calculated;
        value           uniform 0;
    }}
    outlet
    {{
        type            calculated;
        value           uniform 0;
    }}
    walls
    {{
        type            compressible::alphatWallFunction;
        Prt             0.85;
        value           uniform 0;
    }}
    ".*"
    {{
        type            compressible::alphatWallFunction;
        Prt             0.85;
        value           uniform 0;
    }}
}}
""")


# ---- Allrun script ---------------------------------------------------------

def _write_allrun(case_dir: str, config: CFDCaseConfig):
    """Write Allrun shell script for automated case execution."""
    script = f"""#!/bin/bash
# NovaTurbo — OpenFOAM case run script
# Generated by NovaTurbo CFD Case Generator
cd "${{0%/*}}" || exit

echo "=== NovaTurbo CFD Case ==="
echo "Solver: {config.solver}"
echo "Procs:  {config.n_procs}"

# 1. Generate background Cartesian mesh
echo "\\n--- blockMesh ---"
blockMesh 2>&1 | tee log.blockMesh

# 2. Surface-snap mesh (Cartesian + snapping = no singularities)
echo "\\n--- snappyHexMesh ---"
snappyHexMesh -overwrite 2>&1 | tee log.snappyHexMesh

# 3. Check mesh quality
echo "\\n--- checkMesh ---"
checkMesh 2>&1 | tee log.checkMesh

# 4. Decompose for parallel run
if [ {config.n_procs} -gt 1 ]; then
    echo "\\n--- decomposePar ---"
    decomposePar 2>&1 | tee log.decomposePar
    echo "\\n--- {config.solver} (parallel) ---"
    mpirun -np {config.n_procs} {config.solver} -parallel 2>&1 | tee log.{config.solver}
    echo "\\n--- reconstructPar ---"
    reconstructPar -latestTime 2>&1 | tee log.reconstructPar
else
    echo "\\n--- {config.solver} (serial) ---"
    {config.solver} 2>&1 | tee log.{config.solver}
fi

echo "\\n=== Done ==="
"""
    path = os.path.join(case_dir, "Allrun")
    with open(path, "w", newline="\n") as f:
        f.write(script)

    # Also write an Allclean
    clean = """#!/bin/bash
cd "${0%/*}" || exit
foamCleanCase
rm -rf constant/polyMesh
rm -f log.*
echo "Cleaned."
"""
    path = os.path.join(case_dir, "Allclean")
    with open(path, "w", newline="\n") as f:
        f.write(clean)


def _write_file(case_dir: str, relative_path: str, content: str):
    """Write a file inside the case directory."""
    full = os.path.join(case_dir, relative_path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w", newline="\n") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "exports/cfd_case"
    stl = sys.argv[2] if len(sys.argv) > 2 else "exports/stl"
    result = generate_openfoam_case(out, stl)
    print(f"\nCase ready at: {result['case_dir']}")
