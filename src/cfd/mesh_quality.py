"""
NovaTurbo — Mesh Quality Checker

Validates OpenFOAM mesh quality metrics and flags unsafe cells.
Acts as an AI "safe mesh" gate: if quality fails, the case is
re-meshed with tighter parameters before running the solver.

Reference thresholds from OpenFOAM best practices:
  - Max non-orthogonality < 70° (ideal < 40°)
  - Max skewness < 4.0 (ideal < 1.0)
  - Max aspect ratio < 100 (ideal < 20)
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class MeshQualityReport:
    """Results of a mesh quality check."""
    is_safe: bool = False
    cells: int = 0
    points: int = 0
    faces: int = 0

    # Non-orthogonality (angle between face normal and cell-centre vector)
    max_non_ortho: float = 0.0
    avg_non_ortho: float = 0.0
    non_ortho_ok: bool = False

    # Skewness (offset of face centre from ideal position)
    max_skewness: float = 0.0
    skewness_ok: bool = False

    # Aspect ratio (longest/shortest cell edge)
    max_aspect_ratio: float = 0.0
    aspect_ratio_ok: bool = False

    # Overall
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "SAFE" if self.is_safe else "UNSAFE"
        lines = [
            f"Mesh Quality: {status}",
            f"  Cells: {self.cells:,}  Points: {self.points:,}  Faces: {self.faces:,}",
            f"  Non-orthogonality: max={self.max_non_ortho:.1f}° avg={self.avg_non_ortho:.1f}° {'OK' if self.non_ortho_ok else 'FAIL'}",
            f"  Skewness: max={self.max_skewness:.3f} {'OK' if self.skewness_ok else 'FAIL'}",
            f"  Aspect ratio: max={self.max_aspect_ratio:.1f} {'OK' if self.aspect_ratio_ok else 'FAIL'}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        return "\n".join(lines)


# Thresholds — strict for safety, relaxed for retry
THRESHOLDS_STRICT = {
    'max_non_ortho': 65.0,
    'max_skewness': 3.0,
    'max_aspect_ratio': 50.0,
}
THRESHOLDS_RELAXED = {
    'max_non_ortho': 75.0,
    'max_skewness': 5.0,
    'max_aspect_ratio': 100.0,
}


def check_mesh_quality(case_dir: str,
                       thresholds: Optional[dict] = None,
                       run_check_mesh: bool = True) -> MeshQualityReport:
    """
    Check mesh quality for an OpenFOAM case.

    If run_check_mesh=True, runs `checkMesh` utility and parses output.
    Otherwise, reads the last checkMesh log if available.
    """
    report = MeshQualityReport()
    thr = thresholds or THRESHOLDS_STRICT

    log_text = ""
    if run_check_mesh:
        try:
            proc = subprocess.run(
                ["checkMesh", "-case", case_dir],
                capture_output=True, text=True, timeout=120,
            )
            log_text = proc.stdout + proc.stderr
        except FileNotFoundError:
            report.errors.append("checkMesh not found — is OpenFOAM installed?")
            return report
        except subprocess.TimeoutExpired:
            report.errors.append("checkMesh timed out after 120s")
            return report
    else:
        log_path = os.path.join(case_dir, "log.checkMesh")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log_text = f.read()
        else:
            report.errors.append("No checkMesh log found")
            return report

    # Parse checkMesh output
    report = _parse_check_mesh(log_text, thr)
    return report


def _parse_check_mesh(log: str, thr: dict) -> MeshQualityReport:
    """Parse checkMesh output into a quality report."""
    report = MeshQualityReport()

    m = re.search(r'cells:\s+(\d+)', log)
    if m:
        report.cells = int(m.group(1))
    m = re.search(r'points:\s+(\d+)', log)
    if m:
        report.points = int(m.group(1))
    m = re.search(r'faces:\s+(\d+)', log)
    if m:
        report.faces = int(m.group(1))

    # Non-orthogonality
    m = re.search(r'Max non-orthogonality\s*[=:]\s*([\d.]+)', log, re.IGNORECASE)
    if m:
        report.max_non_ortho = float(m.group(1))
    m = re.search(r'average non-orthogonality\s*[=:]\s*([\d.]+)', log, re.IGNORECASE)
    if m:
        report.avg_non_ortho = float(m.group(1))
    report.non_ortho_ok = report.max_non_ortho <= thr.get('max_non_ortho', 65)

    # Skewness
    m = re.search(r'Max skewness\s*[=:]\s*([\d.]+)', log, re.IGNORECASE)
    if m:
        report.max_skewness = float(m.group(1))
    report.skewness_ok = report.max_skewness <= thr.get('max_skewness', 3.0)

    # Aspect ratio
    m = re.search(r'Max aspect ratio\s*[=:]\s*([\d.]+)', log, re.IGNORECASE)
    if m:
        report.max_aspect_ratio = float(m.group(1))
    report.aspect_ratio_ok = report.max_aspect_ratio <= thr.get('max_aspect_ratio', 50)

    # Warnings
    if report.max_non_ortho > 40:
        report.warnings.append(f"Non-orthogonality ({report.max_non_ortho:.0f}°) above ideal (<40°)")
    if report.max_skewness > 1.0:
        report.warnings.append(f"Skewness ({report.max_skewness:.2f}) above ideal (<1.0)")
    if report.max_aspect_ratio > 20:
        report.warnings.append(f"Aspect ratio ({report.max_aspect_ratio:.0f}) above ideal (<20)")

    report.is_safe = report.non_ortho_ok and report.skewness_ok and report.aspect_ratio_ok
    return report


def suggest_refinement(report: MeshQualityReport) -> dict:
    """Suggest snappyHexMesh parameter adjustments for failing mesh."""
    adjustments = {}

    if not report.non_ortho_ok:
        adjustments['nSmoothPatch'] = 5        # more smoothing iterations
        adjustments['nRelaxIter'] = 8           # more snapping relaxation
        adjustments['nFeatureSnapIter'] = 15

    if not report.skewness_ok:
        adjustments['maxBoundarySkewness'] = 3.0
        adjustments['maxInternalSkewness'] = 3.0
        adjustments['nSmoothScale'] = 6

    if not report.aspect_ratio_ok:
        adjustments['minRefinementCells'] = 5
        adjustments['resolveFeatureAngle'] = 20  # finer feature resolution

    return adjustments
