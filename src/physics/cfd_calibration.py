"""
NovaTurbo — CFD calibration utilities

Bridges OpenFOAM/SU2 results into the UI learning loop by:
1) Running externally configured CFD commands for saved UI variants
2) Fitting affine corrections for thrust and TSFC
3) Applying those corrections to training labels
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Optional
import json
import os
import subprocess

import numpy as np
import pandas as pd


OPENFOAM_CMD_ENV = "NOVATURBO_OPENFOAM_CMD_TEMPLATE"
SU2_CMD_ENV = "NOVATURBO_SU2_CMD_TEMPLATE"


@dataclass
class CFDCalibrationModel:
    thrust_scale: float = 1.0
    thrust_bias: float = 0.0
    tsfc_scale: float = 1.0
    tsfc_bias: float = 0.0
    sample_count: int = 0
    solver: str = "none"
    created_at: str = ""


def _fit_affine(x_vals, y_vals):
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return 1.0, 0.0
    if len(x) == 1:
        if abs(x[0]) < 1e-9:
            return 1.0, float(y[0])
        scale = float(y[0] / x[0])
        return scale, 0.0

    # y = a*x + b
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def fit_calibration_from_samples(samples_df: pd.DataFrame, solver: str = "mixed") -> CFDCalibrationModel:
    required = [
        'baseline_thrust_N', 'cfd_thrust_N',
        'baseline_tsfc_kg_N_s', 'cfd_tsfc_kg_N_s'
    ]
    missing = [c for c in required if c not in samples_df.columns]
    if missing:
        raise ValueError(f"Missing CFD sample columns: {missing}")
    if samples_df.empty:
        raise ValueError("No CFD samples available for calibration.")

    a_t, b_t = _fit_affine(samples_df['baseline_thrust_N'], samples_df['cfd_thrust_N'])
    a_s, b_s = _fit_affine(samples_df['baseline_tsfc_kg_N_s'], samples_df['cfd_tsfc_kg_N_s'])

    return CFDCalibrationModel(
        thrust_scale=a_t,
        thrust_bias=b_t,
        tsfc_scale=a_s,
        tsfc_bias=b_s,
        sample_count=int(len(samples_df)),
        solver=solver,
        created_at=datetime.utcnow().isoformat() + 'Z',
    )


def save_calibration_model(model: CFDCalibrationModel, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(model), f, indent=2)


def load_calibration_model(path: str) -> Optional[CFDCalibrationModel]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return CFDCalibrationModel(**payload)


def apply_calibration_to_metrics(metrics: Dict, model: Optional[CFDCalibrationModel]) -> Dict:
    if model is None:
        return dict(metrics)

    out = dict(metrics)
    thrust_raw = float(out.get('thrust_N', 0.0))
    tsfc_raw = float(out.get('tsfc_kg_N_s', 0.0))

    thrust_cal = model.thrust_scale * thrust_raw + model.thrust_bias
    tsfc_cal = model.tsfc_scale * tsfc_raw + model.tsfc_bias
    tsfc_cal = max(tsfc_cal, 1e-10)

    out['thrust_N'] = thrust_cal
    out['tsfc_kg_N_s'] = tsfc_cal

    if 'fuel_flow_kg_s' in out:
        out['fuel_flow_kg_s'] = tsfc_cal * max(thrust_cal, 0.0)
    if 'mass_flow_kg_s' in out and float(out['mass_flow_kg_s']) > 1e-9:
        out['specific_thrust'] = thrust_cal / float(out['mass_flow_kg_s'])
    if 'total_mass_kg' in out and float(out['total_mass_kg']) > 1e-9:
        out['thrust_to_weight'] = thrust_cal / (float(out['total_mass_kg']) * 9.81)

    return out


def apply_calibration_to_dataframe(df: pd.DataFrame, model: Optional[CFDCalibrationModel]) -> pd.DataFrame:
    if model is None:
        return df

    out = df.copy()
    if 'thrust_N' in out.columns:
        out['thrust_N'] = model.thrust_scale * out['thrust_N'] + model.thrust_bias
    if 'tsfc_kg_N_s' in out.columns:
        out['tsfc_kg_N_s'] = model.tsfc_scale * out['tsfc_kg_N_s'] + model.tsfc_bias
        out['tsfc_kg_N_s'] = out['tsfc_kg_N_s'].clip(lower=1e-10)
    if 'fuel_flow_kg_s' in out.columns and 'thrust_N' in out.columns and 'tsfc_kg_N_s' in out.columns:
        out['fuel_flow_kg_s'] = out['tsfc_kg_N_s'] * out['thrust_N'].clip(lower=0.0)
    if 'specific_thrust' in out.columns and 'thrust_N' in out.columns and 'mass_flow_kg_s' in out.columns:
        safe_mdot = out['mass_flow_kg_s'].replace(0, np.nan)
        out['specific_thrust'] = (out['thrust_N'] / safe_mdot).fillna(0.0)
    if 'thrust_to_weight' in out.columns and 'thrust_N' in out.columns and 'total_mass_kg' in out.columns:
        safe_mass = out['total_mass_kg'].replace(0, np.nan)
        out['thrust_to_weight'] = (out['thrust_N'] / (safe_mass * 9.81)).fillna(0.0)
    return out


def _solver_template_for(solver: str) -> str:
    solver = (solver or "").strip().lower()
    if solver == "openfoam":
        env_name = OPENFOAM_CMD_ENV
    elif solver == "su2":
        env_name = SU2_CMD_ENV
    else:
        raise ValueError("solver must be 'openfoam' or 'su2'")

    template = os.environ.get(env_name, "").strip()
    if not template:
        raise RuntimeError(
            f"{env_name} is not set. Configure a command template that prints JSON "
            f"with thrust_N and tsfc_kg_N_s on stdout."
        )
    return template


def run_solver_case(row: Dict, solver: str, timeout_s: int = 900) -> Dict[str, float]:
    template = _solver_template_for(solver)
    cmd = template.format(**row)

    proc = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=int(timeout_s),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"CFD command failed (solver={solver}, code={proc.returncode}): {proc.stderr.strip()}"
        )

    stdout_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    json_line = None
    for ln in reversed(stdout_lines):
        if ln.startswith("{") and ln.endswith("}"):
            json_line = ln
            break
    if json_line is None:
        raise RuntimeError("CFD command output must include a JSON line with thrust_N and tsfc_kg_N_s.")

    payload = json.loads(json_line)
    if 'thrust_N' not in payload or 'tsfc_kg_N_s' not in payload:
        raise RuntimeError("CFD JSON must contain thrust_N and tsfc_kg_N_s.")

    return {
        'thrust_N': float(payload['thrust_N']),
        'tsfc_kg_N_s': float(payload['tsfc_kg_N_s']),
    }

