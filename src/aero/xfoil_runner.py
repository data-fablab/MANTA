"""XFOIL runner for 2D airfoil polar computation at low Reynolds.

Calls XFOIL 6.99 via subprocess (same pattern as avl_runner.py).
Used for validation of NeuralFoil predictions at Re = 50k-500k.
"""

import subprocess
import tempfile
import re
from pathlib import Path
from typing import Optional

import numpy as np
import aerosandbox as asb

_DEFAULT_XFOIL_CMD = "d:/UAV/tools/xfoil/xfoil.exe"


def run_xfoil_polar(
    airfoil: asb.Airfoil | asb.KulfanAirfoil,
    alpha_range: tuple[float, float, float] = (-2, 12, 0.5),
    Re: float = 200_000,
    mach: float = 0.0,
    n_crit: float = 9.0,
    max_iter: int = 100,
    xfoil_command: str = _DEFAULT_XFOIL_CMD,
    timeout: float = 15.0,
) -> Optional[dict]:
    """Run XFOIL to compute a 2D airfoil polar.

    Parameters
    ----------
    airfoil : asb.Airfoil or asb.KulfanAirfoil
    alpha_range : (alpha_min, alpha_max, alpha_step) in degrees
    Re : Reynolds number
    mach : Mach number
    n_crit : Turbulence parameter (9 = clean wind tunnel)
    max_iter : XFOIL viscous iteration limit
    xfoil_command : path to xfoil.exe
    timeout : max seconds

    Returns
    -------
    dict with 'alpha', 'CL', 'CD', 'CDp', 'CM', 'Cpmin', 'Top_Xtr', 'Bot_Xtr'
    arrays, or None if XFOIL fails.
    """
    # Convert KulfanAirfoil to Airfoil if needed
    if isinstance(airfoil, asb.KulfanAirfoil):
        airfoil = airfoil.to_airfoil()

    # Repanel for XFOIL (needs clean coordinates)
    af = airfoil.repanel(160)
    coords = af.coordinates.copy()
    coords[:, 0] = np.clip(coords[:, 0], 0.0, 1.0)

    a_min, a_max, a_step = alpha_range

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        af_file = tmpdir / "airfoil.dat"
        polar_file = tmpdir / "polar.txt"

        # Write airfoil coordinates
        with open(af_file, "w") as f:
            f.write(f"{airfoil.name}\n")
            for x, y in coords:
                f.write(f"{x:.8f}  {y:.8f}\n")

        # Build XFOIL keystrokes
        keystrokes = (
            f"LOAD {af_file.name}\n"
            f"\n"  # accept name
            f"PANE\n"  # repanel internally
            f"OPER\n"
            f"VISC {Re:.0f}\n"
            f"MACH {mach:.3f}\n"
            f"VPAR\n"
            f"N {n_crit:.1f}\n"
            f"\n"  # back to OPER
            f"ITER {max_iter}\n"
            f"PACC\n"
            f"{polar_file.name}\n"  # output file
            f"\n"  # no dump file
            f"ASEQ {a_min:.1f} {a_max:.1f} {a_step:.2f}\n"
            f"PACC\n"  # close polar accumulation
            f"\n"
            f"QUIT\n"
        )

        try:
            proc = subprocess.run(
                [xfoil_command],
                cwd=str(tmpdir),
                input=keystrokes,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None

        if not polar_file.exists():
            return None

        return _parse_xfoil_polar(polar_file.read_text())


def _parse_xfoil_polar(text: str) -> Optional[dict]:
    """Parse XFOIL polar output file."""
    lines = text.strip().split("\n")

    # Skip header lines (start with '---' separator)
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("------"):
            data_start = i + 1
            break

    if data_start is None:
        return None

    alphas, cls, cds, cdps, cms, cpmins, xtrs_top, xtrs_bot = (
        [], [], [], [], [], [], [], []
    )

    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            alphas.append(float(parts[0]))
            cls.append(float(parts[1]))
            cds.append(float(parts[2]))
            cdps.append(float(parts[3]))
            cms.append(float(parts[4]))
            xtrs_top.append(float(parts[5]))
            xtrs_bot.append(float(parts[6]))
        except ValueError:
            continue

    if len(alphas) == 0:
        return None

    return {
        "alpha": np.array(alphas),
        "CL": np.array(cls),
        "CD": np.array(cds),
        "CDp": np.array(cdps),
        "CM": np.array(cms),
        "Top_Xtr": np.array(xtrs_top),
        "Bot_Xtr": np.array(xtrs_bot) if xtrs_bot else None,
    }


def compare_neuralfoil_vs_xfoil(
    airfoil: asb.KulfanAirfoil,
    Re_values: list[float] = [100_000, 200_000, 300_000, 500_000],
    alpha_range: tuple[float, float, float] = (-2, 10, 1.0),
    xfoil_command: str = _DEFAULT_XFOIL_CMD,
) -> dict:
    """Compare NeuralFoil vs XFOIL predictions across multiple Re.

    Returns dict with per-Re comparison data and aggregate error metrics.
    """
    comparisons = {}

    for Re in Re_values:
        # XFOIL polar
        xfoil_data = run_xfoil_polar(
            airfoil, alpha_range=alpha_range, Re=Re,
            xfoil_command=xfoil_command,
        )

        if xfoil_data is None or len(xfoil_data["alpha"]) < 3:
            comparisons[Re] = {"status": "xfoil_failed"}
            continue

        # NeuralFoil at same alpha points
        nf_cls, nf_cds = [], []
        for alpha in xfoil_data["alpha"]:
            try:
                nf = airfoil.get_aero_from_neuralfoil(
                    alpha=float(alpha), Re=float(Re), mach=0.0,
                    model_size="large",
                )
                cl_val = nf["CL"]
                cd_val = nf["CD"]
                nf_cls.append(float(cl_val[0]) if hasattr(cl_val, '__len__') else float(cl_val))
                nf_cds.append(float(cd_val[0]) if hasattr(cd_val, '__len__') else float(cd_val))
            except Exception:
                nf_cls.append(np.nan)
                nf_cds.append(np.nan)

        nf_cl = np.array(nf_cls)
        nf_cd = np.array(nf_cds)

        # Compute errors (only where both have valid data)
        valid = ~np.isnan(nf_cl) & ~np.isnan(nf_cd)
        if valid.sum() < 3:
            comparisons[Re] = {"status": "insufficient_data"}
            continue

        cl_err = nf_cl[valid] - xfoil_data["CL"][valid]
        cd_err = nf_cd[valid] - xfoil_data["CD"][valid]

        comparisons[Re] = {
            "status": "ok",
            "alpha": xfoil_data["alpha"],
            "xfoil_CL": xfoil_data["CL"],
            "xfoil_CD": xfoil_data["CD"],
            "xfoil_CM": xfoil_data["CM"],
            "xfoil_Xtr_top": xfoil_data["Top_Xtr"],
            "neuralfoil_CL": nf_cl,
            "neuralfoil_CD": nf_cd,
            "CL_bias": float(np.mean(cl_err)),
            "CL_rmse": float(np.sqrt(np.mean(cl_err**2))),
            "CD_bias": float(np.mean(cd_err)),
            "CD_rmse": float(np.sqrt(np.mean(cd_err**2))),
            "CD_ratio": float(np.mean(nf_cd[valid] / xfoil_data["CD"][valid])),
        }

    # Aggregate CD correction factor
    cd_ratios = [c["CD_ratio"] for c in comparisons.values()
                 if isinstance(c, dict) and c.get("status") == "ok"]
    avg_cd_ratio = float(np.mean(cd_ratios)) if cd_ratios else 1.0

    return {
        "per_Re": comparisons,
        "cd_correction_factor": avg_cd_ratio,
        "n_valid": len(cd_ratios),
    }
