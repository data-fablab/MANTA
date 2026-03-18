"""Lightweight AVL runner for direct stability derivatives.

Calls AVL once per geometry and extracts all stability derivatives
(CLa, Cma, Cnb, Clb, etc.) analytically via AVL's `st` command —
no finite-difference sideslip runs needed.
"""

import subprocess
import tempfile
import re
from pathlib import Path
from typing import Optional

import numpy as np
import aerosandbox as asb


# Default AVL executable path
_DEFAULT_AVL_CMD = "d:/UAV/tools/avl/avl.exe"


def run_avl_stability(
    airplane: asb.Airplane,
    alpha: float,
    velocity: float = 20.0,
    density: float = 1.225,
    avl_command: str = _DEFAULT_AVL_CMD,
    timeout: float = 10.0,
    chordwise_res: int = 6,
    spanwise_res: int = 8,
) -> Optional[dict]:
    """Run AVL at given alpha and return stability derivatives.

    Parameters
    ----------
    airplane : asb.Airplane
        AeroSandbox airplane (used for geometry export).
    alpha : float
        Angle of attack [deg].
    velocity, density : float
        Freestream conditions.
    avl_command : str
        Path to AVL executable.
    timeout : float
        Max seconds for AVL process.
    chordwise_res, spanwise_res : int
        Panel resolution per surface.

    Returns
    -------
    dict or None
        Dictionary with forces, moments, and all stability derivatives.
        None if AVL fails.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        avl_file = tmpdir / "airplane.avl"

        # Write .avl file with clamped airfoils
        _write_avl_file(
            airplane, avl_file,
            chordwise_res=chordwise_res,
            spanwise_res=spanwise_res,
        )

        # Keystrokes for AVL
        output_file = "output.txt"
        keystrokes = (
            f"plop\ng\n\n"       # disable graphics
            f"oper\n"             # enter OPER mode
            f"m\n"               # modify parameters
            f"mn 0.0\n"          # Mach
            f"v {velocity}\n"    # velocity
            f"d {density}\n"     # density
            f"g 9.81\n"          # gravity
            f"\n"                # back to OPER
            f"a a {alpha}\n"     # set alpha
            f"b b 0.0\n"         # beta = 0
            f"x\n"               # execute
            f"st\n"              # stability derivatives output
            f"{output_file}\n"   # filename
            f"\n"                # confirm
            f"quit\n"            # exit
        )

        try:
            proc = subprocess.run(
                [avl_command, str(avl_file.name)],
                cwd=str(tmpdir),
                input=keystrokes,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None

        out_path = tmpdir / output_file
        if not out_path.exists():
            return None

        return _parse_avl_st_output(out_path.read_text())


def _write_avl_file(
    airplane: asb.Airplane,
    filepath: Path,
    chordwise_res: int = 6,
    spanwise_res: int = 8,
) -> None:
    """Write .avl geometry file with clamped airfoil coordinates."""
    filepath = Path(filepath)

    lines = []
    lines.append(airplane.name)
    lines.append("#Mach")
    lines.append("0.0")
    lines.append("#IYsym  IZsym  Zsym")
    lines.append("0  0  0.0")
    lines.append("#Sref   Cref   Bref")
    lines.append(f"{airplane.s_ref}  {airplane.c_ref}  {airplane.b_ref}")
    lines.append("#Xref   Yref   Zref")
    lines.append(f"{airplane.xyz_ref[0]}  {airplane.xyz_ref[1]}  {airplane.xyz_ref[2]}")
    lines.append("# CDp")
    lines.append("0")

    af_counter = 0

    for wing in airplane.wings:
        lines.append("#" + "=" * 60)
        lines.append("SURFACE")
        lines.append(wing.name)
        lines.append(f"#Nchord  Cspace  Nspan  Sspace")
        lines.append(f"{chordwise_res}  1  {spanwise_res}  1")
        lines.append("")

        if wing.symmetric:
            lines.append("YDUPLICATE")
            lines.append("0")
            lines.append("")

        for xsec in wing.xsecs:
            lines.append("#" + "-" * 40)
            lines.append("SECTION")
            lines.append("#Xle  Yle  Zle  Chord  Ainc")
            xyz = xsec.xyz_le
            lines.append(
                f"{xyz[0]:.8g}  {xyz[1]:.8g}  {xyz[2]:.8g}  "
                f"{xsec.chord:.8g}  {xsec.twist:.8g}"
            )
            lines.append("")

            # Write airfoil file with clamped coordinates
            af_path = filepath.parent / f"af_{af_counter}.dat"
            af_counter += 1
            _write_clamped_airfoil(xsec.airfoil, af_path)

            lines.append("AFIL")
            lines.append(af_path.name)
            lines.append("")

            # CLAF correction for thick airfoils
            tc = xsec.airfoil.max_thickness()
            claf = 1.0 + 0.77 * tc
            lines.append("CLAF")
            lines.append(f"{claf:.6f}")
            lines.append("")

    filepath.write_text("\n".join(lines))


def _write_clamped_airfoil(airfoil: asb.Airfoil, filepath: Path) -> None:
    """Write airfoil .dat file with coordinates clamped to [0, 1] in x."""
    # Repanel to consistent number of points
    af = airfoil.repanel(50)
    coords = af.coordinates.copy()

    # Clamp x to [0, 1] — fixes thick body airfoils where spline
    # interpolation can produce x slightly outside bounds
    coords[:, 0] = np.clip(coords[:, 0], 0.0, 1.0)

    with open(filepath, "w") as f:
        f.write(f"{airfoil.name}\n")
        for x, y in coords:
            f.write(f"{x:.8f}  {y:.8f}\n")


def _parse_avl_st_output(text: str) -> dict:
    """Parse AVL stability derivative output file.

    Extracts:
    - Total forces: CL, CD, Cm, CY, Cl, Cn
    - Alpha derivatives: CLa, CDa, Cma, CYa, Cla, Cna
    - Beta derivatives: CLb, CDb, Cmb, CYb, Clb, Cnb
    - Rate derivatives: CLp, CYp, Clp, Cnp, CLq, Cmq, CLr, CYr, Clr, Cnr
    - Neutral point: Xnp
    - Trefftz: CLff, CDff, e
    """
    result = {}

    # Parse "key = value" pairs — keep first occurrence (don't overwrite)
    for match in re.finditer(r'(\w[\w\'/]*)\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)', text):
        key = match.group(1)
        try:
            value = float(match.group(2))
        except ValueError:
            continue

        # Normalize keys: CXtot -> CX, CLtot -> CL, etc.
        clean_key = key.replace("tot", "").replace("'", "p")

        # Keep first occurrence — the ratio line "Clb Cnr / Clr Cnb = ..."
        # at the end would otherwise overwrite the real Cnb derivative
        if clean_key not in result:
            result[clean_key] = value

    return result
