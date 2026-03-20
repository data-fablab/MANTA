"""Lightweight AVL runner for direct stability derivatives.

Calls AVL once per geometry and extracts all stability derivatives
(CLa, Cma, Cnb, Clb, etc.) analytically via AVL's `st` command —
no finite-difference sideslip runs needed.

v2: Added control surface support (elevon + aileron) with trim capability.
"""

import subprocess
import tempfile
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import aerosandbox as asb


# Default AVL executable path
_DEFAULT_AVL_CMD = "d:/UAV/tools/avl/avl.exe"


# ═══════════════════════════════════════════════════════════════════════════
# Control surface definition
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ControlSurface:
    """A control surface defined by span range and chord fraction.

    Parameters
    ----------
    name : str
        AVL control variable name (max 8 chars).
    cf_ratio : float
        Flap chord / local chord (e.g. 0.25 = 25% chord flap).
    eta_inner : float
        Inner span boundary (fraction of outer wing, 0 = body blend).
    eta_outer : float
        Outer span boundary (fraction of outer wing, 1 = tip).
    sgn_dup : float
        Sign of duplicate surface deflection:
        +1.0 = symmetric (elevator/pitch trim — both sides same),
        -1.0 = antisymmetric (aileron/roll — sides opposite).
    gain : float
        Deflection gain multiplier.
    """
    name: str
    cf_ratio: float
    eta_inner: float
    eta_outer: float
    sgn_dup: float
    gain: float = 1.0

    @property
    def xhinge(self) -> float:
        """Hinge x/c position (from LE). AVL convention."""
        return 1.0 - self.cf_ratio


@dataclass
class ControlConfig:
    """Configuration for aircraft control surfaces.

    Default: nEUROn-style elevon + aileron layout for BWB.
    """
    surfaces: list[ControlSurface] = field(default_factory=list)
    trim_elevon: bool = True   # use AVL trim for pitch with first surface

    @classmethod
    def default_bwb(cls) -> "ControlConfig":
        """Standard BWB layout: inboard elevon + outboard aileron."""
        return cls(surfaces=[
            ControlSurface(
                name="elevon",
                cf_ratio=0.25,       # 25% chord
                eta_inner=0.0,       # from body blend
                eta_outer=0.50,      # to 50% outer span
                sgn_dup=1.0,         # symmetric (both sides same → pitch)
            ),
            ControlSurface(
                name="aileron",
                cf_ratio=0.25,       # 25% chord
                eta_inner=0.55,      # from 55% outer span
                eta_outer=0.90,      # to 90% outer span
                sgn_dup=-1.0,        # antisymmetric (sides opposite → roll)
            ),
        ])


# Outer wing station fractions (must match bwb_aircraft.py)
_OUTER_WING_STATIONS = [0.0, 0.25, 0.50, 0.75, 0.90, 1.0]


# ═══════════════════════════════════════════════════════════════════════════
# AVL runner
# ═══════════════════════════════════════════════════════════════════════════

def run_avl_stability(
    airplane: asb.Airplane,
    alpha: float,
    velocity: float = 20.0,
    density: float = 1.225,
    avl_command: str = _DEFAULT_AVL_CMD,
    timeout: float = 10.0,
    chordwise_res: int = 6,
    spanwise_res: int = 8,
    controls: ControlConfig | None = None,
) -> Optional[dict]:
    """Run AVL at given alpha and return stability + control derivatives.

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
    controls : ControlConfig, optional
        Control surface configuration. If None, no control surfaces.

    Returns
    -------
    dict or None
        Dictionary with forces, moments, stability derivatives,
        and control derivatives (if controls provided).
        None if AVL fails.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        avl_file = tmpdir / "airplane.avl"

        # Write .avl file with clamped airfoils and control surfaces
        _write_avl_file(
            airplane, avl_file,
            chordwise_res=chordwise_res,
            spanwise_res=spanwise_res,
            controls=controls,
        )

        # Build keystrokes
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
        )

        # Add trim constraint: elevon trims pitch moment to 0
        if controls and controls.trim_elevon and controls.surfaces:
            keystrokes += f"d1 pm 0\n"   # control 1 → trim Cm=0

        keystrokes += (
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

        return _parse_avl_st_output(out_path.read_text(), controls)


def _write_avl_file(
    airplane: asb.Airplane,
    filepath: Path,
    chordwise_res: int = 6,
    spanwise_res: int = 8,
    controls: ControlConfig | None = None,
) -> None:
    """Write .avl geometry file with clamped airfoil coordinates and control surfaces."""
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

        is_outer_wing = (wing.name == "Outer Wing")

        for i_sec, xsec in enumerate(wing.xsecs):
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

            # Add CONTROL definitions for outer wing sections
            if is_outer_wing and controls:
                eta = _OUTER_WING_STATIONS[i_sec] if i_sec < len(_OUTER_WING_STATIONS) else 1.0
                for surf in controls.surfaces:
                    if surf.eta_inner <= eta <= surf.eta_outer:
                        lines.append("CONTROL")
                        lines.append(
                            f"{surf.name}  {surf.gain:.1f}  {surf.xhinge:.4f}  "
                            f"0.0  0.0  0.0  {surf.sgn_dup:.1f}"
                        )
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


def _parse_avl_st_output(text: str, controls: ControlConfig | None = None) -> dict:
    """Parse AVL stability derivative output file.

    Extracts:
    - Total forces: CL, CD, Cm, CY, Cl, Cn
    - Alpha derivatives: CLa, CDa, Cma, CYa, Cla, Cna
    - Beta derivatives: CLb, CDb, Cmb, CYb, Clb, Cnb
    - Rate derivatives: CLp, CYp, Clp, Cnp, CLq, Cmq, CLr, CYr, Clr, Cnr
    - Control derivatives: CLdN, CmdN, CYdN, CldN, CndN (N = control index)
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

    # Extract control surface deflections from AVL output
    # AVL prints lines like: "elevon          =   5.12345"
    if controls:
        for i, surf in enumerate(controls.surfaces):
            # Look for the deflection value by control name
            pattern = rf'{re.escape(surf.name)}\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result[f"delta_{surf.name}"] = float(match.group(1))

            # Map generic dN keys to named keys for clarity
            d_key = f"d{i+1}"
            for prefix in ["CL", "CD", "CY", "Cl", "Cn", "Cm"]:
                generic = f"{prefix}{d_key}"
                named = f"{prefix}_d{surf.name}"
                if generic in result:
                    result[named] = result[generic]

    return result
