"""Linearized 6-DOF dynamic stability analysis.

Computes longitudinal and lateral-directional eigenvalues from AVL
stability and rate derivatives. Identifies classical flight modes:
- Longitudinal: phugoid, short period
- Lateral: Dutch roll, spiral, roll subsidence

References:
- Nelson, R.C. "Flight Stability and Automatic Control" (2nd ed.)
- Etkin & Reid "Dynamics of Flight" (3rd ed.)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class FlightMode:
    """A single dynamic mode (eigenvalue pair or real root)."""
    name: str
    eigenvalues: np.ndarray      # 1 or 2 eigenvalues
    damping_ratio: float         # zeta (0 = undamped, 1 = critically damped)
    natural_freq_hz: float       # omega_n / (2*pi)
    period_s: float              # 2*pi / omega_d (if oscillatory)
    time_to_half_s: float        # ln(2) / |sigma| (if stable)
    is_oscillatory: bool
    is_stable: bool


def compute_longitudinal_modes(
    CL_0: float, CD_0: float,
    CL_a: float, CD_a: float, CM_a: float,
    CL_q: float, CM_q: float,
    velocity: float, rho: float, S_ref: float, c_ref: float,
    mass: float, Iy: float | None = None,
) -> list[FlightMode]:
    """Compute longitudinal modes (phugoid + short period).

    Parameters
    ----------
    CL_0, CD_0 : float — Trim lift/drag coefficients
    CL_a, CD_a, CM_a : float — Alpha derivatives [1/rad]
    CL_q, CM_q : float — Pitch rate derivatives [1/rad]
    velocity : float — Trim airspeed [m/s]
    rho : float — Air density [kg/m3]
    S_ref : float — Reference area [m2]
    c_ref : float — Reference chord [m]
    mass : float — Aircraft mass [kg]
    Iy : float, optional — Pitch moment of inertia [kg.m2]. Estimated if None.
    """
    g = 9.81
    q_bar = 0.5 * rho * velocity**2
    u0 = velocity

    # Dimensionless mass parameter
    mu = mass / (rho * S_ref * c_ref / 2)

    # Estimate Iy if not provided (thin plate approximation)
    if Iy is None:
        # For a BWB: Iy ~ m * (c_ref/2)^2 * k_yy^2, with k_yy ~ 0.3
        Iy = mass * (c_ref * 0.3) ** 2

    # Dimensional stability derivatives (body-axis)
    X_u = -q_bar * S_ref * (2 * CD_0) / (mass * u0)
    Z_u = -q_bar * S_ref * (2 * CL_0) / (mass * u0)
    Z_a = -q_bar * S_ref * CL_a / mass
    Z_q = -q_bar * S_ref * c_ref * CL_q / (2 * mass * u0)
    M_a = q_bar * S_ref * c_ref * CM_a / Iy
    M_q = q_bar * S_ref * c_ref**2 * CM_q / (2 * Iy * u0)

    # 4x4 state matrix [u, alpha, q, theta]
    # Simplified (neglect X_a, M_u terms for clarity)
    A = np.array([
        [X_u,       0,       0, -g],
        [Z_u / u0,  Z_a / u0, 1 + Z_q / u0, 0],
        [0,         M_a,     M_q, 0],
        [0,         0,       1,   0],
    ])

    eigenvalues = np.linalg.eigvals(A)
    return _classify_longitudinal(eigenvalues)


def compute_lateral_modes(
    CY_b: float, Cl_b: float, Cn_b: float,
    Cl_p: float, Cn_p: float,
    Cl_r: float, Cn_r: float,
    CY_r: float,
    velocity: float, rho: float, S_ref: float, b_ref: float,
    mass: float,
    Ix: float | None = None, Iz: float | None = None,
) -> list[FlightMode]:
    """Compute lateral-directional modes (Dutch roll, spiral, roll subsidence).

    Parameters
    ----------
    CY_b, Cl_b, Cn_b : float — Sideslip derivatives [1/rad]
    Cl_p, Cn_p : float — Roll rate derivatives [1/rad]
    Cl_r, Cn_r : float — Yaw rate derivatives [1/rad]
    CY_r : float — Side force due to yaw rate [1/rad]
    velocity, rho, S_ref, b_ref : float — Flight condition & geometry
    mass : float — Aircraft mass [kg]
    Ix, Iz : float, optional — Roll/yaw inertias [kg.m2]. Estimated if None.
    """
    g = 9.81
    q_bar = 0.5 * rho * velocity**2
    u0 = velocity

    # Estimate inertias if not provided
    if Ix is None:
        # BWB: Ix ~ m * (b/4)^2
        Ix = mass * (b_ref / 4) ** 2
    if Iz is None:
        # BWB: Iz ~ m * (b/3)^2
        Iz = mass * (b_ref / 3) ** 2

    # Ignore cross-product Ixz for simplicity
    Y_b = q_bar * S_ref * CY_b / mass
    L_b = q_bar * S_ref * b_ref * Cl_b / Ix
    N_b = q_bar * S_ref * b_ref * Cn_b / Iz

    L_p = q_bar * S_ref * b_ref**2 * Cl_p / (2 * Ix * u0)
    N_p = q_bar * S_ref * b_ref**2 * Cn_p / (2 * Iz * u0)
    L_r = q_bar * S_ref * b_ref**2 * Cl_r / (2 * Ix * u0)
    N_r = q_bar * S_ref * b_ref**2 * Cn_r / (2 * Iz * u0)
    Y_r = q_bar * S_ref * b_ref * CY_r / (2 * mass * u0)

    # 4x4 state matrix [beta, p, r, phi]
    A = np.array([
        [Y_b / u0,   0,       -(1 - Y_r / u0), g / u0],
        [L_b,        L_p,     L_r,              0],
        [N_b,        N_p,     N_r,              0],
        [0,          1,       0,                0],
    ])

    eigenvalues = np.linalg.eigvals(A)
    return _classify_lateral(eigenvalues)


def _classify_longitudinal(eigenvalues: np.ndarray) -> list[FlightMode]:
    """Classify 4 eigenvalues into phugoid + short period modes."""
    modes = _group_eigenvalues(eigenvalues)

    result = []
    for eigs in modes:
        mode = _eigenvalue_to_mode(eigs)
        # Short period has higher frequency
        result.append(mode)

    # Sort by frequency (lowest = phugoid, highest = short period)
    result.sort(key=lambda m: m.natural_freq_hz)
    if len(result) >= 2:
        result[0].name = "phugoid"
        result[1].name = "short_period"
    elif len(result) == 1:
        result[0].name = "short_period"

    return result


def _classify_lateral(eigenvalues: np.ndarray) -> list[FlightMode]:
    """Classify 4 eigenvalues into Dutch roll, spiral, roll subsidence."""
    modes = _group_eigenvalues(eigenvalues)

    result = []
    for eigs in modes:
        mode = _eigenvalue_to_mode(eigs)
        result.append(mode)

    # Classify: oscillatory = Dutch roll, real roots = spiral (slow) + roll (fast)
    oscillatory = [m for m in result if m.is_oscillatory]
    real_modes = [m for m in result if not m.is_oscillatory]

    for m in oscillatory:
        m.name = "dutch_roll"

    real_modes.sort(key=lambda m: abs(m.eigenvalues[0].real))
    if len(real_modes) >= 2:
        real_modes[0].name = "spiral"
        real_modes[1].name = "roll_subsidence"
    elif len(real_modes) == 1:
        real_modes[0].name = "roll_subsidence"

    return oscillatory + real_modes


def _group_eigenvalues(eigenvalues: np.ndarray) -> list[np.ndarray]:
    """Group complex conjugate pairs and real eigenvalues."""
    used = set()
    groups = []

    for i, ev in enumerate(eigenvalues):
        if i in used:
            continue
        if abs(ev.imag) > 1e-8:
            # Find conjugate
            for j in range(i + 1, len(eigenvalues)):
                if j not in used and abs(eigenvalues[j] - ev.conjugate()) < 1e-6:
                    groups.append(np.array([ev, eigenvalues[j]]))
                    used.add(i)
                    used.add(j)
                    break
            else:
                groups.append(np.array([ev]))
                used.add(i)
        else:
            groups.append(np.array([ev]))
            used.add(i)

    return groups


def _eigenvalue_to_mode(eigs: np.ndarray) -> FlightMode:
    """Convert eigenvalue(s) to FlightMode description."""
    ev = eigs[0]
    sigma = ev.real
    omega = abs(ev.imag)
    is_oscillatory = omega > 1e-6

    if is_oscillatory:
        omega_n = np.sqrt(sigma**2 + omega**2)
        zeta = -sigma / omega_n if omega_n > 1e-10 else 0.0
        freq_hz = omega_n / (2 * np.pi)
        period = 2 * np.pi / omega if omega > 1e-10 else float("inf")
    else:
        omega_n = abs(sigma)
        zeta = 1.0 if sigma < 0 else -1.0
        freq_hz = omega_n / (2 * np.pi)
        period = float("inf")

    is_stable = sigma < 0
    t_half = np.log(2) / abs(sigma) if abs(sigma) > 1e-10 else float("inf")

    return FlightMode(
        name="unknown",
        eigenvalues=eigs,
        damping_ratio=zeta,
        natural_freq_hz=freq_hz,
        period_s=period,
        time_to_half_s=t_half,
        is_oscillatory=is_oscillatory,
        is_stable=is_stable,
    )


def analyze_stability(avl_result: dict, params, mission) -> dict:
    """High-level analysis from AVL result + aircraft params.

    Parameters
    ----------
    avl_result : dict — Output from AeroEvaluator.evaluate()
    params : BWBParams
    mission : MissionCondition

    Returns
    -------
    dict with 'longitudinal' and 'lateral' lists of FlightMode,
    plus 'summary' with key stability metrics.
    """
    from ..parameterization.bwb_aircraft import compute_wing_area, compute_mac

    S = compute_wing_area(params)
    c = S / (2 * params.half_span)  # mean geometric chord (matches explicit c_ref)
    b = 2 * params.half_span

    lon_modes = compute_longitudinal_modes(
        CL_0=avl_result.get("CL", 0.3),
        CD_0=avl_result.get("CD", 0.02),
        CL_a=avl_result.get("CL_alpha", 5.0),
        CD_a=0.0,  # neglect
        CM_a=avl_result.get("CM_alpha", -0.5),
        CL_q=avl_result.get("CLq", 8.0),
        CM_q=avl_result.get("Cmq", -10.0),
        velocity=mission.velocity,
        rho=mission.density,
        S_ref=S, c_ref=c,
        mass=mission.mtow,
    )

    lat_modes = compute_lateral_modes(
        CY_b=avl_result.get("CYb", -0.3),
        Cl_b=avl_result.get("Cl_beta", -0.05),
        Cn_b=avl_result.get("Cn_beta", 0.01),
        Cl_p=avl_result.get("Clp", -0.4),
        Cn_p=avl_result.get("Cnp", -0.02),
        Cl_r=avl_result.get("Clr", 0.05),
        Cn_r=avl_result.get("Cnr", -0.05),
        CY_r=avl_result.get("CYr", 0.2),
        velocity=mission.velocity,
        rho=mission.density,
        S_ref=S, b_ref=b,
        mass=mission.mtow,
    )

    # Summary
    summary = {}
    for mode in lon_modes + lat_modes:
        summary[mode.name] = {
            "damping": mode.damping_ratio,
            "freq_hz": mode.natural_freq_hz,
            "period_s": mode.period_s,
            "stable": mode.is_stable,
        }

    all_stable = all(m.is_stable for m in lon_modes + lat_modes)
    dutch_roll = [m for m in lat_modes if m.name == "dutch_roll"]
    dutch_roll_ok = all(m.damping_ratio > 0.05 for m in dutch_roll) if dutch_roll else True

    summary["all_stable"] = all_stable
    summary["dutch_roll_ok"] = dutch_roll_ok

    return {
        "longitudinal": lon_modes,
        "lateral": lat_modes,
        "summary": summary,
    }
