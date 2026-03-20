"""Analytical reconstruction of derived quantities from VLM primitives.

The MLP surrogate predicts 7 smooth primitives (CL_0, CL_alpha, CM_0,
CM_alpha, CD0_wing, CD0_body, Cn_beta).  Everything else -- trim state,
stability, drag breakdown, L/D, propulsion balance, penalty -- is
reconstructed here with exact physics, no ML.
"""

import numpy as np
from ..parameterization.design_variables import BWBParams, params_from_vector
from ..aero.mission import MissionCondition, FeasibilityConfig, DEFAULT_FEASIBILITY
from ..aero.drag import oswald_efficiency
from ..parameterization.bwb_aircraft import (
    build_airplane,
    compute_wing_area,
    compute_mac,
    compute_aspect_ratio,
    estimate_structural_mass,
    compute_internal_volume,
)
from ..propulsion.edf_model import thrust_at_speed, duct_fits_in_body, intake_drag
from ..systems.cg import compute_cg, CGConfig, DEFAULT_CG_CONFIG
from ..evaluation.manufacturability import compute_manufacturability

# Design-vector indices (match design_variables.py)
_IDX_HALF_SPAN = 0
_IDX_WING_ROOT_CHORD = 1
_IDX_TAPER = 2
_IDX_SWEEP = 3
_IDX_BODY_CHORD_RATIO = 4
_IDX_BODY_HALFWIDTH = 5
_IDX_BODY_TC_ROOT = 6


def reconstruct_aero(
    primitives: dict,
    x: np.ndarray,
    mission: MissionCondition | None = None,
    feasibility: FeasibilityConfig | None = None,
    cg_config: CGConfig | None = None,
) -> dict:
    """Reconstruct all derived aerodynamic and propulsion quantities from
    7 VLM primitives.

    Parameters
    ----------
    primitives : dict
        Keys: CL_0, CL_alpha, CM_0, CM_alpha, CD0_wing, CD0_body, Cn_beta
        (scalars or 1-D arrays).
    x : np.ndarray
        Design vector(s), shape ``(30,)`` or ``(N, 30)``.
    mission : MissionCondition, optional
    feasibility : FeasibilityConfig, optional

    Returns
    -------
    dict with: alpha_eq, CL, CM, static_margin, oswald_e, AR, S_ref, MAC,
               CDi, CD, CD0, CD0_wing, CD0_body, Cn_beta, L_over_D,
               struct_mass, internal_volume, CL_required,
               T_available, T_over_D, drag_force, P_thrust, P_elec,
               endurance_s, endurance_min, duct_fits,
               penalty, is_feasible.
    """
    mission = mission or MissionCondition()
    feasibility = feasibility or DEFAULT_FEASIBILITY

    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)

    n = x.shape[0]

    # --- Geometry (exact, zero ML) ---
    hs = x[:, _IDX_HALF_SPAN]
    rc = x[:, _IDX_WING_ROOT_CHORD]
    tr = x[:, _IDX_TAPER]
    sweep = x[:, _IDX_SWEEP]
    bcr = x[:, _IDX_BODY_CHORD_RATIO]
    bw = x[:, _IDX_BODY_HALFWIDTH]
    btc = x[:, _IDX_BODY_TC_ROOT]

    tip_chord = rc * tr
    body_root_chord = rc * bcr
    outer_span = hs - bw

    # Wing area: body trapezoid + outer wing trapezoid
    s_ref = (body_root_chord + rc) * bw + (rc + tip_chord) * outer_span
    ar = (2 * hs) ** 2 / np.maximum(s_ref, 1e-6)
    mac = (2 / 3) * rc * (1 + tr + tr ** 2) / (1 + tr)

    e_oswald = np.array([
        oswald_efficiency(a, s, t) for a, s, t in zip(ar, sweep, tr)
    ])

    # Structural mass and internal volume via parameterization
    struct_mass = np.array([
        estimate_structural_mass(params_from_vector(x[i]), mtow=mission.mtow)
        for i in range(n)
    ])
    internal_volume = np.array([
        compute_internal_volume(params_from_vector(x[i]))
        for i in range(n)
    ])

    # CL_required for level flight
    q = mission.dynamic_pressure
    weight = mission.weight
    cl_required = weight / (q * s_ref)

    # --- Primitives (from surrogate) ---
    cl_0 = np.atleast_1d(np.asarray(primitives["CL_0"], dtype=float))
    cl_alpha = np.atleast_1d(np.asarray(primitives["CL_alpha"], dtype=float))
    cm_0 = np.atleast_1d(np.asarray(primitives["CM_0"], dtype=float))
    cm_alpha = np.atleast_1d(np.asarray(primitives["CM_alpha"], dtype=float))
    cd0_wing = np.atleast_1d(np.asarray(primitives["CD0_wing"], dtype=float))
    cd0_body = np.atleast_1d(np.asarray(primitives["CD0_body"], dtype=float))
    cn_beta = np.atleast_1d(np.asarray(primitives["Cn_beta"], dtype=float))

    # --- Aerodynamic reconstruction ---
    alpha_eq_rad = np.where(
        np.abs(cl_alpha) > 0.1,
        (cl_required - cl_0) / cl_alpha,
        np.radians(5.0),
    )
    alpha_eq = np.clip(np.degrees(alpha_eq_rad), -2.0, 12.0)
    alpha_eq_rad = np.radians(alpha_eq)

    cl = cl_0 + cl_alpha * alpha_eq_rad
    cm = cm_0 + cm_alpha * alpha_eq_rad
    sm_raw = np.where(np.abs(cl_alpha) > 0.01, -cm_alpha / cl_alpha, 0.0)

    # ── CG correction ──
    # The surrogate was trained with x_ref = 0.30 * body_root_chord.
    # Real CG is different. The correction uses c_ref (AeroSandbox reference chord),
    # NOT MAC, because AVL normalizes moments by c_ref.
    # c_ref ≈ mean chord of the full aircraft (body + wing), approximated as:
    #   c_ref = S_ref / (2 * half_span) = wing_area / span
    cg_config = cg_config or DEFAULT_CG_CONFIG
    x_cg_old = body_root_chord * 0.30  # training reference
    c_ref = s_ref / (2.0 * hs)  # AeroSandbox convention: area / span
    sm = sm_raw.copy()
    x_cg_real = np.zeros(n)
    manuf_score = np.zeros(n)
    for i in range(n):
        params_i = params_from_vector(x[i])
        cg_result = compute_cg(
            params_i,
            battery_mass=mission.battery.mass,
            edf_mass=mission.edf.mass,
            avionics_mass=mission.avionics_mass,
            config=cg_config,
        )
        x_cg_real[i] = cg_result["x_cg"]
        # Manufacturability (geometry-only, no AVL)
        manuf_score[i] = compute_manufacturability(params_i)["manufacturability_score"]
        # Correct SM for real CG: SM_real = SM_raw + (x_cg_old - x_cg_real) / c_ref
        if c_ref[i] > 0.01:
            sm[i] = sm_raw[i] + (x_cg_old[i] - x_cg_real[i]) / c_ref[i]
            # Correct CM at trim for real CG
            cm[i] = cm[i] + cl[i] * (x_cg_real[i] - x_cg_old[i]) / c_ref[i]

    # Elevon trim deflection from CM at trim
    # delta_e [deg] ≈ -CM_trim / Cm_de  where Cm_de is [1/deg]
    # Use predicted Cm_de if available (v2 surrogate), otherwise estimate
    cmd01 = np.atleast_1d(np.asarray(primitives.get("Cmd01", 0.0), dtype=float))
    cl_beta_pred = np.atleast_1d(np.asarray(primitives.get("Cl_beta", 0.0), dtype=float))
    # Cmd01 is per radian from AVL; convert to per degree
    cm_de_per_deg = np.where(np.abs(cmd01) > 1e-6, cmd01 / np.degrees(1.0), -0.01)
    elevon_defl = np.where(
        np.abs(cm) > 0.001,
        -cm / cm_de_per_deg,  # [deg]
        0.0,
    )
    elevon_defl = np.clip(elevon_defl, -25.0, 25.0)

    # Stall speed
    vs = np.sqrt(2 * weight / (mission.density * s_ref * feasibility.cl_max_clean))

    # Total CD0 = wing + body + intake
    cd_intake = np.array([
        intake_drag(mission.edf, s_ref[i]) for i in range(n)
    ])
    cd0 = cd0_wing + cd0_body + cd_intake

    cdi = np.where(ar > 1.0, cl ** 2 / (np.pi * ar * e_oswald), 0.1)

    # Trim drag estimate from elevon deflection
    cd_trim = np.where(
        np.abs(elevon_defl) > 0.1,
        0.0001 * np.abs(elevon_defl) / 10.0,  # profile drag increment
        0.0,
    )
    cd = cd0 + cdi + cd_trim
    ld = np.where(cd > 1e-4, cl / cd, 0.0)
    cd_effective = cd * feasibility.drag_margin

    # --- Propulsion reconstruction (uses cd_effective for conservative margin) ---
    drag_force = cd_effective * q * s_ref
    t_available = np.array([
        thrust_at_speed(mission.edf, mission.velocity) for _ in range(n)
    ])
    t_over_d = np.where(drag_force > 0.01, t_available / drag_force, float("inf"))

    p_thrust = drag_force * mission.velocity
    eta_total = mission.edf.eta_fan * mission.edf.eta_motor
    p_elec = np.where(eta_total > 0, p_thrust / eta_total, float("inf"))

    usable_frac = 0.80
    e_battery = mission.battery.energy_j
    endurance_s = np.where(p_elec > 0, e_battery * usable_frac / p_elec, float("inf"))
    endurance_min = endurance_s / 60.0

    duct_fits_arr = np.array([
        duct_fits_in_body(btc[i], body_root_chord[i], mission.edf)
        for i in range(n)
    ], dtype=bool)

    # --- Penalty & feasibility ---
    penalty = np.zeros(n)
    is_feasible = np.zeros(n, dtype=bool)
    for i in range(n):
        p_val, _ = feasibility.compute_penalty(
            static_margin=sm[i],
            cm=cm[i],
            struct_mass=struct_mass[i],
            mass_budget=mission.mass_budget,
            cl_required=cl_required[i],
            ar=ar[i],
            internal_volume=internal_volume[i],
            cn_beta=cn_beta[i],
            t_over_d=t_over_d[i],
            endurance_s=endurance_s[i],
            duct_fits=bool(duct_fits_arr[i]),
            vs=float(vs[i]),
            alpha_trim=float(alpha_eq[i]),
            elevon_deflection=float(elevon_defl[i]),
            cl_beta=float(cl_beta_pred[i]) if len(cl_beta_pred) > 1 else float(cl_beta_pred[0]),
            manufacturability_score=float(manuf_score[i]),
        )
        penalty[i] = p_val
        is_feasible[i] = feasibility.is_feasible(
            static_margin=sm[i],
            cm=cm[i],
            struct_mass=struct_mass[i],
            mass_budget=mission.mass_budget,
            cl_required=cl_required[i],
            ar=ar[i],
            internal_volume=internal_volume[i],
            cn_beta=cn_beta[i],
            t_over_d=t_over_d[i],
            endurance_s=endurance_s[i],
            duct_fits=bool(duct_fits_arr[i]),
            vs=float(vs[i]),
            alpha_trim=float(alpha_eq[i]),
            elevon_deflection=float(elevon_defl[i]),
            cl_beta=float(cl_beta_pred[i]) if len(cl_beta_pred) > 1 else float(cl_beta_pred[0]),
            manufacturability_score=float(manuf_score[i]),
        )

    result = {
        # Aerodynamics
        "L_over_D": ld,
        "CL": cl,
        "CM": cm,
        "static_margin": sm,
        "alpha_eq": alpha_eq,
        "CDi": cdi,
        "CD": cd,
        "CD0": cd0,
        "CD0_wing": cd0_wing,
        "CD0_body": cd0_body,
        "CD_intake": cd_intake,
        "CD_trim": cd_trim,
        "CD_effective": cd_effective,
        "Cn_beta": cn_beta,
        "oswald_e": e_oswald,
        "AR": ar,
        "S_ref": s_ref,
        "MAC": mac,
        "struct_mass": struct_mass,
        "internal_volume": internal_volume,
        "Vs": vs,
        "CL_required": cl_required,
        "manufacturability_score": manuf_score,
        "Cl_beta": cl_beta_pred,
        "Cmd01": cmd01,
        "elevon_deflection": elevon_defl,
        # Propulsion
        "T_available": t_available,
        "T_over_D": t_over_d,
        "drag_force": drag_force,
        "P_thrust": p_thrust,
        "P_elec": p_elec,
        "endurance_s": endurance_s,
        "endurance_min": endurance_min,
        "duct_fits": duct_fits_arr,
        # Constraints
        "penalty": penalty,
        "is_feasible": is_feasible,
    }

    if single:
        result = {
            k: (float(v[0]) if hasattr(v, '__len__') and len(v) > 0 else float(v))
            for k, v in result.items()
        }
        # Convert duct_fits back to bool for scalar case
        result["duct_fits"] = bool(result["duct_fits"])
        result["is_feasible"] = bool(result["is_feasible"])

    return result
