"""Analytical reconstruction of derived quantities from VLM primitives.

The MLP surrogate predicts smooth primitives (CL_0, CL_alpha, CM_0,
CM_alpha, CD0_wing, CD0_body, Cn_beta, Cl_beta, CL_de, Cm_de, CL).
Everything else -- trim state, stability, drag breakdown, L/D,
propulsion balance, penalty -- is reconstructed here with exact
physics, no ML.

Performance: all geometry is computed with vectorized numpy,
no per-design Python loops, no Kulfan airfoil construction.
"""

import numpy as np
from ..parameterization.design_variables import BWBParams, params_from_vector
from ..aero.mission import MissionCondition, FeasibilityConfig, DEFAULT_FEASIBILITY
from ..aero.drag import oswald_efficiency
from ..systems.cg import CGConfig, DEFAULT_CG_CONFIG
from ..parameterization.bwb_aircraft import vectorized_body_tc_at_xc

# Design-vector indices (match design_variables.py — 32 variables)
_IDX_HALF_SPAN = 0
_IDX_WING_ROOT_CHORD = 1
_IDX_TAPER = 2
_IDX_SWEEP = 3
_IDX_BODY_CHORD_RATIO = 4
_IDX_BODY_HALFWIDTH = 5
_IDX_BODY_KU_U2 = 6
_IDX_BODY_KU_U6 = 7
_IDX_BODY_KU_U7 = 8
_IDX_BODY_KU_L2 = 9
_IDX_BODY_KU_L6 = 10
_IDX_BODY_KU_L7 = 11
_IDX_BODY_TWIST = 12
_IDX_BODY_SWEEP_DELTA = 13
_IDX_TWIST_1 = 14
_IDX_TWIST_TIP = 18
_IDX_DIHEDRAL_0 = 19
_IDX_DIHEDRAL_TIP = 23


def _body_tc_from_x(x: np.ndarray, x_frac: float = 0.30) -> np.ndarray:
    """Extract body t/c at arbitrary x/c from design vector, vectorized."""
    return vectorized_body_tc_at_xc(
        x[:, _IDX_BODY_KU_U2], x[:, _IDX_BODY_KU_U6], x[:, _IDX_BODY_KU_U7],
        x[:, _IDX_BODY_KU_L2], x[:, _IDX_BODY_KU_L6], x[:, _IDX_BODY_KU_L7],
        x_frac,
    )


# ── Fast vectorized geometry ────────────────────────────────────────────────
# These replace the expensive per-design calls to estimate_structural_mass,
# compute_internal_volume, compute_cg, compute_manufacturability.
# Accuracy is ~5-10% vs the full Kulfan-based functions, which is sufficient
# for surrogate-guided DE (AVL validates the final candidates anyway).

def _fast_structural_mass(x: np.ndarray, mtow: float) -> np.ndarray:
    """Vectorized structural mass estimate [kg].

    Simplified version of estimate_structural_mass: uses body_tc_root
    directly instead of building Kulfan airfoils.
    """
    from ..constants import G

    hs = x[:, _IDX_HALF_SPAN]
    rc = x[:, _IDX_WING_ROOT_CHORD]
    tr = x[:, _IDX_TAPER]
    bcr = x[:, _IDX_BODY_CHORD_RATIO]
    bw = x[:, _IDX_BODY_HALFWIDTH]
    btc = np.clip(_body_tc_from_x(x, 0.30), 0.10, 0.30)

    tip_chord = rc * tr
    body_root_chord = rc * bcr

    # Average t/c: body uses btc directly, wing uses ~10% (typical Kulfan)
    avg_tc_wing = 0.10  # good approximation for the Kulfan template
    avg_tc_body = btc * 0.8

    # Shell mass: wetted area * density * (1 + tc correction)
    density = 0.6  # kg/m²
    wing_wetted = 2.05 * (rc + tip_chord) * (hs - bw)
    body_wetted = 2.05 * (body_root_chord + rc) * bw
    shell_mass = density * (
        wing_wetted * (1 + avg_tc_wing) +
        body_wetted * (1 + avg_tc_body)
    )

    # Spar mass (beam bending)
    weight = mtow * G
    m_root = 0.5 * 2.5 * 1.5 * weight * hs / 3.0  # n_load=2.5, fos=1.5
    mac = (2 / 3) * rc * (1 + tr + tr**2) / (1 + tr)
    h_spar = np.maximum(avg_tc_wing * mac, 0.005)
    a_cap = m_root / (200e6 * h_spar)  # sigma=200 MPa
    spar_mass = 2.0 * a_cap * hs * 1500.0  # rho=1500 kg/m³

    return (shell_mass + spar_mass) * 1.20  # 20% adder


def _fast_internal_volume(x: np.ndarray) -> np.ndarray:
    """Vectorized internal volume estimate [m³].

    Elliptical cross-section approximation for the center body.
    """
    rc = x[:, _IDX_WING_ROOT_CHORD]
    bcr = x[:, _IDX_BODY_CHORD_RATIO]
    bw = x[:, _IDX_BODY_HALFWIDTH]
    btc = np.clip(_body_tc_from_x(x, 0.30), 0.10, 0.30)

    body_root_chord = rc * bcr
    # Elliptical cross-section: width=2*bw, height=btc*body_chord
    # Volume ≈ pi/4 * width * height * chord * packing_factor
    height = btc * body_root_chord
    volume = np.pi / 4 * (2 * bw) * height * body_root_chord * 0.55
    return volume


def _fast_cg(x: np.ndarray, mission: MissionCondition,
             cg_config: CGConfig) -> np.ndarray:
    """Vectorized CG x-position estimate [m].

    Simplified: weighted average of component CG positions.
    """
    rc = x[:, _IDX_WING_ROOT_CHORD]
    bcr = x[:, _IDX_BODY_CHORD_RATIO]
    bw = x[:, _IDX_BODY_HALFWIDTH]
    btc = np.clip(_body_tc_from_x(x, 0.30), 0.10, 0.30)
    hs = x[:, _IDX_HALF_SPAN]
    tr = x[:, _IDX_TAPER]
    sweep = np.radians(x[:, _IDX_SWEEP])

    body_root_chord = rc * bcr
    tip_chord = rc * tr

    # Structural mass split (approximate)
    wing_wetted = 2.05 * (rc + tip_chord) * (hs - bw)
    body_wetted = 2.05 * (body_root_chord + rc) * bw
    total_wetted = wing_wetted + body_wetted
    body_frac = body_wetted / np.maximum(total_wetted, 1e-6)

    struct_mass = _fast_structural_mass(x, mission.mtow)
    body_struct = struct_mass * body_frac * 1.0
    wing_struct = struct_mass - body_struct

    # CG positions
    x_body_cg = body_root_chord * 0.40  # body struct at 40% chord
    # Wing struct at MAC aerodynamic center
    lam = tr
    eta_mac = (1 + 2 * lam) / (3 * (1 + lam))
    body_sweep = np.radians(x[:, _IDX_SWEEP] + x[:, _IDX_BODY_SWEEP_DELTA])
    x_le_blend = bw * np.tan(body_sweep)
    x_le_mac = x_le_blend + eta_mac * (hs - bw) * np.tan(sweep)
    mac = (2 / 3) * rc * (1 + tr + tr**2) / (1 + tr)
    x_wing_cg = x_le_mac + 0.40 * mac

    # Component positions (from CGConfig)
    x_batt = body_root_chord * cg_config.battery_xc
    x_edf = body_root_chord * cg_config.edf_xc
    x_avionics = body_root_chord * cg_config.avionics_xc
    x_servo = body_root_chord * cg_config.servo_xc

    # Weighted CG
    m_batt = mission.battery.mass
    m_edf = mission.edf.mass
    m_avionics = mission.avionics_mass
    m_servo = cg_config.servo_mass
    m_gear = cg_config.gear_mass
    x_gear = body_root_chord * cg_config.gear_xc

    total_mass = (body_struct + wing_struct + m_batt + m_edf +
                  m_avionics + m_servo + m_gear)
    x_cg = (body_struct * x_body_cg + wing_struct * x_wing_cg +
             m_batt * x_batt + m_edf * x_edf +
             m_avionics * x_avionics + m_servo * x_servo +
             m_gear * x_gear) / np.maximum(total_mass, 1e-6)

    return x_cg


def _fast_manufacturability(x: np.ndarray) -> np.ndarray:
    """Vectorized manufacturability score [0-1].

    Simplified: uses design vector directly without building params.
    """
    hs = x[:, _IDX_HALF_SPAN]
    rc = x[:, _IDX_WING_ROOT_CHORD]
    tr = x[:, _IDX_TAPER]
    bcr = x[:, _IDX_BODY_CHORD_RATIO]
    btc = np.clip(_body_tc_from_x(x, 0.30), 0.10, 0.30)
    bw = x[:, _IDX_BODY_HALFWIDTH]
    sweep_delta = x[:, _IDX_BODY_SWEEP_DELTA]

    # Tip thickness
    tip_chord = rc * tr
    tip_tc = 0.08  # approximate
    thickness_tip_mm = tip_chord * tip_tc * 1000

    # Twist range and gradient
    twist_1 = x[:, _IDX_TWIST_1]
    twist_tip = x[:, _IDX_TWIST_TIP]
    twist_range = np.abs(twist_tip - twist_1)
    outer_span = hs - bw
    twist_gradient = np.where(outer_span > 0.01, twist_range / outer_span, 0)

    # Dihedral gradient (simplified)
    dih_0 = x[:, _IDX_DIHEDRAL_0]
    dih_tip = x[:, _IDX_DIHEDRAL_TIP]
    dih_range = np.abs(dih_tip - dih_0)
    dih_gradient = np.where(outer_span > 0.01, dih_range / outer_span, 0)

    # Internal volume
    body_root_chord = rc * bcr
    vol = np.pi / 4 * (2 * bw) * (btc * body_root_chord) * body_root_chord * 0.55

    def _ramp(val, good, bad):
        return np.clip((bad - val) / (bad - good + 1e-9), 0, 1)

    def _ramp_inv(val, good, bad):
        return np.clip((val - bad) / (good - bad + 1e-9), 0, 1)

    scores = (
        0.15 * _ramp(twist_gradient, 5, 20) +
        0.10 * _ramp(dih_gradient, 30, 100) +
        0.10 * 0.5 +  # dihedral breaks: approximate
        0.15 * _ramp_inv(thickness_tip_mm, 8, 3) +
        0.10 * _ramp_inv(tr, 0.25, 0.08) +
        0.10 * _ramp(bcr, 1.3, 2.0) +
        0.05 * _ramp(sweep_delta, 3, 12) +
        0.10 * _ramp(twist_range, 3, 8) +
        0.10 * _ramp_inv(vol, 0.004, 0.002) +
        0.05 * _ramp(hs, 0.8, 1.0)
    )
    return scores



# ── Main reconstruction ─────────────────────────────────────────────────────

def reconstruct_aero(
    primitives: dict,
    x: np.ndarray,
    mission: MissionCondition | None = None,
    feasibility: FeasibilityConfig | None = None,
    cg_config: CGConfig | None = None,
    aero_model=None,
    servo_config=None,
) -> dict:
    """Reconstruct all derived aerodynamic and propulsion quantities from
    surrogate primitives.

    All geometry is computed with vectorized numpy — no per-design loops
    for the expensive functions (structural mass, volume, CG, manufacturability).
    """
    mission = mission or MissionCondition()
    feasibility = feasibility or DEFAULT_FEASIBILITY
    cg_config = cg_config or DEFAULT_CG_CONFIG

    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)

    n = x.shape[0]

    # --- Geometry (vectorized, zero loops) ---
    hs = x[:, _IDX_HALF_SPAN]
    rc = x[:, _IDX_WING_ROOT_CHORD]
    tr = x[:, _IDX_TAPER]
    sweep = x[:, _IDX_SWEEP]
    bcr = x[:, _IDX_BODY_CHORD_RATIO]
    bw = x[:, _IDX_BODY_HALFWIDTH]
    btc = np.clip(_body_tc_from_x(x, 0.30), 0.10, 0.30)

    tip_chord = rc * tr
    body_root_chord = rc * bcr
    outer_span = hs - bw

    s_ref = (body_root_chord + rc) * bw + (rc + tip_chord) * outer_span
    ar = (2 * hs) ** 2 / np.maximum(s_ref, 1e-6)
    mac = (2 / 3) * rc * (1 + tr + tr ** 2) / (1 + tr)

    e_oswald = np.array([
        oswald_efficiency(a, s, t) for a, s, t in zip(ar, sweep, tr)
    ])

    # Use surrogate-predicted geometry if available, else fast fallback
    if "struct_mass" in primitives:
        struct_mass = np.atleast_1d(np.asarray(primitives["struct_mass"], dtype=float))
        internal_volume = np.atleast_1d(np.asarray(primitives["internal_volume"], dtype=float))
        manuf_score = np.atleast_1d(np.asarray(primitives["manufacturability_score"], dtype=float))
        x_cg_frac = np.atleast_1d(np.asarray(primitives["x_cg_frac"], dtype=float))
        x_cg_real = x_cg_frac * body_root_chord
        # Vs: always compute analytically (exact formula, no MLP approximation error)
        vs = None  # will be computed below from s_ref
    else:
        # Legacy fallback for models without geometry targets
        struct_mass = _fast_structural_mass(x, mission.mtow)
        internal_volume = _fast_internal_volume(x)
        x_cg_real = _fast_cg(x, mission, cg_config)
        manuf_score = _fast_manufacturability(x)
        vs = None  # computed below

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
    cm_de = np.atleast_1d(np.asarray(primitives.get("Cm_de", 0.0), dtype=float))
    cl_de_pred = np.atleast_1d(np.asarray(primitives.get("CL_de", 0.015), dtype=float))
    cl_beta_pred = np.atleast_1d(np.asarray(primitives.get("Cl_beta", 0.0), dtype=float))

    alpha_eq_rad = np.where(
        np.abs(cl_alpha) > 0.1,
        (cl_required - cl_0) / cl_alpha,
        np.radians(5.0),
    )
    alpha_eq = np.clip(np.degrees(alpha_eq_rad), -2.0, 12.0)
    alpha_eq_rad = np.radians(alpha_eq)

    # Elevon deflection: use direct surrogate prediction (AVL coupled trim)
    # if available, otherwise fall back to linear approximation
    elevon_pred = primitives.get("elevon_deflection", None)
    if elevon_pred is not None:
        elevon_defl = np.clip(
            np.atleast_1d(np.asarray(elevon_pred, dtype=float)), -25.0, 25.0
        )
    else:
        cm_untrimmed = cm_0 + cm_alpha * alpha_eq_rad
        cm_de_per_deg = np.where(np.abs(cm_de) > 1e-6, cm_de / np.degrees(1.0), -0.01)
        elevon_defl = np.clip(
            np.where(np.abs(cm_untrimmed) > 0.001, -cm_untrimmed / cm_de_per_deg, 0.0),
            -25.0, 25.0,
        )

    # Use trimmed CL from surrogate if available, else linear fallback
    cl_pred = primitives.get("CL", None)
    if cl_pred is not None:
        cl = np.atleast_1d(np.asarray(cl_pred, dtype=float))
    else:
        cl = cl_0 + cl_alpha * alpha_eq_rad
    cm = cm_0 + cm_alpha * alpha_eq_rad + cm_de * np.radians(elevon_defl)
    # Static margin: use direct surrogate prediction if available,
    # otherwise fall back to analytical CG correction
    sm_pred = primitives.get("static_margin", None)
    if sm_pred is not None:
        sm = np.atleast_1d(np.asarray(sm_pred, dtype=float))
    else:
        sm_raw = np.where(np.abs(cl_alpha) > 0.01, -cm_alpha / cl_alpha, 0.0)
        x_cg_old = body_root_chord * 0.30
        c_ref = s_ref / (2.0 * hs)
        safe_cref = np.where(c_ref > 0.01, c_ref, 1.0)
        sm = sm_raw + (x_cg_old - x_cg_real) / safe_cref
        cm = cm + cl * (x_cg_real - x_cg_old) / safe_cref

    # ── Servo hinge moment (vectorized) ──
    from ..config import ServoConfig
    _sc = servo_config or ServoConfig()
    cf_ratio_elevon = 0.25  # from controls config default
    c_flap = cf_ratio_elevon * mac
    b_flap = 0.50 * outer_span  # elevon eta_outer default
    delta_rad_servo = np.radians(np.abs(elevon_defl))
    m_hinge = (0.5 * mission.density * mission.velocity**2
               * c_flap**2 * b_flap * abs(_sc.ch_delta) * delta_rad_servo)
    servo_torque = m_hinge / _sc.n_servos_elevon * 100 / 9.81

    # Stall speed (use predicted if available, else compute)
    if vs is None:
        vs = np.sqrt(2 * weight / (mission.density * s_ref * feasibility.cl_max_clean))

    # Total CD0 = wing + body + intake + gear + bump
    from ..config import AeroModelConfig
    _ac = aero_model or AeroModelConfig()
    s_intake = np.pi / 4 * mission.edf.intake_diameter ** 2
    cd_intake = 0.05 * s_intake / np.maximum(s_ref, 1e-6)
    cd_gear = _ac.cd_gear

    # Bump fairing drag: duct protrusion above body OML at 3 stations
    duct_od = mission.edf.duct_outer_diameter
    bump_h = np.zeros(n)
    bump_start_xf = np.full(n, 0.80)
    bump_end_xf = np.full(n, 0.15)
    for xf in (0.15, 0.40, 0.80):
        body_h = _body_tc_from_x(x, xf) * body_root_chord
        protrusion = np.maximum(0.0, duct_od - body_h)
        has_bump = protrusion > 0
        bump_h = np.maximum(bump_h, protrusion)
        bump_start_xf = np.where(has_bump, np.minimum(bump_start_xf, xf), bump_start_xf)
        bump_end_xf = np.where(has_bump, np.maximum(bump_end_xf, xf), bump_end_xf)
    bump_l = np.maximum((bump_end_xf - bump_start_xf) * body_root_chord,
                        0.05 * body_root_chord)
    bump_w = np.full_like(bump_h, duct_od)
    # Hoerner form factor for streamlined protuberance
    hl = bump_h / np.maximum(bump_l, 1e-6)
    re_bump = mission.velocity * bump_l / max(mission.kinematic_viscosity, 1e-9)
    cf_bump = np.where(re_bump > 5e5,
                       0.455 / np.log10(np.maximum(re_bump, 10)) ** 2.58,
                       1.328 / np.sqrt(np.maximum(re_bump, 1e3)))
    ff_bump = 1.0 + 1.5 * hl ** 1.5 + 7.0 * hl ** 3.0
    s_wet_bump = np.pi * bump_h * bump_l / 2.0 + 2.0 * bump_h * bump_w
    cd0_bump = cf_bump * ff_bump * s_wet_bump / np.maximum(s_ref, 1e-6)
    cd0_bump = np.where(bump_h > 0, cd0_bump, 0.0)

    # Nozzle base drag (reduced by jet wake filling)
    from ..config import DuctAeroConfig, DuctGeometryConfig
    _dac = DuctAeroConfig()
    _dgc = DuctGeometryConfig()
    _fan_d = mission.edf.fan_diameter
    _a_exit = np.pi / 4 * _fan_d ** 2 * _dgc.nozzle_area_ratio
    cd_nozzle = _dac.k_base_te * _a_exit / np.maximum(s_ref, 1e-6) * (1 - _dac.f_jet_recovery)

    cd0 = cd0_wing + cd0_body + cd_intake + cd_gear + cd0_bump + cd_nozzle

    # CD and L/D: use direct surrogate predictions if available,
    # otherwise fall back to analytical reconstruction
    cdi = np.where(ar > 1.0, cl ** 2 / (np.pi * ar * e_oswald), 0.1)

    cl_de_pred = np.atleast_1d(np.asarray(primitives.get("CL_de", 0.015), dtype=float))
    delta_cl_trim = cl_de_pred * np.radians(elevon_defl)
    cd_trim_induced = np.where(ar > 1.0, delta_cl_trim**2 / (np.pi * ar * e_oswald), 0.0)
    cd_trim = np.where(
        np.abs(elevon_defl) > 0.1,
        cd_trim_induced + 0.0001 * np.abs(elevon_defl) / 10.0,
        0.0,
    )

    if "CD" in primitives:
        cd = np.atleast_1d(np.asarray(primitives["CD"], dtype=float))
    else:
        cd = cd0 + cdi + cd_trim

    # L/D for level flight: use CL_required / CD (always positive for flyable designs)
    # The surrogate L_over_D target is the AVL trim value which can be negative
    # for untrimmed designs — not useful as an optimization objective.
    ld = np.where(cd > 1e-4, cl_required / cd, 0.0)

    cd_effective = cd * feasibility.drag_margin

    # --- Propulsion reconstruction (delegated to balance.py) ---
    # Approximate duct pressure recovery (no DuctPlacement in surrogate)
    _pr_sduct = 1.0 - _dac.k_sduct * 0.32 ** 2  # avg offset_ratio ~0.32
    _pr_total = _dac.pr_intake * _pr_sduct * _dac.pr_nozzle

    drag_force = cd_effective * q * s_ref
    from ..propulsion.balance import compute_propulsion_balance
    _prop = compute_propulsion_balance(drag_force, mission.velocity,
                                        mission.edf, mission.battery,
                                        pr_total=_pr_total)
    t_available = _prop["T_available"]
    t_over_d = _prop["T_over_D"]
    p_elec = _prop["P_elec"]
    endurance_s = _prop["endurance_s"]
    endurance_min = _prop["endurance_min"]
    p_thrust = drag_force * mission.velocity
    range_km = _prop["range_km"]

    # --- Penalty & feasibility (vectorized where possible) ---
    penalty = np.zeros(n)
    is_feasible = np.zeros(n, dtype=bool)
    for i in range(n):
        p_val, c_dict = feasibility.compute_penalty(
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
            vs=float(vs[i]),
            alpha_trim=float(alpha_eq[i]),
            elevon_deflection=float(elevon_defl[i]),
            cl_beta=float(cl_beta_pred[i]) if len(cl_beta_pred) > 1 else float(cl_beta_pred[0]),
            manufacturability_score=float(manuf_score[i]),
            servo_torque_kgcm=float(servo_torque[i]),
            bump_height_mm=float(bump_h[i]) * 1000,
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
            vs=float(vs[i]),
            alpha_trim=float(alpha_eq[i]),
            elevon_deflection=float(elevon_defl[i]),
            cl_beta=float(cl_beta_pred[i]) if len(cl_beta_pred) > 1 else float(cl_beta_pred[0]),
            manufacturability_score=float(manuf_score[i]),
            servo_torque_kgcm=float(servo_torque[i]),
            bump_height_mm=float(bump_h[i]) * 1000,
        )

    result = {
        "L_over_D": ld, "CL": cl, "CM": cm, "static_margin": sm,
        "alpha_eq": alpha_eq, "CDi": cdi, "CD": cd, "CD0": cd0,
        "CD0_wing": cd0_wing, "CD0_body": cd0_body, "CD_intake": cd_intake,
        "CD_gear": cd_gear, "CD0_bump": cd0_bump, "CD_nozzle": cd_nozzle,
        "CD_trim": cd_trim, "CD_effective": cd_effective, "Cn_beta": cn_beta,
        "oswald_e": e_oswald, "AR": ar, "S_ref": s_ref, "MAC": mac,
        "struct_mass": struct_mass, "internal_volume": internal_volume,
        "Vs": vs, "CL_required": cl_required,
        "manufacturability_score": manuf_score,
        "Cl_beta": cl_beta_pred, "Cm_de": cm_de,
        "elevon_deflection": elevon_defl,
        "servo_torque_kgcm": servo_torque,
        "T_available": t_available, "T_over_D": t_over_d,
        "drag_force": drag_force, "P_thrust": p_thrust, "P_elec": p_elec,
        "endurance_s": endurance_s, "endurance_min": endurance_min,
        "penalty": penalty, "is_feasible": is_feasible,
        "constraints": c_dict,  # last iteration's constraints (used in single mode)
    }

    if single:
        constraints_save = result["constraints"]
        result = {
            k: (float(v[0]) if hasattr(v, '__len__') and len(v) > 0 else float(v))
            for k, v in result.items()
            if k != "constraints"
        }
        result["is_feasible"] = bool(result["is_feasible"])
        result["constraints"] = constraints_save

    return result
