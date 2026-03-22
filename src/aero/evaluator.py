"""Aerodynamic evaluator — AVL + drag + stability + propulsion + CG.

Uses AVL for inviscid aerodynamics and direct stability derivatives
(CL_0, CL_alpha, CM_0, CM_alpha, Cn_beta), plus NeuralFoil/analytical
for viscous drag.

v2: Adds control surface support (elevon trim) and real CG computation.

Produces 7 surrogate primitives plus full derived quantities including
propulsion balance (T/D, endurance) and control authority.
"""

import numpy as np
from ..parameterization.design_variables import BWBParams
from ..parameterization.bwb_aircraft import (
    build_airplane, compute_wing_area, compute_mac, compute_aspect_ratio,
    estimate_structural_mass, compute_internal_volume,
)
from .mission import MissionCondition, FeasibilityConfig, DEFAULT_FEASIBILITY
from .drag import compute_wing_cd0, compute_body_cd0, oswald_efficiency
from .avl_runner import run_avl_stability, ControlConfig
from ..propulsion.edf_model import (
    thrust_at_speed, endurance as compute_endurance_full,
    duct_fits_in_body, intake_drag,
)
from ..systems.cg import compute_cg, CGConfig, DEFAULT_CG_CONFIG
from ..evaluation.manufacturability import compute_manufacturability

class AeroEvaluator:
    """Evaluate BWB aircraft aerodynamics via AVL + NeuralFoil."""

    def __init__(self, mission: MissionCondition | None = None,
                 feasibility: FeasibilityConfig | None = None,
                 use_neuralfoil: bool = True,
                 avl_command: str | None = None,
                 controls: ControlConfig | None = None,
                 cg_config: CGConfig | None = None,
                 use_cg: bool = True,
                 aero_model=None,
                 structure_config=None,
                 **kwargs):
        self.mission = mission or MissionCondition()
        self.feasibility = feasibility or DEFAULT_FEASIBILITY
        self.use_neuralfoil = use_neuralfoil
        from ..config import load_config
        if avl_command is None:
            _cfg = load_config()
            avl_command = _cfg.get("avl", {}).get("command", "avl")
        else:
            _cfg = load_config()
        self.avl_command = avl_command
        avl_cfg = _cfg.get("avl", {})
        self.avl_spanwise_res = avl_cfg.get("spanwise_res", 20)
        self.avl_chordwise_res = avl_cfg.get("chordwise_res", 6)
        self.avl_timeout = avl_cfg.get("timeout", 10.0)
        self.controls = controls or ControlConfig.default_bwb()
        self.cg_config = cg_config or DEFAULT_CG_CONFIG
        self.use_cg = use_cg
        self.aero_model = aero_model
        self.structure_config = structure_config

    def evaluate(self, params: BWBParams) -> dict:
        """Full aero + stability + propulsion + control evaluation.

        Uses two AVL runs:
        1. alpha=0 -> CL_0, CM_0, CL_alpha, CM_alpha, Cn_beta
        2. alpha=trim -> CL, CM at operating point (with elevon trim)

        Returns dict with 7 primitives, derived quantities, propulsion balance,
        and control authority metrics.
        """
        mission = self.mission

        # ── Compute real CG ──
        x_cg = None
        cg_result = None
        if self.use_cg:
            cg_result = compute_cg(
                params,
                battery_mass=mission.battery_mass,
                edf_mass=mission.motor_mass,
                avionics_mass=mission.avionics_mass,
                config=self.cg_config,
                structure_config=self.structure_config,
            )
            x_cg = cg_result["x_cg"]

        airplane = build_airplane(params, x_cg=x_cg)
        s_ref = compute_wing_area(params)
        mac = compute_mac(params)
        ar = compute_aspect_ratio(params)
        cl_required = mission.weight / (mission.dynamic_pressure * s_ref)

        # Default control derivative values
        elevon_deflection = 0.0
        cl_de = 0.0
        cm_de = 0.0
        cl_da = 0.0
        cn_da = 0.0

        try:
            # AVL at alpha=0: get CL_0, CM_0 + all stability derivatives
            # trim=False: we want untrimmed CM_0 for the surrogate
            avl0 = run_avl_stability(
                airplane, alpha=0.0, velocity=mission.velocity,
                avl_command=self.avl_command,
                controls=self.controls, trim=False,
                spanwise_res=self.avl_spanwise_res,
                chordwise_res=self.avl_chordwise_res,
                timeout=self.avl_timeout,
            )
            if avl0 is None:
                raise RuntimeError("AVL failed at alpha=0")

            cl_0 = avl0["CL"]
            cm_0 = avl0["Cm"]
            cl_alpha = avl0["CLa"]      # direct analytical derivative
            cm_alpha = avl0["Cma"]      # direct analytical derivative
            cn_beta = avl0["Cnb"]       # direct analytical derivative
            cl_beta = avl0.get("Clb", 0.0)

            # Extract control derivatives if available
            # AVL uses CLd01, Cmd01 format (zero-padded index)
            cl_de = avl0.get("CLd01", avl0.get("CLd1", 0.0))
            cm_de = avl0.get("Cmd01", avl0.get("Cmd1", 0.0))
            cl_da = avl0.get("Cld02", avl0.get("Cld2", 0.0))
            cn_da = avl0.get("Cnd02", avl0.get("Cnd2", 0.0))

            # Trim alpha from linearized CL(alpha) = CL_0 + CL_alpha * alpha
            if abs(cl_alpha) > 0.1:
                alpha_eq = np.degrees((cl_required - cl_0) / cl_alpha)
            else:
                alpha_eq = 5.0
            from ..config import AeroModelConfig
            _ac = self.aero_model or AeroModelConfig()
            alpha_eq = float(np.clip(alpha_eq, _ac.alpha_trim_clip_min, _ac.alpha_trim_clip_max))

            # AVL at trim alpha: with elevon trim (CM→0)
            avl_t = run_avl_stability(
                airplane, alpha=alpha_eq, velocity=mission.velocity,
                avl_command=self.avl_command,
                controls=self.controls,
                spanwise_res=self.avl_spanwise_res,
                chordwise_res=self.avl_chordwise_res,
                timeout=self.avl_timeout,
            )
            if avl_t is None:
                raise RuntimeError("AVL failed at trim alpha")

            cl = avl_t["CL"]
            cm = avl_t["Cm"]

            # Get trim elevon deflection from AVL output
            elevon_deflection = avl_t.get("delta_elevon", 0.0)

            # Update control derivatives from trim run (more accurate)
            cl_de = avl_t.get("CLd01", cl_de)
            cm_de = avl_t.get("Cmd01", cm_de)
            cl_da = avl_t.get("Cld02", cl_da)
            cn_da = avl_t.get("Cnd02", cn_da)

            success = True
        except Exception:
            cl = cl_required
            cm = 0.0
            cl_0 = 0.0
            cm_0 = 0.0
            cl_alpha = 2 * np.pi * ar / (ar + 2)
            cm_alpha = 0.0
            alpha_eq = 5.0
            cn_beta = 0.0
            cl_beta = 0.0
            success = False

        # ── Drag buildup ──
        cd_intake = intake_drag(mission.edf, s_ref)

        if self.use_neuralfoil:
            cd0_wing = compute_wing_cd0(params, alpha_eq, mission.velocity,
                                        mission.kinematic_viscosity,
                                        config=self.aero_model)
        else:
            from .drag import estimate_parasite_drag
            cd0_wing = estimate_parasite_drag(params, mission.velocity,
                                              mission.kinematic_viscosity) * 0.7

        cd0_body = compute_body_cd0(params, alpha_eq, mission.velocity,
                                     mission.kinematic_viscosity, s_ref, cd_intake,
                                     config=self.aero_model)

        cd0 = cd0_wing + cd0_body
        e = oswald_efficiency(ar, params.le_sweep_deg, params.taper_ratio)
        cd_induced = cl**2 / (np.pi * ar * e) if ar > 1.0 else 0.1

        # Trim drag: additional induced drag from elevon deflection
        # Approximate: delta_CD_trim ≈ (CL_de * delta_e)^2 / (pi * AR * e)
        # plus profile drag increment ≈ 0.0001 * |delta_e_deg| / 10
        cd_trim = 0.0
        if abs(elevon_deflection) > 0.1:
            delta_cl_trim = cl_de * np.radians(elevon_deflection)
            cd_trim = (delta_cl_trim ** 2 / (np.pi * ar * e) if ar > 1.0 else 0.0)
            cd_trim += 0.0001 * abs(elevon_deflection) / 10.0  # profile drag increment

        cd = cd0 + cd_induced + cd_trim
        l_over_d = cl / cd if cd > 1e-4 else 0.0
        cd_effective = cd * self.feasibility.drag_margin

        # ── Structural + volume ──
        struct_mass = estimate_structural_mass(params, mtow=mission.mtow,
                                                structure_config=self.structure_config)
        internal_volume = compute_internal_volume(params)
        static_margin = -cm_alpha / cl_alpha if abs(cl_alpha) > 0.01 else 0.0

        # ── Stall speed ──
        fc = self.feasibility
        vs = mission.stall_speed(s_ref, fc.cl_max_clean)

        # ── Propulsion balance (uses cd_effective for conservative margin) ──
        drag_force = cd_effective * mission.dynamic_pressure * s_ref
        t_available = thrust_at_speed(mission.edf, mission.velocity)
        t_over_d = t_available / drag_force if drag_force > 0.01 else float("inf")

        prop = compute_endurance_full(drag_force, mission.velocity,
                                       mission.edf, mission.battery)
        endurance_s = prop["endurance_s"]

        # Duct fit constraint (adaptive placement + station-by-station clearance)
        body_chord = params.body_root_chord
        from ..propulsion.duct_geometry import compute_duct_placement, validate_duct_clearance
        from ..config import duct_from_config, load_config
        _duct_cfg = duct_from_config(load_config())
        placement = compute_duct_placement(params, mission.edf, config=_duct_cfg)
        _duct_ok, _clr = validate_duct_clearance(placement, params, min_clearance_mm=0.0)
        min_duct_clearance_mm = min(
            min(r.clearance_top_mm, r.clearance_bot_mm) for r in _clr
        ) if _clr else 0.0
        duct_ok = min_duct_clearance_mm >= 0.0

        # Manufacturability score (geometry-only, no AVL needed)
        manuf = compute_manufacturability(params)
        manuf_score = manuf["manufacturability_score"]

        # ── Feasibility ──
        is_feasible = fc.is_feasible(
            static_margin, cm, struct_mass, mission.mass_budget,
            cl_required, ar, internal_volume, cn_beta,
            t_over_d, endurance_s, duct_ok, success,
            vs=vs, alpha_trim=alpha_eq,
            elevon_deflection=elevon_deflection,
            cl_beta=cl_beta,
            manufacturability_score=manuf_score,
        )
        penalty, constraints = fc.compute_penalty(
            static_margin, cm, struct_mass, mission.mass_budget,
            cl_required, ar, internal_volume, cn_beta,
            t_over_d, endurance_s, duct_ok,
            vs=vs, alpha_trim=alpha_eq,
            elevon_deflection=elevon_deflection,
            cl_beta=cl_beta,
            manufacturability_score=manuf_score,
        )

        result = {
            # 7 surrogate primitives (unchanged for backward compat)
            "CL_0": cl_0,
            "CL_alpha": cl_alpha,
            "CM_0": cm_0,
            "CM_alpha": cm_alpha,
            "CD0_wing": cd0_wing,
            "CD0_body": cd0_body,
            "Cn_beta": cn_beta,
            # Derived aerodynamics
            "CL": cl,
            "CD": cd,
            "CD0": cd0,
            "CDi": cd_induced,
            "CD_trim": cd_trim,
            "CD_effective": cd_effective,
            "CM": cm,
            "L_over_D": l_over_d,
            "alpha_eq": alpha_eq,
            "CL_required": cl_required,
            "Cl_beta": cl_beta,
            "static_margin": static_margin,
            "oswald_e": e,
            "S_ref": s_ref,
            "AR": ar,
            "MAC": mac,
            # Control authority
            "elevon_deflection": elevon_deflection,
            "CL_de": cl_de,
            "Cm_de": cm_de,
            "Cl_da": cl_da,
            "Cn_da": cn_da,
            # CG
            "x_cg": x_cg if x_cg is not None else params.body_root_chord * 0.30,
            "x_cg_frac": (x_cg / body_chord if x_cg else 0.30),
            # Manufacturability
            "manufacturability_score": manuf_score,
            # Structure
            "struct_mass": struct_mass,
            "internal_volume": internal_volume,
            "Vs": vs,
            # Propulsion
            "T_over_D": t_over_d,
            "T_available": t_available,
            "drag_force": drag_force,
            "P_elec": prop["P_elec"],
            "endurance_s": endurance_s,
            "endurance_min": prop["endurance_min"],
            "range_km": prop["range_km"],
            "duct_fits": duct_ok,
            "min_duct_clearance_mm": min_duct_clearance_mm,
            "CD_intake": cd_intake,
            # Feasibility
            "is_feasible": is_feasible,
            "penalty": penalty,
            "constraints": constraints,
            "success": success,
        }

        if cg_result:
            # Store as JSON-serializable dicts (not dataclass objects)
            result["cg_components"] = [
                {"name": c.name, "mass": c.mass, "x_cg": c.x_cg}
                for c in cg_result["components"]
            ]

        return result

def quick_eval(params: BWBParams, mission: MissionCondition | None = None) -> dict:
    """Convenience function for single evaluation."""
    return AeroEvaluator(mission).evaluate(params)
