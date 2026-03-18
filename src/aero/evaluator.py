"""Aerodynamic evaluator — AVL + drag + stability + propulsion.

Uses AVL for inviscid aerodynamics and direct stability derivatives
(CL_0, CL_alpha, CM_0, CM_alpha, Cn_beta), plus NeuralFoil/analytical
for viscous drag.

Produces 7 surrogate primitives plus full derived quantities including
propulsion balance (T/D, endurance).
"""

import numpy as np
from ..parameterization.design_variables import BWBParams
from ..parameterization.bwb_aircraft import (
    build_airplane, compute_wing_area, compute_mac, compute_aspect_ratio,
    estimate_structural_mass, compute_internal_volume,
)
from .mission import MissionCondition, FeasibilityConfig, DEFAULT_FEASIBILITY
from .drag import compute_wing_cd0, compute_body_cd0, oswald_efficiency
from .avl_runner import run_avl_stability
from ..propulsion.edf_model import (
    thrust_at_speed, endurance as compute_endurance_full,
    duct_fits_in_body, intake_drag,
)

_DEFAULT_AVL_CMD = "d:/UAV/tools/avl/avl.exe"


class AeroEvaluator:
    """Evaluate BWB aircraft aerodynamics via AVL + NeuralFoil."""

    def __init__(self, mission: MissionCondition | None = None,
                 feasibility: FeasibilityConfig | None = None,
                 use_neuralfoil: bool = True,
                 avl_command: str = _DEFAULT_AVL_CMD):
        self.mission = mission or MissionCondition()
        self.feasibility = feasibility or DEFAULT_FEASIBILITY
        self.use_neuralfoil = use_neuralfoil
        self.avl_command = avl_command

    def evaluate(self, params: BWBParams) -> dict:
        """Full aero + stability + propulsion evaluation.

        Uses two AVL runs:
        1. alpha=0 -> CL_0, CM_0, CL_alpha, CM_alpha, Cn_beta
        2. alpha=trim -> CL, CM at operating point

        Returns dict with 7 primitives, derived quantities, and propulsion balance.
        """
        mission = self.mission
        airplane = build_airplane(params)
        s_ref = compute_wing_area(params)
        mac = compute_mac(params)
        ar = compute_aspect_ratio(params)
        cl_required = mission.weight / (mission.dynamic_pressure * s_ref)

        try:
            # AVL at alpha=0: get CL_0, CM_0 + all stability derivatives
            avl0 = run_avl_stability(
                airplane, alpha=0.0, velocity=mission.velocity,
                avl_command=self.avl_command,
            )
            if avl0 is None:
                raise RuntimeError("AVL failed at alpha=0")

            cl_0 = avl0["CL"]
            cm_0 = avl0["Cm"]
            cl_alpha = avl0["CLa"]      # direct analytical derivative
            cm_alpha = avl0["Cma"]      # direct analytical derivative
            cn_beta = avl0["Cnb"]       # direct analytical derivative
            cl_beta = avl0.get("Clb", 0.0)

            # Trim alpha from linearized CL(alpha) = CL_0 + CL_alpha * alpha
            if abs(cl_alpha) > 0.1:
                alpha_eq = np.degrees((cl_required - cl_0) / cl_alpha)
            else:
                alpha_eq = 5.0
            alpha_eq = float(np.clip(alpha_eq, -2, 12))

            # AVL at trim alpha: get CL, CM at operating point
            avl_t = run_avl_stability(
                airplane, alpha=alpha_eq, velocity=mission.velocity,
                avl_command=self.avl_command,
            )
            if avl_t is None:
                raise RuntimeError("AVL failed at trim alpha")

            cl = avl_t["CL"]
            cm = avl_t["Cm"]

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
                                        mission.kinematic_viscosity)
        else:
            from .drag import estimate_parasite_drag
            cd0_wing = estimate_parasite_drag(params, mission.velocity,
                                              mission.kinematic_viscosity) * 0.7

        cd0_body = compute_body_cd0(params, alpha_eq, mission.velocity,
                                     mission.kinematic_viscosity, s_ref, cd_intake)

        cd0 = cd0_wing + cd0_body
        e = oswald_efficiency(ar, params.le_sweep_deg, params.taper_ratio)
        cd_induced = cl**2 / (np.pi * ar * e) if ar > 1.0 else 0.1
        cd = cd0 + cd_induced
        l_over_d = cl / cd if cd > 1e-4 else 0.0
        cd_effective = cd * self.feasibility.drag_margin

        # ── Structural + volume ──
        struct_mass = estimate_structural_mass(params, mtow=mission.mtow)
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

        # Duct fit constraint
        body_chord = params.body_root_chord
        duct_ok = duct_fits_in_body(params.body_tc_root, body_chord, mission.edf)

        # ── Feasibility ──
        is_feasible = fc.is_feasible(
            static_margin, cm, struct_mass, mission.mass_budget,
            cl_required, ar, internal_volume, cn_beta,
            t_over_d, endurance_s, duct_ok, success,
            vs=vs, alpha_trim=alpha_eq,
        )
        penalty, constraints = fc.compute_penalty(
            static_margin, cm, struct_mass, mission.mass_budget,
            cl_required, ar, internal_volume, cn_beta,
            t_over_d, endurance_s, duct_ok,
            vs=vs, alpha_trim=alpha_eq,
        )

        return {
            # 7 surrogate primitives
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
            "CD_intake": cd_intake,
            # Feasibility
            "is_feasible": is_feasible,
            "penalty": penalty,
            "constraints": constraints,
            "success": success,
        }

def quick_eval(params: BWBParams, mission: MissionCondition | None = None) -> dict:
    """Convenience function for single evaluation."""
    return AeroEvaluator(mission).evaluate(params)
