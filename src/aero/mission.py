"""Mission condition and feasibility constraints.

MissionCondition is a data container for flight state and hardware specs.
Propulsion computations (thrust, endurance, duct mass) live in
src.propulsion.balance — not here.
"""

import numpy as np
import aerosandbox as asb
from dataclasses import dataclass, field
from ..propulsion.edf_model import (
    EDFSpec, BatterySpec, EDF_70MM, BATTERY_3S_1800,
)


@dataclass
class MissionCondition:
    """Flight condition with propulsion hardware specification.

    This is a data container. Propulsion computations (thrust, endurance,
    duct mass) are in src.propulsion.balance.
    """

    # Flight state
    velocity: float = 20.0          # [m/s] cruise speed
    altitude: float = 100.0         # [m] altitude
    mtow: float = 1.5              # [kg] max takeoff weight

    # Fixed component masses (see systems/avionics.py for detailed BOM)
    avionics_mass: float = 0.179    # [kg] detailed BOM
    auxiliary_mass: float = 0.080   # [kg] linkages, camera window, FC mount, battery box
    payload_mass: float = 0.200     # [kg] mission payload (camera, gimbal, sensors)

    # Propulsion system (fixed hardware)
    edf: EDFSpec = field(default_factory=lambda: EDF_70MM)
    battery: BatterySpec = field(default_factory=lambda: BATTERY_3S_1800)

    # Duct structure mass — set externally or use default estimate
    _duct_mass: float = 0.019       # [kg] default for 70mm EDF

    @property
    def motor_mass(self) -> float:
        return self.edf.mass

    @property
    def battery_mass(self) -> float:
        return self.battery.mass

    @property
    def duct_mass(self) -> float:
        return self._duct_mass

    @duct_mass.setter
    def duct_mass(self, value: float):
        self._duct_mass = value

    @property
    def mass_budget(self) -> float:
        """Available mass for airframe structure [kg]."""
        return (self.mtow - self.battery_mass - self.motor_mass
                - self.avionics_mass - self.auxiliary_mass
                - self.payload_mass - self.duct_mass)

    @property
    def atmosphere(self) -> asb.Atmosphere:
        return asb.Atmosphere(altitude=self.altitude)

    @property
    def density(self) -> float:
        return float(self.atmosphere.density())

    @property
    def dynamic_pressure(self) -> float:
        return 0.5 * self.density * self.velocity**2

    @property
    def weight(self) -> float:
        from ..constants import G
        return self.mtow * G

    @property
    def kinematic_viscosity(self) -> float:
        atm = self.atmosphere
        return float(atm.dynamic_viscosity() / atm.density())

    def stall_speed(self, s_ref: float, cl_max: float) -> float:
        """Stall speed [m/s] for given S_ref and CL_max."""
        return float(np.sqrt(2 * self.weight / (self.density * s_ref * cl_max)))


@dataclass
class FeasibilityConfig:
    """Feasibility constraint thresholds and penalty weights.

    Extends v1 with 3 propulsion constraints: T/D, endurance, duct fit.
    """

    # ── Aerodynamic constraints ──
    sm_min: float = 0.05           # min static margin (% MAC)
    sm_max: float = 0.25           # max static margin (% MAC)
    cm_max: float = 0.20           # max |CM| at trim
    cl_max: float = 0.75           # max CL_required at cruise (not stall CL_max)
    cl_min: float = 0.05           # min CL_required (wing not oversized)
    ar_min: float = 4.5            # min aspect ratio
    volume_min: float = 0.002      # [m³] min internal volume
    cn_beta_min: float = 0.005     # min Cn_beta (directional stability)

    # ── Propulsion constraints ──
    td_min: float = 1.2            # min T/D ratio (20% thrust margin)
    endurance_min: float = 480.0   # [s] min endurance (8 minutes)

    # ── Drag margin (VLM compensation) ──
    drag_margin: float = 1.25      # multiplier on CD for T/D and endurance

    # ── Stall / landing constraints ──
    vs_max: float = 12.0           # [m/s] max stall speed
    cl_max_clean: float = 0.85     # CL_max for clean BWB with reflex airfoil

    # ── Control authority constraints ──
    alpha_trim_max: float = 10.0   # [deg] max trim alpha
    elevon_deflection_max: float = 15.0  # [deg] max trim elevon deflection
    cl_beta_cn_beta_max: float = -30.0   # max |Cl_beta/Cn_beta| ratio (dutch roll)

    # ── Servo torque constraint ──
    servo_torque_max_kgcm: float = 1.3  # max servo torque (Emax ES3004)

    # ── Manufacturability constraint ──
    manufacturability_min: float = 0.40  # min composite score [0-1]

    # ── Aerodynamic penalty weights ──
    w_sm_lo: float = 20.0
    w_sm_hi: float = 25.0
    w_trim: float = 15.0
    w_mass: float = 10.0
    w_cl: float = 10.0
    w_ar: float = 5.0
    w_volume: float = 15.0
    w_cn_beta: float = 15.0

    # ── Propulsion penalty weights ──
    w_td: float = 20.0            # T/D margin (critical: can't fly)
    w_endurance: float = 15.0     # endurance

    # ── Stall / control penalty weights ──
    w_vs: float = 20.0            # stall speed (critical: can't land)
    w_alpha_trim: float = 10.0    # trim alpha out of range
    w_elevon_defl: float = 15.0   # elevon deflection too large for trim
    w_dutch_roll: float = 10.0    # Cl_beta/Cn_beta ratio (lateral coupling)
    w_servo_torque: float = 15.0  # servo torque exceeds spec
    w_manufacturability: float = 10.0  # manufacturability score too low

    def compute_penalty(
        self,
        static_margin: float,
        cm: float,
        struct_mass: float,
        mass_budget: float,
        cl_required: float,
        ar: float,
        internal_volume: float | None = None,
        cn_beta: float | None = None,
        t_over_d: float | None = None,
        endurance_s: float | None = None,
        vs: float | None = None,
        alpha_trim: float | None = None,
        elevon_deflection: float | None = None,
        cl_beta: float | None = None,
        manufacturability_score: float | None = None,
        servo_torque_kgcm: float | None = None,
    ) -> tuple[float, dict]:
        """Compute total penalty and constraint violations.

        Returns (penalty, constraints_dict). Violations g <= 0 means satisfied.
        """
        # ── Aerodynamic constraints ──
        sm_range = max(self.sm_max - self.sm_min, 0.01)
        penalty = 0.0
        constraints = {}

        g_sm_lo = self.sm_min - static_margin
        g_sm_hi = static_margin - self.sm_max
        g_trim = abs(cm) - self.cm_max
        g_mass = struct_mass - mass_budget
        g_cl = cl_required - self.cl_max
        g_ar = self.ar_min - ar

        penalty += self.w_sm_lo * max(0, g_sm_lo / sm_range)
        penalty += self.w_sm_hi * max(0, g_sm_hi / sm_range)
        penalty += self.w_trim * max(0, g_trim / max(self.cm_max, 0.01))
        penalty += self.w_mass * max(0, g_mass / max(mass_budget, 0.01))
        penalty += self.w_cl * max(0, g_cl / max(self.cl_max - self.cl_min, 0.01))
        penalty += self.w_ar * max(0, g_ar / max(self.ar_min, 0.01))

        constraints.update({
            "g_sm_lo": g_sm_lo, "g_sm_hi": g_sm_hi, "g_trim": g_trim,
            "g_mass": g_mass, "g_cl": g_cl, "g_ar": g_ar,
        })

        # Internal volume
        if internal_volume is not None:
            g_vol = self.volume_min - internal_volume
            penalty += self.w_volume * max(0, g_vol / max(self.volume_min, 0.0001))
            constraints["g_volume"] = g_vol

        # Directional stability
        if cn_beta is not None:
            g_cn = self.cn_beta_min - cn_beta
            penalty += self.w_cn_beta * max(0, g_cn / max(self.cn_beta_min, 0.001))
            constraints["g_cn_beta"] = g_cn

        # ── Propulsion constraints ──
        if t_over_d is not None:
            g_td = self.td_min - t_over_d
            penalty += self.w_td * max(0, g_td / max(self.td_min, 0.01))
            constraints["g_td"] = g_td

        if endurance_s is not None:
            g_endurance = self.endurance_min - endurance_s
            penalty += self.w_endurance * max(0, g_endurance / max(self.endurance_min, 1.0))
            constraints["g_endurance"] = g_endurance

        # ── Stall / control constraints ──
        if vs is not None:
            g_vs = vs - self.vs_max
            penalty += self.w_vs * max(0, g_vs / max(self.vs_max, 1.0))
            constraints["g_vs"] = g_vs

        if alpha_trim is not None:
            alpha_trim_min = -2.0
            alpha_range = max(self.alpha_trim_max - alpha_trim_min, 1.0)
            g_alpha_lo = alpha_trim_min - alpha_trim
            g_alpha_hi = alpha_trim - self.alpha_trim_max
            penalty += self.w_alpha_trim * max(0, g_alpha_lo / alpha_range)
            penalty += self.w_alpha_trim * max(0, g_alpha_hi / alpha_range)
            constraints["g_alpha_trim_lo"] = g_alpha_lo
            constraints["g_alpha_trim_hi"] = g_alpha_hi

        # ── Elevon deflection constraint ──
        if elevon_deflection is not None:
            g_elev = abs(elevon_deflection) - self.elevon_deflection_max
            penalty += self.w_elevon_defl * max(0, g_elev / max(self.elevon_deflection_max, 1.0))
            constraints["g_elevon_defl"] = g_elev

        # ── Servo torque constraint ──
        if servo_torque_kgcm is not None:
            g_servo = servo_torque_kgcm - self.servo_torque_max_kgcm
            penalty += self.w_servo_torque * max(0, g_servo / max(self.servo_torque_max_kgcm, 0.01))
            constraints["g_servo_torque"] = g_servo

        # ── Dutch roll lateral coupling ──
        if cl_beta is not None and cn_beta is not None and cn_beta > 0.001:
            ratio = cl_beta / cn_beta
            g_dutch = ratio - self.cl_beta_cn_beta_max  # ratio is negative, max is negative
            # Penalize if ratio is too negative (strong adverse coupling)
            penalty += self.w_dutch_roll * max(0, -g_dutch / max(abs(self.cl_beta_cn_beta_max), 1.0))
            constraints["g_dutch_roll"] = g_dutch

        # ── Manufacturability constraint ──
        if manufacturability_score is not None:
            g_manuf = self.manufacturability_min - manufacturability_score
            penalty += self.w_manufacturability * max(0, g_manuf / max(self.manufacturability_min, 0.01))
            constraints["g_manufacturability"] = g_manuf

        return penalty, constraints

    def is_feasible(
        self,
        static_margin: float,
        cm: float,
        struct_mass: float,
        mass_budget: float,
        cl_required: float,
        ar: float,
        internal_volume: float | None = None,
        cn_beta: float | None = None,
        t_over_d: float | None = None,
        endurance_s: float | None = None,
        success: bool = True,
        vs: float | None = None,
        alpha_trim: float | None = None,
        elevon_deflection: float | None = None,
        cl_beta: float | None = None,
        manufacturability_score: float | None = None,
        servo_torque_kgcm: float | None = None,
    ) -> bool:
        """Check if all constraints are satisfied."""
        checks = [
            success,
            self.cl_min < cl_required < self.cl_max,
            struct_mass < mass_budget,
            self.sm_min <= static_margin <= self.sm_max,
            abs(cm) < self.cm_max,
            ar > self.ar_min,
        ]
        if internal_volume is not None:
            checks.append(internal_volume >= self.volume_min)
        if cn_beta is not None:
            checks.append(cn_beta >= self.cn_beta_min)
        if t_over_d is not None:
            checks.append(t_over_d >= self.td_min)
        if endurance_s is not None:
            checks.append(endurance_s >= self.endurance_min)
        if vs is not None:
            checks.append(vs <= self.vs_max)
        if alpha_trim is not None:
            checks.append(-2.0 <= alpha_trim <= self.alpha_trim_max)
        if elevon_deflection is not None:
            checks.append(abs(elevon_deflection) <= self.elevon_deflection_max)
        if cl_beta is not None and cn_beta is not None and cn_beta > 0.001:
            checks.append(cl_beta / cn_beta >= self.cl_beta_cn_beta_max)
        if servo_torque_kgcm is not None:
            checks.append(servo_torque_kgcm <= self.servo_torque_max_kgcm)
        if manufacturability_score is not None:
            checks.append(manufacturability_score >= self.manufacturability_min)
        return all(checks)


DEFAULT_FEASIBILITY = FeasibilityConfig()
