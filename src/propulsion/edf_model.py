"""EDF propulsion model for BWB flying wing.

Provides thrust, power, endurance, and geometric integration constraints
for Electric Ducted Fan units. The EDF is fixed hardware — not optimized —
but its integration into the airframe creates critical design constraints.

References:
- Schubeler DS-51 (70mm): T_static ≈ 10N, V_max ≈ 35 m/s
- Wemotec Mini Fan (70mm): similar performance class
- Actuator disk theory for T(V) model
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class EDFSpec:
    """Electric Ducted Fan specification (fixed hardware)."""

    name: str
    fan_diameter: float          # [m] fan blade diameter
    duct_outer_diameter: float   # [m] duct + shroud outer diameter
    thrust_static: float         # [N] static thrust (V=0)
    velocity_max: float          # [m/s] max exit velocity (no-load)
    eta_fan: float               # [-] fan isentropic efficiency
    eta_motor: float             # [-] motor + ESC efficiency
    mass: float                  # [kg] EDF unit mass (fan + motor + shroud)
    intake_diameter: float       # [m] intake opening diameter


@dataclass(frozen=True)
class BatterySpec:
    """LiPo battery specification."""

    name: str
    mass: float                  # [kg]
    voltage: float               # [V] nominal voltage
    capacity_ah: float           # [Ah] capacity
    specific_energy: float       # [Wh/kg] gravimetric energy density

    @property
    def energy_wh(self) -> float:
        """Total energy [Wh]."""
        return self.mass * self.specific_energy

    @property
    def energy_j(self) -> float:
        """Total energy [J]."""
        return self.energy_wh * 3600.0


# ── Presets ──────────────────────────────────────────────────────────────

EDF_70MM = EDFSpec(
    name="EDF 70mm (Schubeler DS-51 class)",
    fan_diameter=0.070,
    duct_outer_diameter=0.078,
    thrust_static=10.0,
    velocity_max=35.0,
    eta_fan=0.55,
    eta_motor=0.85,
    mass=0.060,
    intake_diameter=0.075,
)

EDF_90MM = EDFSpec(
    name="EDF 90mm (Schubeler DS-75 class)",
    fan_diameter=0.090,
    duct_outer_diameter=0.100,
    thrust_static=22.0,
    velocity_max=45.0,
    eta_fan=0.60,
    eta_motor=0.87,
    mass=0.130,
    intake_diameter=0.095,
)

BATTERY_3S_800 = BatterySpec(
    name="3S 800mAh LiPo",
    mass=0.070,
    voltage=11.1,
    capacity_ah=0.800,
    specific_energy=130.0,
)

BATTERY_3S_1300 = BatterySpec(
    name="3S 1300mAh LiPo",
    mass=0.110,
    voltage=11.1,
    capacity_ah=1.300,
    specific_energy=130.0,
)

BATTERY_3S_1800 = BatterySpec(
    name="3S 1800mAh LiPo",
    mass=0.150,
    voltage=11.1,
    capacity_ah=1.800,
    specific_energy=130.0,
)

BATTERY_4S_1000 = BatterySpec(
    name="4S 1000mAh LiPo",
    mass=0.120,
    voltage=14.8,
    capacity_ah=1.000,
    specific_energy=125.0,
)


# ── Thrust model ─────────────────────────────────────────────────────────

def thrust_at_speed(edf: EDFSpec, velocity: float) -> float:
    """Available thrust [N] at forward speed [m/s].

    Uses a simplified actuator-disk model:
        T(V) = T_static * (1 - V / V_max)

    This is conservative: real EDFs have a more gradual drop-off,
    but this keeps the constraint on the safe side.
    """
    if velocity >= edf.velocity_max:
        return 0.0
    return edf.thrust_static * (1.0 - velocity / edf.velocity_max)


def thrust_curve(edf: EDFSpec, v_range: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (velocities, thrusts) arrays for plotting."""
    if v_range is None:
        v_range = np.linspace(0, edf.velocity_max, 100)
    thrusts = np.array([thrust_at_speed(edf, v) for v in v_range])
    return v_range, thrusts


# ── Power & endurance ────────────────────────────────────────────────────

def power_required(drag: float, velocity: float, edf: EDFSpec) -> dict:
    """Compute power balance from total drag [N] and cruise speed [m/s].

    Returns dict with:
        P_thrust  — thrust power [W] = D * V
        P_shaft   — shaft power [W] = P_thrust / eta_fan
        P_elec    — electrical power [W] = P_shaft / eta_motor
    """
    p_thrust = drag * velocity
    p_shaft = p_thrust / edf.eta_fan
    p_elec = p_shaft / edf.eta_motor
    return {
        "P_thrust": p_thrust,
        "P_shaft": p_shaft,
        "P_elec": p_elec,
    }


def endurance(drag: float, velocity: float, edf: EDFSpec,
              battery: BatterySpec,
              usable_fraction: float = 0.80) -> dict:
    """Compute endurance and range from drag and battery.

    Parameters
    ----------
    drag : float
        Total aerodynamic drag [N] at cruise.
    velocity : float
        Cruise speed [m/s].
    edf : EDFSpec
    battery : BatterySpec
    usable_fraction : float
        Fraction of battery energy actually usable (LiPo safety margin).

    Returns
    -------
    dict with:
        endurance_s   — flight time [s]
        endurance_min — flight time [min]
        range_m       — range [m]
        range_km      — range [km]
        P_elec        — electrical power [W]
        T_available   — thrust available at cruise [N]
        T_over_D      — thrust margin T_available / D
    """
    pwr = power_required(drag, velocity, edf)
    p_elec = pwr["P_elec"]

    usable_energy = battery.energy_j * usable_fraction

    t_available = thrust_at_speed(edf, velocity)

    if p_elec > 0:
        endurance_s = usable_energy / p_elec
    else:
        endurance_s = float("inf")

    return {
        "endurance_s": endurance_s,
        "endurance_min": endurance_s / 60.0,
        "range_m": velocity * endurance_s,
        "range_km": velocity * endurance_s / 1000.0,
        "P_elec": p_elec,
        "P_shaft": pwr["P_shaft"],
        "P_thrust": pwr["P_thrust"],
        "T_available": t_available,
        "T_over_D": t_available / drag if drag > 0.01 else float("inf"),
    }


# ── Geometric constraints ────────────────────────────────────────────────

def duct_fits_in_body(body_tc_root: float, body_chord: float,
                      edf: EDFSpec, safety_margin: float = 0.80) -> bool:
    """Check if the EDF duct fits inside the BWB center body.

    The available height is body_tc * body_chord * safety_margin.
    The duct needs duct_outer_diameter of clearance.
    """
    available_height = body_tc_root * body_chord * safety_margin
    return available_height >= edf.duct_outer_diameter


def min_body_tc_for_duct(body_chord: float, edf: EDFSpec,
                         safety_margin: float = 0.80) -> float:
    """Minimum body t/c ratio to fit the EDF duct."""
    if body_chord <= 0:
        return 1.0
    return edf.duct_outer_diameter / (body_chord * safety_margin)


def intake_drag(edf: EDFSpec, s_ref: float, cd_intake_coeff: float = 0.05) -> float:
    """Estimate intake drag coefficient increment.

    NACA submerged intake drag: CD_intake ≈ cd_intake_coeff * A_intake / S_ref
    Typical cd_intake_coeff for flush NACA intake: 0.04 - 0.06.
    """
    a_intake = np.pi / 4 * edf.intake_diameter**2
    return cd_intake_coeff * a_intake / s_ref if s_ref > 0 else 0.0
