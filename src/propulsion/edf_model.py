"""EDF propulsion model for BWB flying wing.

Provides thrust, power, and endurance for Electric Ducted Fan units.

Thrust model: power-law lapse T(V) = T0 × (1 - (V/Vmax)^n), n=1.5.
This is a middle ground between the linear (n=1, too conservative) and
quadratic (n=2, too optimistic) models. Validated against Schubeler
DS-30/DS-51 published thrust curves and eCalc fanCalc predictions.

References:
- Schubeler DS-30-AXI HDS 70mm: T_static ≈ 10N, V_max ≈ 35 m/s
- NASA TN-D-4014: ducted fan performance theory
- eCalc fanCalc: empirical EDF performance validation
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

    def discharge_efficiency(self, dod: float = 0.80) -> float:
        """Voltage sag efficiency for LiPo discharge up to given DOD.

        Models the integral of V(DOD)/V_nom over [0, dod].
        For 3S LiPo: voltage drops from 12.6V to 11.1V over 0-80% DOD,
        then rapidly below.  Polynomial fit from typical discharge curves:
            eta = 1.0 - 0.05 * dod - 0.15 * dod^3

        Returns efficiency factor (0.88-0.97 for typical dod=0.80).
        """
        dod = max(0.0, min(dod, 1.0))
        return 1.0 - 0.05 * dod - 0.15 * dod ** 3


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

def thrust_at_speed(edf: EDFSpec, velocity: float, n_lapse: float = 1.5) -> float:
    """Available thrust [N] at forward speed [m/s].

    Power-law thrust lapse model:
        T(V) = T_static × (1 - (V / V_max)^n)

    With n=1.5 (default), this sits between:
    - n=1.0 (linear, conservative, old model)
    - n=2.0 (quadratic, optimistic)

    Validated against Schubeler DS-30 data and eCalc predictions.
    At 25 m/s with V_max=35: linear gives 2.9N, n=1.5 gives 4.0N,
    quadratic gives 4.9N. Empirical data suggests ~3.5-4.5N.
    """
    if velocity >= edf.velocity_max:
        return 0.0
    v_ratio = velocity / edf.velocity_max
    return edf.thrust_static * (1.0 - v_ratio ** n_lapse)


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

    eta_discharge = battery.discharge_efficiency(usable_fraction)
    usable_energy = battery.energy_j * usable_fraction * eta_discharge

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



def intake_drag(edf: EDFSpec, s_ref: float, cd_intake_coeff: float = 0.06) -> float:
    """Estimate intake drag coefficient increment.

    Scoop lip intake drag: CD_intake ≈ cd_intake_coeff * A_intake / S_ref
    Typical cd_intake_coeff for scoop intake: 0.05 - 0.07.
    """
    a_intake = np.pi / 4 * edf.intake_diameter**2
    return cd_intake_coeff * a_intake / s_ref if s_ref > 0 else 0.0
