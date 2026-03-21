"""Avionics component specification and placement for BWB UAV.

Defines a detailed Bill of Materials (BOM) for all avionics components
with mass, dimensions, power consumption, and fuselage placement.
Replaces the placeholder 100g avionics mass with a component-level model.

BOM validated by FFAM review:
- ESC upgraded to 60A aero (EDF-compatible)
- BEC upgraded to 5A (servo peak margin)
- VTX antenna changed to omnidirectional (no masking on BWB)
- Glass inner ply added to body (anti-splinter)
- LED visibility lights added
"""

from dataclasses import dataclass, field
import numpy as np

from ..parameterization.design_variables import BWBParams
from ..parameterization.bwb_aircraft import build_body_airfoil


@dataclass
class Component:
    """A discrete avionics component with physical properties."""
    name: str
    model: str
    mass_g: float
    dimensions_mm: tuple[float, float, float]  # (L, W, H)
    position_xc: float       # placement as fraction of body_root_chord
    power_w: float = 0.0     # electrical consumption [W]
    price_eur: float = 0.0
    notes: str = ""


@dataclass
class AvionicsSpec:
    """Complete avionics BOM with mass, power, and placement data."""
    components: list[Component] = field(default_factory=list)

    @property
    def total_mass_g(self) -> float:
        return sum(c.mass_g for c in self.components)

    @property
    def total_mass_kg(self) -> float:
        return self.total_mass_g / 1000.0

    @property
    def total_power_w(self) -> float:
        return sum(c.power_w for c in self.components)

    @property
    def total_price_eur(self) -> float:
        return sum(c.price_eur for c in self.components)

    @classmethod
    def default_bwb(cls) -> 'AvionicsSpec':
        """Default BOM for a 1.5 kg BWB UAV with 70mm EDF."""
        return cls(components=[
            # --- Flight controller ---
            Component("FC", "Matek F405-WMN",
                      mass_g=10, dimensions_mm=(36, 36, 6),
                      position_xc=0.28, power_w=0.8, price_eur=25,
                      notes="iNav, gyro 3-axis, baro, OSD, 4 servo outputs"),

            # --- Radio ---
            Component("Receiver", "ELRS EP2 2.4G",
                      mass_g=2, dimensions_mm=(18, 11, 5),
                      position_xc=0.28, power_w=0.3, price_eur=15,
                      notes="ExpressLRS, low latency, 2+ km range"),
            Component("Antenna RC", "T-dipole 2.4G",
                      mass_g=2, dimensions_mm=(30, 5, 5),
                      position_xc=0.90, price_eur=2,
                      notes="PTFE tube through skin, aft position"),

            # --- Navigation sensors ---
            Component("GPS", "BN-880Q",
                      mass_g=10, dimensions_mm=(28, 28, 8),
                      position_xc=0.15, power_w=0.2, price_eur=12,
                      notes="GPS + compass, ceramic patch antenna up"),
            Component("Pitot", "MS4525DO + tube",
                      mass_g=5, dimensions_mm=(15, 15, 10),
                      position_xc=0.05, power_w=0.1, price_eur=15,
                      notes="Airspeed sensor, tube protruding 20mm from nose"),

            # --- Power ---
            Component("ESC", "60A aero (YGE class)",
                      mass_g=25, dimensions_mm=(45, 30, 8),
                      position_xc=0.35, price_eur=45,
                      notes="EDF-compatible, high KV support, 50-60A burst"),
            Component("BEC", "Matek UBEC 5V/5A",
                      mass_g=5, dimensions_mm=(30, 18, 6),
                      position_xc=0.30, price_eur=10,
                      notes="5A peak for 4 servos + avionics"),
            Component("Current sensor", "Matek FCHUB-A5",
                      mass_g=3, dimensions_mm=(36, 36, 3),
                      position_xc=0.33, price_eur=8,
                      notes="Battery monitoring, mAh counter, OSD voltage"),

            # --- Servos (4x for elevons + ailerons) ---
            Component("Servo elevon L", "Emax ES3004",
                      mass_g=17, dimensions_mm=(32, 12, 30),
                      position_xc=0.70, power_w=2.0, price_eur=8,
                      notes="17g metal gear, 3.5 kg.cm @ 6V"),
            Component("Servo elevon R", "Emax ES3004",
                      mass_g=17, dimensions_mm=(32, 12, 30),
                      position_xc=0.70, power_w=2.0, price_eur=8),
            Component("Servo aileron L", "Emax ES3004",
                      mass_g=17, dimensions_mm=(32, 12, 30),
                      position_xc=0.70, power_w=2.0, price_eur=8,
                      notes="Mounted in wing, pushrod to hinge"),
            Component("Servo aileron R", "Emax ES3004",
                      mass_g=17, dimensions_mm=(32, 12, 30),
                      position_xc=0.70, power_w=2.0, price_eur=8),

            # --- FPV ---
            Component("Camera FPV", "Caddx Ant Nano",
                      mass_g=4, dimensions_mm=(14, 14, 15),
                      position_xc=0.08, power_w=1.2, price_eur=20,
                      notes="Nose position, 10-15° down tilt, transparent window"),
            Component("VTX", "Rush Tiny Tank",
                      mass_g=5, dimensions_mm=(20, 20, 5),
                      position_xc=0.25, power_w=1.6, price_eur=25,
                      notes="400mW, ~2 km FPV range"),
            Component("Antenna VTX", "Pagoda omni 5.8G",
                      mass_g=3, dimensions_mm=(25, 25, 5),
                      position_xc=0.92, price_eur=8,
                      notes="Omni (FFAM: no masking on BWB), protrudes through skin"),

            # --- Safety & misc ---
            Component("Buzzer", "5V piezo",
                      mass_g=2, dimensions_mm=(12, 12, 6),
                      position_xc=0.30, price_eur=1,
                      notes="Lost model finder"),
            Component("LED tip L", "White 5V LED",
                      mass_g=1, dimensions_mm=(10, 5, 5),
                      position_xc=0.95, power_w=0.25, price_eur=1.5,
                      notes="Visibility (FFAM requirement)"),
            Component("LED tip R", "White 5V LED",
                      mass_g=1, dimensions_mm=(10, 5, 5),
                      position_xc=0.95, power_w=0.25, price_eur=1.5),

            # --- Wiring ---
            Component("Wiring", "22-26 AWG silicone",
                      mass_g=25, dimensions_mm=(0, 0, 0),
                      position_xc=0.35, price_eur=5,
                      notes="Distributed, ~1.5m total length"),
            Component("Connectors", "XT60 + JST + servo",
                      mass_g=8, dimensions_mm=(0, 0, 0),
                      position_xc=0.30, price_eur=3),
        ])

    def print_bom(self) -> None:
        """Print formatted Bill of Materials."""
        print(f"\n{'='*80}")
        print(f"  AVIONICS BILL OF MATERIALS")
        print(f"{'='*80}")
        print(f"  {'Component':<22s} {'Model':<22s} {'Mass':>6s} {'Power':>6s} "
              f"{'x/c':>5s} {'Price':>7s}")
        print(f"  {'-'*70}")
        for c in self.components:
            pwr = f"{c.power_w:.1f}W" if c.power_w > 0 else "—"
            print(f"  {c.name:<22s} {c.model:<22s} {c.mass_g:5.0f}g "
                  f"{pwr:>6s} {c.position_xc:5.2f} {c.price_eur:6.1f}€")
        print(f"  {'-'*70}")
        print(f"  {'TOTAL':<22s} {'':22s} {self.total_mass_g:5.0f}g "
              f"{self.total_power_w:5.1f}W {'':5s} {self.total_price_eur:6.1f}€")
        print()

    def compute_power_budget(self, bec_amps: float = 5.0,
                             bec_voltage: float = 5.0) -> dict:
        """Compute electrical power budget and BEC margin.

        Returns dict with total_power_w, bec_current_a, bec_margin_pct.
        """
        total_w = self.total_power_w
        current_a = total_w / bec_voltage
        margin_pct = (1.0 - current_a / bec_amps) * 100
        return {
            "total_power_w": total_w,
            "bec_current_a": round(current_a, 2),
            "bec_capacity_a": bec_amps,
            "bec_margin_pct": round(margin_pct, 1),
        }


def compute_component_positions_3d(
    params: BWBParams,
    avionics: AvionicsSpec | None = None,
) -> list[dict]:
    """Convert x/c placements to 3D positions and check envelope fit.

    Each component is placed at its x/c position along the body chord,
    centered laterally (y=0) and vertically at the body camber line.
    The bounding box is checked against the local body envelope.

    Returns list of dicts with: name, position_mm, bbox_mm, fits, clearance_mm.
    """
    if avionics is None:
        avionics = AvionicsSpec.default_bwb()

    p = params
    bc = p.body_root_chord  # body chord [m]

    # Build body airfoil for envelope check
    af = build_body_airfoil(
        tc=p.body_tc_root, camber=p.body_camber,
        reflex=p.body_reflex, le_droop=p.body_le_droop,
        n_pts=200,
    )
    coords = af.coordinates  # (N, 2) normalized [0,1]
    mid = np.argmin(coords[:, 0])
    upper_x = coords[:mid + 1, 0][::-1]
    upper_z = coords[:mid + 1, 1][::-1]
    lower_x = coords[mid:, 0]
    lower_z = coords[mid:, 1]

    results = []
    for c in avionics.components:
        xc = c.position_xc
        x_mm = xc * bc * 1000

        # Local body envelope at this x/c
        z_up = float(np.interp(xc, upper_x, upper_z)) * bc * 1000
        z_lo = float(np.interp(xc, lower_x, lower_z)) * bc * 1000
        z_center = (z_up + z_lo) / 2
        height_avail = z_up - z_lo

        L, W, H = c.dimensions_mm
        half_h = H / 2

        fits = H <= height_avail * 0.85  # 15% margin
        clearance = (height_avail - H) / 2

        results.append({
            "name": c.name,
            "model": c.model,
            "mass_g": c.mass_g,
            "position_mm": [round(x_mm, 1), 0.0, round(z_center, 1)],
            "bbox_mm": [
                round(x_mm - L / 2, 1), round(x_mm + L / 2, 1),
                round(-W / 2, 1), round(W / 2, 1),
                round(z_center - half_h, 1), round(z_center + half_h, 1),
            ],
            "envelope_mm": [round(z_lo, 1), round(z_up, 1)],
            "height_avail_mm": round(height_avail, 1),
            "fits": fits,
            "clearance_mm": round(clearance, 1),
        })

    return results


def print_placement_summary(
    params: BWBParams,
    avionics: AvionicsSpec | None = None,
) -> None:
    """Print component placement summary with fit status."""
    positions = compute_component_positions_3d(params, avionics)

    print(f"\n{'='*80}")
    print(f"  COMPONENT PLACEMENT (body chord = {params.body_root_chord*1000:.0f} mm)")
    print(f"{'='*80}")
    print(f"  {'Component':<22s} {'x[mm]':>7s} {'z[mm]':>7s} {'H[mm]':>6s} "
          f"{'Avail':>6s} {'Clr':>5s} {'Fit':>4s}")
    print(f"  {'-'*60}")

    all_fit = True
    for p in positions:
        L_W_H = [c for c in (avionics or AvionicsSpec.default_bwb()).components
                 if c.name == p["name"]][0].dimensions_mm
        h = L_W_H[2]
        status = "OK" if p["fits"] else "FAIL"
        if not p["fits"]:
            all_fit = False
        if h > 0:
            print(f"  {p['name']:<22s} {p['position_mm'][0]:7.1f} "
                  f"{p['position_mm'][2]:7.1f} {h:6.0f} "
                  f"{p['height_avail_mm']:6.0f} {p['clearance_mm']:5.0f} "
                  f"{status:>4s}")

    print(f"  {'-'*60}")
    print(f"  Overall: {'ALL FIT' if all_fit else 'SOME DO NOT FIT'}")
    print()
