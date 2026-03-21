"""Composite manufacturing plan for BWB UAV.

Defines layup schedules, material specifications, and manufacturing
procedures for a prepreg carbon + foam core construction.

FFAM review amendments applied:
- Glass 80g inner ply on body (anti-splinter)
- Aramide hinge + glass bridge + CA reinforcement
- LED visibility lights integrated
"""

from dataclasses import dataclass, field
import numpy as np

from ..parameterization.design_variables import BWBParams
from ..parameterization.bwb_aircraft import compute_wing_area


# ═══════════════════════════════════════════════════════════════════════════
# Material database
# ═══════════════════════════════════════════════════════════════════════════

MATERIALS = {
    "carbon_200gsm": {
        "description": "Carbon twill 2×2, 200 g/m²",
        "type": "fabric", "weight_gsm": 200, "thickness_mm": 0.25,
        "tensile_mpa": 600, "price_eur_m2": 25,
    },
    "carbon_ud_300gsm": {
        "description": "Carbon unidirectional, 300 g/m²",
        "type": "fabric", "weight_gsm": 300, "thickness_mm": 0.30,
        "tensile_mpa": 1200, "price_eur_m2": 35,
    },
    "glass_80gsm": {
        "description": "E-glass plain weave, 80 g/m²",
        "type": "fabric", "weight_gsm": 80, "thickness_mm": 0.08,
        "tensile_mpa": 200, "price_eur_m2": 5,
    },
    "aramide_50gsm": {
        "description": "Aramide tape, 50 g/m² (hinges)",
        "type": "fabric", "weight_gsm": 50, "thickness_mm": 0.05,
        "tensile_mpa": 400, "price_eur_m2": 15,
    },
    "foam_xps_2mm": {
        "description": "XPS foam core, 2 mm, 40 kg/m³",
        "type": "core", "weight_gsm": 80, "thickness_mm": 2.0,
        "density_kg_m3": 40, "price_eur_m2": 2,
    },
    "foam_xps_3mm": {
        "description": "XPS foam core, 3 mm, 40 kg/m³",
        "type": "core", "weight_gsm": 120, "thickness_mm": 3.0,
        "density_kg_m3": 40, "price_eur_m2": 3,
    },
    "foam_xps_5mm": {
        "description": "XPS foam core, 5 mm, 40 kg/m³",
        "type": "core", "weight_gsm": 200, "thickness_mm": 5.0,
        "density_kg_m3": 40, "price_eur_m2": 4,
    },
    "rohacell_2mm": {
        "description": "Rohacell 51 IG-F, 2 mm, 52 kg/m³",
        "type": "core", "weight_gsm": 104, "thickness_mm": 2.0,
        "density_kg_m3": 52, "price_eur_m2": 45,
    },
    "prepreg_epoxy": {
        "description": "Epoxy resin system (prepreg ratio 42%)",
        "type": "resin", "ratio": 0.42,
        "cure_temp_c": 120, "cure_time_min": 60,
        "ramp_c_per_min": 2.0, "price_eur_kg": 30,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Layup specification
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Ply:
    """A single ply in a composite layup."""
    material: str              # key in MATERIALS
    orientation_deg: float     # fiber orientation (0, ±45, 90)

    @property
    def thickness_mm(self) -> float:
        return MATERIALS[self.material]["thickness_mm"]

    @property
    def weight_gsm(self) -> float:
        return MATERIALS[self.material]["weight_gsm"]


@dataclass
class ZoneLayup:
    """Composite layup for a specific structural zone."""
    zone: str
    description: str
    plies: list[Ply]

    @property
    def total_thickness_mm(self) -> float:
        return sum(p.thickness_mm for p in self.plies)

    @property
    def areal_mass_gsm(self) -> float:
        """Total areal mass in g/m² (fibers + resin for fabrics)."""
        total = 0.0
        resin_ratio = MATERIALS["prepreg_epoxy"]["ratio"]
        for p in self.plies:
            mat = MATERIALS[p.material]
            if mat["type"] == "fabric":
                total += mat["weight_gsm"] / (1.0 - resin_ratio)
            else:  # core
                total += mat["weight_gsm"]
        return total

    def print_layup(self) -> None:
        """Print layup sequence."""
        print(f"  {self.zone} — {self.description}")
        print(f"  {'Ply':<4s} {'Material':<25s} {'Orient':>7s} {'t[mm]':>6s} "
              f"{'gsm':>5s}")
        for i, p in enumerate(self.plies, 1):
            orient = f"{p.orientation_deg:+.0f}°" if p.orientation_deg != 0 else "0°"
            print(f"  {i:<4d} {MATERIALS[p.material]['description']:<25s} "
                  f"{orient:>7s} {p.thickness_mm:6.2f} {p.weight_gsm:5.0f}")
        print(f"  {'':4s} {'TOTAL':<25s} {'':7s} {self.total_thickness_mm:6.2f} "
              f"{self.areal_mass_gsm:5.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# Manufacturing plan
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ManufacturingPlan:
    """Complete manufacturing plan with layups and material quantities."""
    layups: dict[str, ZoneLayup]
    params: BWBParams | None = None

    @classmethod
    def default_bwb(cls, params: BWBParams) -> 'ManufacturingPlan':
        """Default layup for a 1.5 kg BWB with moule femelle + prepreg."""
        layups = {
            "body_upper": ZoneLayup(
                "body_upper", "Body extrados (sandwich)",
                [Ply("carbon_200gsm", 0), Ply("carbon_200gsm", 45),
                 Ply("carbon_200gsm", -45), Ply("carbon_200gsm", 0),
                 Ply("foam_xps_3mm", 0),
                 Ply("glass_80gsm", 0)],  # FFAM: anti-splinter inner ply
            ),
            "body_lower": ZoneLayup(
                "body_lower", "Body intrados (sandwich)",
                [Ply("carbon_200gsm", 0), Ply("carbon_200gsm", 45),
                 Ply("carbon_200gsm", -45),
                 Ply("foam_xps_3mm", 0),
                 Ply("glass_80gsm", 0)],  # FFAM: anti-splinter inner ply
            ),
            "wing_upper": ZoneLayup(
                "wing_upper", "Wing extrados (monolithic)",
                [Ply("carbon_200gsm", 0), Ply("carbon_200gsm", 45),
                 Ply("carbon_200gsm", -45)],
            ),
            "wing_lower": ZoneLayup(
                "wing_lower", "Wing intrados (monolithic)",
                [Ply("carbon_200gsm", 0), Ply("carbon_200gsm", 45)],
            ),
            "spar_cap": ZoneLayup(
                "spar_cap", "Spar caps (unidirectional)",
                [Ply("carbon_ud_300gsm", 0), Ply("carbon_ud_300gsm", 0),
                 Ply("carbon_ud_300gsm", 0), Ply("carbon_ud_300gsm", 0)],
            ),
            "spar_web": ZoneLayup(
                "spar_web", "Spar web (foam + shear plies)",
                [Ply("carbon_200gsm", 45), Ply("foam_xps_5mm", 0),
                 Ply("carbon_200gsm", -45)],
            ),
            "leading_edge": ZoneLayup(
                "leading_edge", "Leading edge (impact resistant)",
                [Ply("carbon_200gsm", 45), Ply("carbon_200gsm", -45),
                 Ply("rohacell_2mm", 0)],
            ),
            "control_surface": ZoneLayup(
                "control_surface", "Elevons & ailerons (sandwich)",
                [Ply("carbon_200gsm", 0), Ply("carbon_200gsm", 45),
                 Ply("carbon_200gsm", -45), Ply("foam_xps_2mm", 0)],
            ),
            "hinge": ZoneLayup(
                "hinge", "Control surface hinge (FFAM: reinforced)",
                [Ply("aramide_50gsm", 0), Ply("glass_80gsm", 45)],
            ),
        }
        return cls(layups=layups, params=params)

    def estimate_composite_mass(self, params: BWBParams | None = None) -> dict:
        """Estimate structural mass from actual layup (replaces empirical formula).

        Returns dict with zone masses and total.
        """
        p = params or self.params
        if p is None:
            raise ValueError("params required")

        bw = p.body_halfwidth
        bc = p.body_root_chord
        outer_span = p.outer_half_span

        # Body wetted area (approx: half-perimeter × body width, one side)
        body_half_perim = bc * (1 + p.body_tc_root) * 1.05  # chord × (1+t/c) approx
        body_area_half = body_half_perim * bw  # one side (upper or lower)

        # Wing areas (per one side of one wing)
        root_chord = p.wing_root_chord
        tip_chord = p.tip_chord
        avg_chord = (root_chord + tip_chord) / 2

        # Leading edge strip: ~10% chord
        le_area = 0.10 * avg_chord * outer_span * 2  # both wings

        # Control surfaces: 25% chord × control span
        ctrl_span = outer_span * 0.85  # elevon + aileron coverage
        ctrl_chord = 0.25 * avg_chord
        ctrl_area_one_side = ctrl_chord * ctrl_span * 2  # both wings

        # Main wing skin = total wing planform - LE strip - control surfaces
        wing_planform_one = avg_chord * outer_span * 2  # both wings
        main_skin_one_side = wing_planform_one - le_area - ctrl_area_one_side

        # Spar: length × spar height × 2 (both wings)
        spar_height = root_chord * 0.12
        spar_cap_area = outer_span * 0.04 * 2  # 40mm wide caps, 2 wings
        spar_web_area = outer_span * spar_height * 2  # 2 wings

        zones = {
            "body_upper": body_area_half * self.layups["body_upper"].areal_mass_gsm / 1000,
            "body_lower": body_area_half * self.layups["body_lower"].areal_mass_gsm / 1000,
            "wing_skin_upper": main_skin_one_side * self.layups["wing_upper"].areal_mass_gsm / 1000,
            "wing_skin_lower": main_skin_one_side * self.layups["wing_lower"].areal_mass_gsm / 1000,
            "spar_caps": spar_cap_area * self.layups["spar_cap"].areal_mass_gsm / 1000,
            "spar_web": spar_web_area * self.layups["spar_web"].areal_mass_gsm / 1000,
            "leading_edge": le_area * self.layups["leading_edge"].areal_mass_gsm / 1000,
            "control_surfaces": ctrl_area_one_side * 2 * self.layups["control_surface"].areal_mass_gsm / 1000,
        }

        # Fittings, glue, finishing: +15%
        subtotal = sum(zones.values())
        zones["fittings_glue_finish"] = subtotal * 0.15
        zones["total_kg"] = subtotal * 1.15

        return zones

    def compute_materials_bom(self, params: BWBParams | None = None) -> list[dict]:
        """Compute shopping list: material quantities and prices.

        Returns list of {material, description, quantity_m2, price_eur}.
        """
        p = params or self.params
        if p is None:
            raise ValueError("params required")

        mass_data = self.estimate_composite_mass(p)

        # Estimate area per material from zone areas (rough)
        bw = p.body_halfwidth
        bc = p.body_root_chord
        outer_span = p.outer_half_span
        root_chord = p.wing_root_chord
        tip_chord = p.tip_chord

        body_area = np.pi * (bc * p.body_tc_root + bc) / 2 * bw * 2
        wing_area = (root_chord + tip_chord) / 2 * outer_span * 2 * 2

        # Count plies per material across all zones (weighted by area)
        material_areas: dict[str, float] = {}
        zone_areas = {
            "body_upper": body_area / 2,
            "body_lower": body_area / 2,
            "wing_upper": wing_area / 2,
            "wing_lower": wing_area / 2,
            "spar_cap": outer_span * 0.04 * 2,
            "spar_web": outer_span * root_chord * 0.12 * 2,
            "leading_edge": 0.10 * (root_chord + tip_chord) / 2 * outer_span * 2,
            "control_surface": 0.25 * (root_chord + tip_chord) / 2 * outer_span * 0.85 * 2 * 2,
            "hinge": outer_span * 0.85 * 0.03 * 2,  # 30mm wide hinge strip
        }

        for zone_key, layup in self.layups.items():
            area = zone_areas.get(zone_key, 0.01)
            for ply in layup.plies:
                material_areas[ply.material] = material_areas.get(ply.material, 0) + area

        # Add 20% waste margin
        bom = []
        for mat_key, area in sorted(material_areas.items()):
            mat = MATERIALS[mat_key]
            area_with_waste = area * 1.20
            price = area_with_waste * mat.get("price_eur_m2", 0)
            bom.append({
                "material": mat_key,
                "description": mat["description"],
                "quantity_m2": round(area_with_waste, 3),
                "price_eur": round(price, 1),
            })

        # Add resin
        total_fabric_area = sum(
            a for k, a in material_areas.items()
            if MATERIALS[k]["type"] == "fabric"
        )
        resin_kg = total_fabric_area * 0.15 * 1.20  # ~150g/m² resin, +20% waste
        bom.append({
            "material": "prepreg_epoxy",
            "description": MATERIALS["prepreg_epoxy"]["description"],
            "quantity_m2": 0,
            "quantity_kg": round(resin_kg, 2),
            "price_eur": round(resin_kg * MATERIALS["prepreg_epoxy"]["price_eur_kg"], 1),
        })

        return bom

    def print_layup_summary(self) -> None:
        """Print all zone layups."""
        print(f"\n{'='*80}")
        print(f"  COMPOSITE LAYUP PLAN")
        print(f"{'='*80}")
        for layup in self.layups.values():
            layup.print_layup()
            print()

    def print_materials_bom(self, params: BWBParams | None = None) -> None:
        """Print shopping list."""
        bom = self.compute_materials_bom(params)
        print(f"\n{'='*80}")
        print(f"  MATERIALS SHOPPING LIST (incl. 20% waste)")
        print(f"{'='*80}")
        print(f"  {'Material':<30s} {'Qty':>8s} {'Price':>8s}")
        print(f"  {'-'*50}")
        total_price = 0
        for item in bom:
            qty_str = f"{item['quantity_m2']:.2f} m²" if item['quantity_m2'] > 0 else f"{item.get('quantity_kg', 0):.2f} kg"
            print(f"  {item['description']:<30s} {qty_str:>8s} {item['price_eur']:7.1f}€")
            total_price += item['price_eur']
        print(f"  {'-'*50}")
        print(f"  {'TOTAL':<30s} {'':>8s} {total_price:7.1f}€")
        print()

    def print_mass_breakdown(self, params: BWBParams | None = None) -> None:
        """Print structural mass from layup vs empirical estimate."""
        from ..parameterization.bwb_aircraft import estimate_structural_mass

        p = params or self.params
        zones = self.estimate_composite_mass(p)
        empirical = estimate_structural_mass(p)

        print(f"\n{'='*80}")
        print(f"  STRUCTURAL MASS BREAKDOWN (from layup)")
        print(f"{'='*80}")
        for zone, mass_kg in zones.items():
            if zone == "total_kg":
                continue
            print(f"  {zone:<25s} {mass_kg*1000:7.0f} g")
        print(f"  {'-'*35}")
        print(f"  {'TOTAL (layup)':<25s} {zones['total_kg']*1000:7.0f} g")
        print(f"  {'TOTAL (empirical)':<25s} {empirical*1000:7.0f} g")
        delta = (zones['total_kg'] - empirical) / empirical * 100
        print(f"  Delta: {delta:+.0f}%")
        print()


MANUFACTURING_STEPS = """
MANUFACTURING PROCEDURE — Moule femelle + prépreg carbone
══════════════════════════════════════════════════════════

Phase 1 — Outillage (1-2 semaines)
  1. Exporter STEP OML depuis le pipeline (export_aircraft_step)
  2. Usiner moules femelles CNC (MDF 30mm ou tooling board Cibatool)
     - 2 demi-coques body (extrados/intrados), plan de joint au plan de symétrie
     - 2 demi-coques aile droite (extrados/intrados)
     - Miroir pour aile gauche (ou usiner 2 moules supplémentaires)
  3. Poncer moules (grain 400 → 800 → 1200)
  4. Bouche-porer (2 couches polyuréthane), poncer fin
  5. Cirer (3 couches de cire Frekote + 1 couche PVA)

Phase 2 — Découpe noyaux (2-3 jours)
  6. Découpe mousse XPS au fil chaud depuis profils CAD (.dat)
     - Noyaux body : 3 tranches (center, mid, blend)
     - Noyaux aile : 4 tranches par demi-aile (root, 25%, 50%, tip)
  7. Ponçage des noyaux aux gabarits (cales profilées)
  8. Découpe logements : batterie, FC, ESC, servos dans les noyaux

Phase 3 — Stratification (3-5 jours)
  9. Découpe prépreg carbone aux gabarits papier (du plan de drapage)
  10. Drapage extrados dans moule femelle (body) :
      - Gelcoat époxy (optionnel, meilleur état de surface)
      - 4 plis carbone 200g [0/+45/-45/0]
      - Noyau mousse XPS 3mm
      - 1 pli verre 80g (FFAM: anti-éclats)
  11. Drapage extrados aile :
      - 3 plis carbone 200g [0/+45/-45]
  12. Drapage intrados (body + aile) selon plan de drapage
  13. Longeron :
      - Semelles : 4 plis carbone UD 300g [0°]
      - Âme : carbone ±45 + mousse XPS 5mm + carbone ±45
  14. Bord d'attaque : 2 plis carbone ±45 + rohacell 2mm
  15. Mise sous vide (sac nylon, tissu d'arrachage, feutre de drainage)
  16. Cuisson four : rampe 2°C/min → 120°C, maintien 60 min, refroidissement lent

Phase 4 — Assemblage (3-5 jours)
  17. Démoulage, détourage Dremel, ponçage des bords
  18. Collage longeron dans demi-coque extrados (époxy + micro-ballons)
  19. Passage câbles servos dans gaines pré-positionnées
  20. Fermeture : collage intrados sur extrados + longeron (époxy thixotrope)
  21. Joints congé époxy internes aux jonctions body/aile
  22. Découpe gouvernes (Dremel + guide aluminium depuis STEP)
  23. Charnières : ruban aramide 50g collé extrados + pont verre 80g + CA
      (FFAM: butée mécanique ±20° pour protéger les servos)

Phase 5 — Intégration avionique (2-3 jours)
  24. Hublot caméra FPV : découpe + collage vitre polycarbonate 1mm
  25. Tube Pitot : collage tube alu 3mm dans le nez, raccord silicone vers capteur
  26. Platine FC : impression 3D TPU, fixation M3 nylon + amortisseurs silicone
  27. Montage servos dans logements (vis M2 dans inserts collés)
  28. Câblage 22-26 AWG silicone + gaines thermo
  29. Antenne ELRS : tube PTFE à travers peau arrière
  30. Antenne VTX pagoda : montage arrière vertical
  31. GPS : sous capot (hublot ou zone sans carbone pour signal)
  32. LEDs visibilité : collage en bout d'aile, câblage vers BEC

Phase 6 — Essais sol (1-2 jours)
  33. Pesée globale et par composant → vérification CG (±5mm du calcul)
  34. Ajustement CG : position batterie ±20mm
  35. Configuration iNav :
      - Calibration accéléromètre + gyro
      - Calibration magnétomètre (loin des moteurs)
      - Calibration Pitot (souffler dans le tube)
      - Endpoints servos, sens de débattement, trims
      - Failsafe : motor off + spirale descendante (FFAM)
  36. Check radio : portée 200m, RSSI, failsafe trigger
  37. Check moteur : rotation, direction, réponse throttle progressive
  38. Check FPV : image, OSD, portée 100m

Phase 7 — Premiers vols (progressif, FFAM)
  39. Lancer plané (moteur off) : vérifier trim, CG, stabilité
  40. Vol moteur réduit (50%) : vérifier contrôle, montée
  41. Vol nominal : exploration enveloppe, vitesse, endurance
"""


def print_manufacturing_steps() -> None:
    """Print the complete manufacturing procedure."""
    print(MANUFACTURING_STEPS)
