# nEUROn v2 — BWB Flying Wing Optimizer

Multi-disciplinary design optimization framework for a Blended Wing Body (BWB) flying wing UAV.
Combines parametric geometry, aerodynamic analysis (VLM + control surfaces), real CG computation,
surrogate modeling (PyTorch, 10 targets), CAD export, and multi-criteria design exploration.

## Pipeline

```
01_mission_and_propulsion   Mission specs, EDF motor selection, constraints
        |
02_parametric_geometry      30-variable BWB parametric model
        |
03_dataset_generation       60k Latin Hypercube samples, AVL evaluation
        |
04_data_preparation         Data cleaning, feature engineering
        |
05_surrogate_training       10-target PyTorch ensemble surrogate
        |
05bis_cn_beta_study         Directional stability parameter study (optional)
        |
06_optimization             Surrogate-assisted DE + NSGA-II multi-objective
        |
07_validation               Full AVL re-evaluation, geometry viz, CG, controls
        |
08_export_cad               STL, STEP v2/v3, profiles + splines
        |
09_propulsion_integration   EDF duct placement, clearance, STEP booleans
        |
10_design_catalog           Multi-design trade-off analysis & batch export
```

## Project Structure

```
nEUROn_v2/
  src/
    parameterization/       30-variable BWB geometry (design_variables.py, bwb_aircraft.py)
    aero/                   AVL runner (with elevon/aileron controls), drag, mission, dynamics
    propulsion/             EDF motor model, duct geometry, duct aerodynamics
    surrogate/              PyTorch ensemble surrogate (10 targets), reconstruct with CG correction
    optimization/           DE, NSGA-II, candidates, database, design catalog
    evaluation/             Manufacturability scoring (geometric complexity metrics)
    visualization/          CAD export (STL, STEP v1-v3), robust booleans, comparison plots
    systems/                CG computation, mass budget
    config.py               YAML configuration loader
  config/
    default.yaml            All pipeline parameters (mission, feasibility, CG, controls, AVL)
  notebooks/                Jupyter notebooks (pipeline steps 01-10)
  scripts/                  AVL regeneration, training + optimization
  requirements.txt          Python dependencies
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install cadquery             # for STEP export
```

AVL and XFOIL must be available for aerodynamic evaluation and profile validation.

## Design Variables

30 continuous variables organized in 5 groups:

| Group | Variables | Count |
|-------|-----------|:-----:|
| **Planform** | half_span, wing_root_chord, taper_ratio, le_sweep_deg | 4 |
| **Center body** | body_chord_ratio, body_halfwidth, body_tc_root, body_camber, body_reflex, body_twist, body_sweep_delta, body_le_droop | 8 |
| **Outer wing twist** | twist_1 .. twist_tip | 5 |
| **Outer wing dihedral** | dihedral_0 .. dihedral_tip | 5 |
| **Kulfan CST airfoil** | kulfan_root_{u2,u6,u7,l2,l6,l7}, kulfan_tip_delta_{tc,camber} | 8 |

## Surrogate Model (10 targets)

The ensemble MLP predicts 10 VLM primitives from the 30 design variables:

| Target | Description | R2 |
|--------|-------------|-----|
| CL_0, CL_alpha | Lift at alpha=0, lift curve slope | >0.99 |
| CM_0, CM_alpha | Pitching moment, longitudinal stability | >0.99 |
| CD0_wing, CD0_body | Parasite drag (NeuralFoil + analytical) | >0.99 |
| Cn_beta | Directional stability | 0.91 |
| **Cl_beta** | Roll due to sideslip (Dutch roll coupling) | **0.996** |
| **CLd01** | Elevon lift effectiveness | **0.999** |
| **Cmd01** | Elevon pitch moment effectiveness | **0.999** |

All derived quantities (L/D, static margin, trim, propulsion, CG, feasibility) are
reconstructed analytically in `src/surrogate/reconstruct.py`.

## Control Surfaces

Elevon + aileron layout defined in `src/aero/avl_runner.py`:
- **Elevon** (pitch): 25% chord, 0-50% outer wing span, symmetric deflection
- **Aileron** (roll): 25% chord, 55-90% outer wing span, antisymmetric deflection
- AVL trim: automatic elevon deflection to achieve CM=0

## CG Computation

Real center of gravity from component placement (`src/systems/cg.py`):
- Body structure, wing structure (shell + spar)
- Battery, EDF motor, avionics, servos
- Configurable positions via `config/default.yaml`
- SM correction: SM_real = SM_raw + (x_cg_old - x_cg_real) / c_ref

## CAD Export

| Format | Function | Description |
|--------|----------|-------------|
| **STL** | `export_aircraft_stl()` | Triangulated mesh |
| **STEP v2** | `export_aircraft_step_v2()` | Watertight solid via ThruSections loft |
| **STEP v3** | `export_aircraft_step_v3()` | OML + propulsion duct with boolean integration |

Robust STEP booleans via ShapeFix + BOPAlgo_CellsBuilder with cascading fallbacks.

## Optimized Design (v2)

```
Geometry:
  Span              1.35 m
  Wing area          2975 cm2
  Aspect ratio       6.1

Aerodynamics:
  L/D               11.57
  CL                 0.175
  CD                 0.0151

Stability & Control:
  Static margin      9.7% MAC (real CG)
  Elevon deflection  9.8 deg
  Cn_beta            0.018

Structure:
  Structural mass     508 g
  Internal volume    1200 cm3

Propulsion (EDF 70mm):
  T/D                3.14
  Endurance          16.0 min
  Range              19.2 km
```

## Usage

Run notebooks in order (01 through 10). Configuration via `config/default.yaml`.

```python
from src.config import load_all

cfg = load_all('config/default.yaml')
mission = cfg['mission']
feasibility = cfg['feasibility']

from src.optimization.problem import run_optimization
result = run_optimization(mission=mission, feasibility=feasibility, method='surrogate')
```
