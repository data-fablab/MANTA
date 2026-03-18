# nEUROn v2 — BWB Flying Wing Optimizer

Multi-disciplinary design optimization framework for a Blended Wing Body (BWB) flying wing UAV.
Combines parametric geometry, aerodynamic analysis (VLM), surrogate modeling (PyTorch), and CAD export
to produce an optimized aircraft design ready for manufacturing.

## Pipeline

```
01_mission_and_propulsion   Mission specs, EDF motor selection
        |
02_parametric_geometry      30-variable BWB parametric model
        |
03_dataset_generation       60k Latin Hypercube samples, AVL evaluation
        |
03bis_data_preparation      Data cleaning, feature engineering
        |
04_surrogate_training       5-output PyTorch ensemble surrogate
        |
04bis_cn_beta_study         Directional stability parameter study
        |
05_optimization             Design optimization via surrogate
        |
06_validation               Re-evaluation with full VLM (AVL)
        |
07_export_cad               STL, STEP, profiles + splines export
```

## Project Structure

```
nEUROn_v2/
  src/
    parameterization/       30-variable BWB geometry (design_variables.py, bwb_aircraft.py)
    aero/                   Aerodynamic evaluation (AVL runner, drag model, mission)
    propulsion/             EDF motor model (thrust, power, endurance)
    surrogate/              PyTorch ensemble surrogate (model, features, reconstruct)
    optimization/           Problem formulation, candidates, evaluation database
    visualization/          CAD export (STL, STEP, profiles) and plotting
  notebooks/                Jupyter notebooks (pipeline steps 01-07)
  scripts/                  Standalone AVL utilities
  requirements.txt          Python dependencies
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install cadquery             # for STEP export
```

AVL (Athena Vortex Lattice) must be available for aerodynamic evaluation.

## Design Variables

30 continuous variables organized in 4 groups:

| Group | Variables | Count |
|-------|-----------|:-----:|
| **Planform** | half_span, wing_root_chord, taper_ratio, le_sweep_deg | 4 |
| **Center body** | body_chord_ratio, body_halfwidth, body_tc_root, body_camber, body_reflex, body_twist, body_sweep_delta, body_le_droop | 8 |
| **Outer wing twist** | twist_1 .. twist_tip | 5 |
| **Outer wing dihedral** | dihedral_0 .. dihedral_tip | 5 |
| **Kulfan CST airfoil** | kulfan_root_{u2,u6,u7,l2,l6,l7}, kulfan_tip_delta_{tc,camber} | 8 |

## CAD Export

Three complementary export modes in `src/visualization/export.py`:

| Format | Function | Description |
|--------|----------|-------------|
| **STL** | `export_aircraft_stl()` | Triangulated mesh for visualization / 3D printing |
| **STEP v1** | `export_aircraft_step()` | Single BSpline surface (approximation) |
| **STEP v2** | `export_aircraft_step_v2()` | Watertight solid via ThruSections loft |

STEP v2 features:
- Exact interpolation through section wires (zero deviation)
- Periodic BSpline wires with native cosine spacing (0.05 mm fidelity)
- C2 continuity, chord-length parametrization
- Mirror + boolean fuse for full symmetric aircraft
- Validation: topology check, volume, surface area, max deviation

## Optimized Design

```
Geometry:
  Span              2.00 m
  Wing area          4976 cm2
  Aspect ratio       8.02
  Body chord          700 mm
  Taper ratio        0.28

Aerodynamics:
  L/D               22.87
  CL                 0.414
  CD                 0.0181
  Static margin      9.0% MAC

Structure:
  Structural mass     825 g
  Internal volume    2051 cm3

Propulsion (EDF 70mm):
  T/D                1.57
  Endurance          8.0 min
  Range              9.6 km
```

## Usage

Run notebooks in order (01 through 07). Each notebook documents its inputs, outputs,
and intermediate results. The final notebook (`07_export_cad`) produces manufacturing-ready
CAD files in `output/cad_export/`.

```python
from src.visualization.export import export_aircraft_step_v2
from src.parameterization.design_variables import BWBParams, params_from_vector
import numpy as np

best_x = np.load('output/best_x.npy')
params = params_from_vector(best_x)
metrics = export_aircraft_step_v2(params, 'output/cad_export/neuron_v2_solid.step')
```
