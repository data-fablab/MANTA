# nEUROn v2 — BWB Flying Wing Optimizer

Multi-disciplinary design optimization framework for a Blended Wing Body (BWB) flying wing UAV.
Combines parametric geometry, aerodynamic analysis (VLM), surrogate modeling (PyTorch), CAD export,
and multi-criteria design exploration to produce optimized aircraft designs ready for manufacturing.

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
        |
08_propulsion_integration   EDF duct placement, clearance, STEP booleans
        |
09_design_catalog           Multi-design trade-off analysis & batch export
```

## Project Structure

```
nEUROn_v2/
  src/
    parameterization/       30-variable BWB geometry (design_variables.py, bwb_aircraft.py)
    aero/                   Aerodynamic evaluation (AVL runner, drag model, mission)
    propulsion/             EDF motor model, duct geometry, duct aerodynamics
    surrogate/              PyTorch ensemble surrogate (model, features, reconstruct)
    optimization/           Problem formulation, candidates, database, design catalog
    evaluation/             Manufacturability scoring (geometric complexity metrics)
    visualization/          CAD export (STL, STEP), robust booleans, comparison plots
  notebooks/                Jupyter notebooks (pipeline steps 01-09)
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

Four export modes in `src/visualization/export.py`:

| Format | Function | Description |
|--------|----------|-------------|
| **STL** | `export_aircraft_stl()` | Triangulated mesh for visualization / 3D printing |
| **STEP v1** | `export_aircraft_step()` | Single BSpline surface (approximation) |
| **STEP v2** | `export_aircraft_step_v2()` | Watertight solid via ThruSections loft |
| **STEP v3** | `export_aircraft_step_v3()` | OML + propulsion duct with boolean integration |

STEP v2/v3 features:
- Exact interpolation through section wires (zero deviation)
- Periodic BSpline wires with native cosine spacing (0.05 mm fidelity)
- C2 continuity, chord-length parametrization
- Robust STEP booleans via `step_booleans.py`: ShapeFix healing +
  BOPAlgo_CellsBuilder with cascading fallbacks (preserves full NURBS quality)
- Validation: topology check, volume, surface area, max deviation

## Design Catalog

The design catalog (`src/optimization/catalog.py`) enables multi-criteria design exploration:

- **Named designs**: baseline (defaults), optimized (best_x.npy), Pareto-optimal, custom
- **α-blending**: interpolate between any two designs in the 30D space
- **Manufacturability scoring** (`src/evaluation/manufacturability.py`): composite score [0-1]
  based on twist/dihedral gradients, minimum thickness, taper severity, mold complexity
- **Comparison plots** (`src/visualization/comparison.py`): Pareto plot, radar chart,
  planform overlay, summary table
- **Batch STEP export**: export all catalog designs for downstream FEA/manufacturing pipeline

```python
from src.optimization.catalog import DesignCatalog

catalog = DesignCatalog()
catalog.add_baseline()
catalog.add_optimized('output/best_x.npy')
catalog.interpolate('baseline', 'optimized', [0.2, 0.4, 0.6, 0.8])
catalog.evaluate_aero(use_surrogate=True)
catalog.evaluate_manufacturability()
catalog.export_all_step('output/catalog/')
catalog.save('output/catalog.json')
```

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

Run notebooks in order (01 through 09). Each notebook documents its inputs, outputs,
and intermediate results. Notebook `07_export_cad` produces manufacturing-ready
CAD files, `08_propulsion_integration` adds the EDF duct, and `09_design_catalog`
enables multi-criteria trade-off analysis between designs.

```python
from src.visualization.export import export_aircraft_step_v2
from src.parameterization.design_variables import BWBParams, params_from_vector
import numpy as np

best_x = np.load('output/best_x.npy')
params = params_from_vector(best_x)
metrics = export_aircraft_step_v2(params, 'output/cad_export/neuron_v2_solid.step')
```
