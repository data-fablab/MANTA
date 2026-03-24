# MANTA — BWB Flying Wing MDO Pipeline

Surrogate-assisted multidisciplinary design optimization for a Blended Wing Body (BWB) UAV with integrated EDF propulsion.

From parametric geometry to Pareto-optimal design in 10,000 evaluations.

## Pipeline

```
01_mission_mass_budget      Mission specs, EDF motor, avionics BOM, mass budget
02_parametric_geometry      32-variable BWB parametric model + duct fit
03_dataset_generation       19,500 LHS samples, AVL + NeuralFoil evaluation
04_surrogate                PyTorch ensemble surrogate (5 networks, 16 targets)
05_optimization             NSGA-II multi-objective (L/D vs mass vs endurance)
06_design_catalog           Pareto analysis, manufacturability scoring
07_validation               Full AVL re-evaluation of selected design
08_propulsion_visualization Duct sizing, clearance, aero performance, STL export
09_manufacturing            Composite layup plan, materials BOM
```

## Project Structure

```
src/
  parameterization/     32-variable BWB geometry (Kulfan CST, AeroSandbox)
  aero/                 AVL runner, NeuralFoil drag, stability, evaluator
  propulsion/           EDF model, duct sizing (physics-based), duct aero
  surrogate/            PyTorch ensemble MLP, feature engineering, reconstruct
  optimization/         NSGA-II, design catalog, evaluation database
  evaluation/           Manufacturability scoring
  visualization/        STL export, plots, comparison charts
  systems/              CG computation, avionics BOM
  manufacturing/        Composite layup planning
  config.py             YAML configuration loader
config/
  default.yaml          All pipeline parameters
notebooks/              Jupyter notebooks (01-09)
scripts/                Training + optimization automation
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

AVL must be on PATH for aerodynamic evaluation.
