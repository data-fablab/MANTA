"""Design catalog: named collection of BWB designs for multi-criteria comparison.

Stores multiple designs (baseline, optimized, Pareto, interpolated) with
their aerodynamic and manufacturability metrics, enabling trade-off analysis
before submitting to structural/manufacturing pipelines.
"""

import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from ..parameterization.design_variables import (
    BWBParams,
    params_from_vector,
    params_to_vector,
    N_VARS,
    BOUNDS,
    VAR_NAMES,
)


@dataclass
class CatalogEntry:
    """A single named design in the catalog."""

    name: str
    x: np.ndarray                           # (30,) design vector
    origin: str                             # "default", "optimization", "pareto",
                                            # "interpolation", "manual"
    aero_metrics: dict = field(default_factory=dict)
    manufacturing_metrics: dict = field(default_factory=dict)
    structural_metrics: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))

    @property
    def params(self) -> BWBParams:
        """Reconstruct BWBParams from design vector."""
        return params_from_vector(self.x)

    @property
    def L_over_D(self) -> float:
        return self.aero_metrics.get("L_over_D", 0.0)

    @property
    def is_feasible(self) -> bool:
        return self.aero_metrics.get("is_feasible", False)

    @property
    def manufacturability_score(self) -> float:
        return self.manufacturing_metrics.get("manufacturability_score", 0.0)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x": self.x.tolist(),
            "origin": self.origin,
            "aero_metrics": self.aero_metrics,
            "manufacturing_metrics": self.manufacturing_metrics,
            "structural_metrics": self.structural_metrics,
            "tags": self.tags,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CatalogEntry":
        return cls(
            name=d["name"],
            x=np.array(d["x"]),
            origin=d["origin"],
            aero_metrics=d.get("aero_metrics", {}),
            manufacturing_metrics=d.get("manufacturing_metrics", {}),
            structural_metrics=d.get("structural_metrics", {}),
            tags=d.get("tags", []),
            timestamp=d.get("timestamp", ""),
        )


class DesignCatalog:
    """Named collection of BWB designs for multi-criteria comparison."""

    def __init__(self):
        self.entries: dict[str, CatalogEntry] = {}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, name: str) -> CatalogEntry:
        return self.entries[name]

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def __iter__(self):
        return iter(self.entries.values())

    @property
    def names(self) -> list[str]:
        return list(self.entries.keys())

    # ── Adding designs ────────────────────────────────────────────────────

    def add(self, name: str, x: np.ndarray, origin: str,
            tags: Optional[list[str]] = None) -> CatalogEntry:
        """Add a design from a raw vector."""
        entry = CatalogEntry(
            name=name,
            x=x.copy(),
            origin=origin,
            tags=tags or [],
        )
        self.entries[name] = entry
        return entry

    def add_baseline(self, name: str = "baseline") -> CatalogEntry:
        """Add the default BWBParams as baseline reference."""
        x = params_to_vector(BWBParams())
        return self.add(name, x, origin="default", tags=["baseline", "reference"])

    def add_from_params(self, name: str, params: BWBParams,
                         origin: str = "manual") -> CatalogEntry:
        """Add from a BWBParams instance."""
        x = params_to_vector(params)
        return self.add(name, x, origin=origin)

    def add_optimized(self, path: str = "output/best_x.npy",
                       name: str = "optimized") -> CatalogEntry:
        """Load optimized design vector from .npy file."""
        x = np.load(path)
        return self.add(name, x, origin="optimization",
                        tags=["optimized", "reference"])

    # ── Interpolation (α-blending) ────────────────────────────────────────

    def interpolate(self, name_a: str, name_b: str,
                    alphas: list[float],
                    prefix: str = "blend") -> list[CatalogEntry]:
        """Generate interpolated designs between two catalog entries.

        x_blend = (1 - α) * x_a + α * x_b

        α=0 → pure A, α=1 → pure B.
        Intermediate values blend in the 30-dimensional design space.
        Results are clipped to variable bounds.
        """
        xa = self.entries[name_a].x
        xb = self.entries[name_b].x

        lb = np.array([BOUNDS[n][0] for n in VAR_NAMES])
        ub = np.array([BOUNDS[n][1] for n in VAR_NAMES])

        results = []
        for alpha in alphas:
            x_blend = (1 - alpha) * xa + alpha * xb
            x_blend = np.clip(x_blend, lb, ub)
            pct = int(round(alpha * 100))
            blend_name = f"{prefix}_{pct:02d}"
            entry = self.add(
                blend_name, x_blend, origin="interpolation",
                tags=["blend", f"alpha={alpha:.2f}"],
            )
            results.append(entry)

        return results

    # ── Pareto extraction from eval database ──────────────────────────────

    def add_from_database(self, db, strategy: str = "pareto",
                           n: int = 10,
                           objectives: tuple[str, str] = ("L_over_D", "struct_mass"),
                           prefix: str = "db") -> list[CatalogEntry]:
        """Extract designs from an EvaluationDatabase.

        Strategies:
          - "pareto": Non-dominated designs on two objectives
          - "top_n": Top N by L/D (feasible only)
          - "diverse": Top N with diversity filtering
        """
        if strategy == "pareto":
            return self._add_pareto(db, objectives, prefix)
        elif strategy == "top_n":
            X, results = db.top_n(n, feasible_only=True)
            entries = []
            for i, (x, r) in enumerate(zip(X, results)):
                name = f"{prefix}_top{i+1}"
                entry = self.add(name, x, origin="optimization",
                                 tags=["top_n", "feasible"])
                entry.aero_metrics = {
                    k: v for k, v in r.items()
                    if isinstance(v, (int, float, bool))
                }
                entries.append(entry)
            return entries
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _add_pareto(self, db, objectives, prefix) -> list[CatalogEntry]:
        """Extract Pareto-optimal designs from the database."""
        obj_a, obj_b = objectives
        X_arr, results = db.to_arrays()
        if len(results) == 0:
            return []

        # Build objective matrix (we want to maximize obj_a, minimize obj_b)
        vals_a = np.array([r.get(obj_a, 0.0) for r in results])
        vals_b = np.array([r.get(obj_b, np.inf) for r in results])

        # Filter to feasible designs only
        feasible_mask = np.array([r.get("is_feasible", False) for r in results])
        if not feasible_mask.any():
            feasible_mask = np.ones(len(results), dtype=bool)

        # Find non-dominated set (maximize a, minimize b)
        pareto_indices = []
        for i in range(len(results)):
            if not feasible_mask[i]:
                continue
            dominated = False
            for j in range(len(results)):
                if i == j or not feasible_mask[j]:
                    continue
                # j dominates i if j is better on both objectives
                if vals_a[j] >= vals_a[i] and vals_b[j] <= vals_b[i]:
                    if vals_a[j] > vals_a[i] or vals_b[j] < vals_b[i]:
                        dominated = True
                        break
            if not dominated:
                pareto_indices.append(i)

        # Sort by first objective
        pareto_indices.sort(key=lambda i: vals_a[i], reverse=True)

        entries = []
        for rank, idx in enumerate(pareto_indices):
            name = f"{prefix}_pareto{rank+1}"
            entry = self.add(name, X_arr[idx], origin="pareto",
                             tags=["pareto", "feasible"])
            entry.aero_metrics = {
                k: v for k, v in results[idx].items()
                if isinstance(v, (int, float, bool))
            }
            entries.append(entry)

        return entries

    # ── Batch evaluation ──────────────────────────────────────────────────

    def evaluate_aero(self, mission=None, feasibility=None,
                       use_surrogate: bool = False,
                       surrogate_path: str = "models/surrogate_v2",
                       verbose: bool = True):
        """Evaluate aerodynamics for all designs missing aero_metrics.

        Args:
            use_surrogate: If True, use the MLP surrogate (instant).
                           If False, use full AVL evaluation.
        """
        if use_surrogate:
            self._evaluate_aero_surrogate(mission, feasibility,
                                           surrogate_path, verbose)
        else:
            self._evaluate_aero_vlm(mission, feasibility, verbose)

    def _evaluate_aero_surrogate(self, mission, feasibility,
                                  surrogate_path, verbose):
        """Fast evaluation via surrogate model."""
        from ..aero.mission import MissionCondition, DEFAULT_FEASIBILITY
        from ..surrogate.model import SurrogateModel
        from ..surrogate.reconstruct import reconstruct_aero

        mission = mission or MissionCondition()
        feasibility = feasibility or DEFAULT_FEASIBILITY
        surrogate = SurrogateModel.load(surrogate_path)

        for entry in self.entries.values():
            if entry.aero_metrics:
                continue
            prim = surrogate.predict(entry.x)
            result = reconstruct_aero(prim, entry.x, mission, feasibility)
            # Convert numpy scalars
            entry.aero_metrics = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else
                   bool(v) if isinstance(v, np.bool_) else v
                for k, v in result.items()
                if isinstance(v, (int, float, bool, np.floating, np.integer, np.bool_))
            }
            if entry.is_feasible and "feasible" not in entry.tags:
                entry.tags.append("feasible")
            if verbose:
                print(f"  {entry.name:20s}  L/D={entry.L_over_D:6.2f}  "
                      f"{'OK' if entry.is_feasible else 'X'}")

    def _evaluate_aero_vlm(self, mission, feasibility, verbose):
        """Full VLM evaluation via AeroEvaluator."""
        from ..aero.evaluator import AeroEvaluator
        from ..aero.mission import MissionCondition, DEFAULT_FEASIBILITY

        mission = mission or MissionCondition()
        feasibility = feasibility or DEFAULT_FEASIBILITY
        evaluator = AeroEvaluator(mission, feasibility=feasibility)

        for entry in self.entries.values():
            if entry.aero_metrics:
                continue
            params = entry.params
            result = evaluator.evaluate(params)
            entry.aero_metrics = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else
                   bool(v) if isinstance(v, np.bool_) else v
                for k, v in result.items()
                if isinstance(v, (int, float, bool, np.floating, np.integer, np.bool_))
            }
            if entry.is_feasible and "feasible" not in entry.tags:
                entry.tags.append("feasible")
            if verbose:
                print(f"  {entry.name:20s}  L/D={entry.L_over_D:6.2f}  "
                      f"{'OK' if entry.is_feasible else 'X'}")

    def evaluate_manufacturability(self, verbose: bool = True):
        """Compute manufacturability metrics for all designs."""
        from ..evaluation.manufacturability import compute_manufacturability

        for entry in self.entries.values():
            metrics = compute_manufacturability(entry.params)
            entry.manufacturing_metrics = metrics
            if verbose:
                score = metrics.get("manufacturability_score", 0)
                print(f"  {entry.name:20s}  manuf={score:.3f}")

    # ── Batch STEP export ─────────────────────────────────────────────────

    def export_all_step(self, output_dir: str = "output/catalog",
                         version: str = "v2",
                         n_profile: int = 100,
                         verbose: bool = True) -> dict[str, str]:
        """Export all designs as STEP files.

        Returns dict of {design_name: step_file_path}.
        """
        from ..visualization.export import (
            export_aircraft_step_v2,
            export_aircraft_step_v3,
        )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}

        export_fn = export_aircraft_step_v2 if version == "v2" else export_aircraft_step_v3

        for entry in self.entries.values():
            step_path = str(out / f"{entry.name}.step")
            if verbose:
                print(f"  Exporting {entry.name}...", end=" ", flush=True)
            result = export_fn(entry.params, step_path, n_profile=n_profile)
            paths[entry.name] = step_path
            if verbose:
                print(f"{result['file_size_kb']:.0f} KB, valid={result['is_valid']}")

        return paths

    # ── DataFrame for analysis ────────────────────────────────────────────

    def to_dataframe(self):
        """Convert catalog to pandas DataFrame for comparison."""
        import pandas as pd

        rows = []
        for entry in self.entries.values():
            row = {
                "name": entry.name,
                "origin": entry.origin,
            }
            # Flatten aero metrics
            for k, v in entry.aero_metrics.items():
                if isinstance(v, (int, float, bool)):
                    row[k] = v
            # Flatten manufacturing metrics
            for k, v in entry.manufacturing_metrics.items():
                if isinstance(v, (int, float, bool)):
                    row[f"manuf_{k}"] = v
            # Key design variables
            p = entry.params
            row["half_span"] = p.half_span
            row["body_tc_root"] = p.body_tc_root
            row["taper_ratio"] = p.taper_ratio
            row["le_sweep_deg"] = p.le_sweep_deg
            rows.append(row)

        return pd.DataFrame(rows).set_index("name")

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str = "output/catalog.json"):
        """Save catalog to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n_designs": len(self.entries),
            "designs": {name: e.to_dict() for name, e in self.entries.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str = "output/catalog.json") -> "DesignCatalog":
        """Load catalog from JSON."""
        with open(path) as f:
            data = json.load(f)
        catalog = cls()
        for name, d in data["designs"].items():
            catalog.entries[name] = CatalogEntry.from_dict(d)
        return catalog

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Print a summary table."""
        lines = [f"DesignCatalog: {len(self)} designs", ""]
        header = f"{'Name':20s} {'Origin':14s} {'L/D':>6s} {'Feas':>5s} {'Manuf':>6s} {'Tags'}"
        lines.append(header)
        lines.append("-" * len(header))
        for e in self.entries.values():
            ld = f"{e.L_over_D:.2f}" if e.L_over_D else "  -  "
            feas = "OK" if e.is_feasible else ("X" if e.aero_metrics else " - ")
            ms = f"{e.manufacturability_score:.3f}" if e.manufacturing_metrics else "  -  "
            tags = ", ".join(e.tags[:3])
            lines.append(f"{e.name:20s} {e.origin:14s} {ld:>6s} {feas:>5s} {ms:>6s} {tags}")
        return "\n".join(lines)
