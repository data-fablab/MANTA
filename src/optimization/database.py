"""
Persistent evaluation database for surrogate-assisted optimization.

Stores design vectors and their full evaluation results,
enabling restart and incremental surrogate training.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class EvaluationDatabase:
    """Cache of evaluated designs with save/load capability."""

    X: list = field(default_factory=list)         # design vectors
    results: list = field(default_factory=list)    # full evaluation result dicts

    def add(self, x: np.ndarray, result: dict):
        """Add a single evaluation."""
        self.X.append(x.copy() if isinstance(x, np.ndarray) else np.array(x))
        self.results.append(result)

    def add_batch(self, X: np.ndarray, results: list[dict]):
        """Add multiple evaluations."""
        for x, r in zip(X, results):
            self.add(x, r)

    def to_arrays(self) -> tuple[np.ndarray, list[dict]]:
        """Return (X_array, results_list)."""
        if not self.X:
            return np.empty((0, 0)), []
        return np.array(self.X), self.results

    def best_feasible(self) -> tuple[np.ndarray | None, dict | None]:
        """Return the best feasible design, or best penalized if none feasible."""
        best_ld = 0.0
        best_x, best_r = None, None

        # First try: best feasible
        for x, r in zip(self.X, self.results):
            if r.get("is_feasible", False) and r["L_over_D"] > best_ld:
                best_ld = r["L_over_D"]
                best_x, best_r = x, r

        if best_x is not None:
            return best_x, best_r

        # Fallback: lowest penalized score (best -L/D + penalty)
        best_score = np.inf
        for x, r in zip(self.X, self.results):
            ld = r.get("L_over_D", 0)
            penalty = r.get("penalty", 100)
            score = -ld + penalty * 10.0
            if score < best_score and ld > 0:
                best_score = score
                best_x, best_r = x, r

        return best_x, best_r

    def top_n(self, n: int, feasible_only: bool = False) -> tuple[np.ndarray, list[dict]]:
        """Return top-N designs by penalized objective (lowest = best).

        Score = -L/D + 10 * penalty (same formula as best_feasible).
        """
        if not self.X:
            return np.empty((0, 0)), []

        scores, indices = [], []
        for i, r in enumerate(self.results):
            if feasible_only and not r.get("is_feasible", False):
                continue
            ld = r.get("L_over_D", 0)
            if ld <= 0:
                continue
            scores.append(-ld + r.get("penalty", 100) * 10.0)
            indices.append(i)

        if not indices:
            return np.empty((0, 0)), []

        order = np.argsort(scores)[:n]
        sel = [indices[o] for o in order]
        return np.array([self.X[i] for i in sel]), [self.results[i] for i in sel]

    def save(self, path: str | Path):
        """Save database to JSON."""
        path = Path(path)
        X_list = [x.tolist() for x in self.X]
        # Convert numpy scalars in results for JSON serialization
        clean_results = []
        for r in self.results:
            clean = {}
            for k, v in r.items():
                if isinstance(v, (np.floating, np.integer)):
                    clean[k] = float(v)
                elif isinstance(v, np.bool_):
                    clean[k] = bool(v)
                elif isinstance(v, dict):
                    clean[k] = {
                        kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                        for kk, vv in v.items()
                    }
                else:
                    clean[k] = v
            clean_results.append(clean)

        data = {"X": X_list, "results": clean_results}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=1)

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationDatabase":
        """Load database from JSON."""
        with open(path) as f:
            data = json.load(f)
        db = cls()
        for x, r in zip(data["X"], data["results"]):
            db.add(np.array(x), r)
        return db

    def __len__(self):
        return len(self.X)
