"""BWB optimization using Differential Evolution.

Supports two methods:
- 'de': Single-objective DE with VLM evaluation (Neuronautics approach)
- 'surrogate': Surrogate-assisted DE -- MLP predicts 7 VLM primitives,
  analytical reconstruction for ranking, VLM validation on top candidates.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult
from scipy.stats import qmc

from ..parameterization.design_variables import (
    BWBParams,
    params_from_vector,
    get_bounds_arrays,
    get_scipy_bounds,
    N_VARS,
)
from ..aero.evaluator import AeroEvaluator
from ..aero.mission import MissionCondition, FeasibilityConfig, DEFAULT_FEASIBILITY
from ..surrogate.model import SurrogateModel
from ..surrogate.reconstruct import reconstruct_aero
from .candidates import generate_candidates
from .database import EvaluationDatabase


# ---------------------------------------------------------------------------
# Batch VLM evaluation (sequential)
# ---------------------------------------------------------------------------

def _evaluate_batch(
    X: np.ndarray,
    mission: MissionCondition,
    feasibility: FeasibilityConfig | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Evaluate a batch of designs sequentially with progress reporting."""
    evaluator = AeroEvaluator(mission, feasibility=feasibility)
    n_designs = len(X)
    results = []
    t0 = time.time()

    for i, x in enumerate(X):
        params = params_from_vector(x)
        result = evaluator.evaluate(params)
        results.append(result)
        if verbose and ((i + 1) % 20 == 0 or i == 0 or i == n_designs - 1):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_designs - i - 1) / rate if rate > 0 else 0
            n_feas = sum(1 for r in results if r.get("is_feasible", False))
            print(f"  evaluated {i+1:4d}/{n_designs} "
                  f"({rate:.1f}/s, ETA {eta:.0f}s, "
                  f"{n_feas} feasible)", flush=True)

    return results


# ---------------------------------------------------------------------------
# Single-objective: Differential Evolution (Neuronautics approach)
# ---------------------------------------------------------------------------

_eval_counter = [0]

def _objective_de(x: np.ndarray, evaluator: AeroEvaluator, history: list) -> float:
    """Objective for Differential Evolution: minimize -L/D + penalty."""
    params = params_from_vector(x)
    result = evaluator.evaluate(params)

    score = -result["L_over_D"] + result["penalty"] * 10.0

    history.append({
        "params": params,
        "result": result,
        "score": score,
    })

    _eval_counter[0] += 1
    n = _eval_counter[0]
    if n % 20 == 0 or n <= 3:
        ld = result["L_over_D"]
        feas = "OK" if result["is_feasible"] else "X"
        print(f"  eval #{n:4d}  L/D={ld:6.2f} [{feas}]", flush=True)

    return score


def run_differential_evolution(
    mission: MissionCondition | None = None,
    feasibility: FeasibilityConfig | None = None,
    maxiter: int = 100,
    popsize: int = 30,
    tol: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run Differential Evolution optimization (Neuronautics-style).

    Single objective: maximize L/D with penalty constraints.
    """
    mission = mission or MissionCondition()
    evaluator = AeroEvaluator(mission, feasibility=feasibility)
    history = []
    bounds = get_scipy_bounds()

    rng = np.random.default_rng(seed)
    lb, ub = get_bounds_arrays()
    init_pop = rng.uniform(lb, ub, size=(popsize, N_VARS))

    _eval_counter[0] = 0

    if verbose:
        print(f"Starting Differential Evolution: pop={popsize}, "
              f"maxiter={maxiter}, vars={N_VARS}")
        print(f"Mission: V={mission.velocity}m/s, MTOW={mission.mtow}kg, "
              f"alt={mission.altitude}m")
        print(f"Total evals estimate: ~{popsize * maxiter}")
        print(flush=True)

    gen_count = [0]
    best_ld = [0.0]

    def callback(xk, convergence):
        gen_count[0] += 1
        p = params_from_vector(xk)
        r = evaluator.evaluate(p)
        ld = r["L_over_D"]
        feasible = r["is_feasible"]
        if ld > best_ld[0] and feasible:
            best_ld[0] = ld
        if verbose:
            print(f"  Gen {gen_count[0]:3d}/{maxiter} | "
                  f"L/D={ld:.2f} {'OK' if feasible else 'X ':>2s} | "
                  f"best={best_ld[0]:.2f} | conv={convergence:.4f}",
                  flush=True)

    result: OptimizeResult = differential_evolution(
        _objective_de,
        bounds=bounds,
        args=(evaluator, history),
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        seed=seed,
        callback=callback,
        polish=False,
        init=init_pop,
        mutation=(0.5, 1.0),
        recombination=0.8,
    )

    best_params = params_from_vector(result.x)
    best_result = evaluator.evaluate(best_params)

    if verbose:
        print(f"\n=== Optimization Complete ===")
        print(f"Evaluations: {result.nfev}")
        print(f"Converged: {result.success} ({result.message})")
        print(f"\nBest design:")
        print(f"  L/D      = {best_result['L_over_D']:.2f}")
        print(f"  CL       = {best_result['CL']:.4f}")
        print(f"  CD       = {best_result['CD']:.5f} "
              f"(CD0={best_result['CD0']:.5f} + CDi={best_result['CDi']:.5f})")
        print(f"  CM       = {best_result['CM']:.4f}")
        print(f"  alpha    = {best_result['alpha_eq']:.2f} deg")
        print(f"  SM       = {best_result['static_margin']:.3f}")
        print(f"  AR       = {best_result['AR']:.1f}")
        print(f"  Oswald e = {best_result['oswald_e']:.3f}")
        print(f"  S_ref    = {best_result['S_ref']:.4f} m^2")
        print(f"  Mass     = {best_result['struct_mass']:.3f} kg")
        print(f"  Span     = {2*best_params.half_span:.2f} m")
        print(f"  Root c   = {best_params.root_chord:.3f} m")
        print(f"  Taper    = {best_params.taper_ratio:.3f}")
        print(f"  Sweep    = {best_params.le_sweep_deg:.1f} deg")
        print(f"  Feasible = {best_result['is_feasible']}")

    return {
        "best_params": best_params,
        "best_result": best_result,
        "scipy_result": result,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Surrogate-assisted optimization
# ---------------------------------------------------------------------------

def _surrogate_objective(
    x: np.ndarray,
    surrogate: SurrogateModel,
    feasibility: FeasibilityConfig,
    mission: MissionCondition,
) -> float:
    """Objective for DE on surrogate: predict primitives -> reconstruct -> penalize."""
    prim = surrogate.predict(x)
    result = reconstruct_aero(prim, x, mission, feasibility)
    return -result["L_over_D"] + result["penalty"] * 10.0


def _surrogate_objective_vectorized(
    X: np.ndarray,
    surrogate: SurrogateModel,
    feasibility: FeasibilityConfig,
    mission: MissionCondition,
) -> np.ndarray:
    """Vectorized objective: scipy passes (N_VARS, pop_size) -> (pop_size,)."""
    X = np.ascontiguousarray(X.T)  # (pop_size, N_VARS)
    prim = surrogate.predict(X)
    result = reconstruct_aero(prim, X, mission, feasibility)
    return -result["L_over_D"] + result["penalty"] * 10.0


def run_surrogate_assisted(
    mission: MissionCondition | None = None,
    feasibility: FeasibilityConfig | None = None,
    surrogate_path: str = "models/surrogate_v2",
    n_restarts: int = 3,
    n_infill: int = 40,
    max_cycles: int = 5,
    surrogate_de_maxiter: int = 200,
    surrogate_de_popsize: int = 50,
    convergence_tol: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Surrogate-assisted optimization using pre-trained MLP ensemble.

    Uses the pre-trained surrogate (60k samples) directly for cheap DE
    exploration, then validates the best candidates with AVL.

    1. Load pre-trained surrogate
    2. Multi-restart DE on surrogate (fast, ~100k evaluations)
    3. Rank top candidates by surrogate objective + uncertainty penalty
    4. Validate top candidates with AVL
    5. Inject validated points, retrain surrogate on augmented data, repeat
    """
    from pathlib import Path

    mission = mission or MissionCondition()
    feasibility = feasibility or DEFAULT_FEASIBILITY
    evaluator = AeroEvaluator(mission, feasibility=feasibility)
    db = EvaluationDatabase()
    lb, ub = get_bounds_arrays()
    bounds_tuple = (lb, ub)
    bounds = get_scipy_bounds()
    cycle_history = []
    total_vlm_evals = 0

    # --- Load pre-trained surrogate ---
    surrogate = SurrogateModel.load(surrogate_path)
    if verbose:
        print(f"=== Surrogate-Assisted Optimization ===")
        print(f"Design variables: {N_VARS}")
        print(f"Loaded pre-trained surrogate from {surrogate_path}")
        print(f"  Architecture: MLP {surrogate.architecture_str}, "
              f"{surrogate.n_ensemble} ensemble")
        print(flush=True)

    best_ld = 0.0

    for cycle in range(max_cycles):
        if verbose:
            print(f"\n--- Cycle {cycle+1}/{max_cycles} ---")

        # --- Phase 1: Multi-restart DE on surrogate (cheap) ---
        if verbose:
            print(f"  Running {n_restarts} DE restarts on surrogate "
                  f"(pop={surrogate_de_popsize}, iter={surrogate_de_maxiter})...")

        rng = np.random.default_rng(seed + cycle * 100)
        de_results = []

        for restart in range(n_restarts):
            surr_result = differential_evolution(
                _surrogate_objective_vectorized,
                bounds=bounds,
                args=(surrogate, feasibility, mission),
                strategy="best1bin",
                maxiter=surrogate_de_maxiter,
                popsize=surrogate_de_popsize,
                tol=1e-6,
                seed=seed + cycle * 100 + restart,
                polish=False,
                mutation=(0.5, 1.0),
                recombination=0.9,
                vectorized=True,
                updating="deferred",  # required for vectorized
            )
            de_results.append((surr_result.x, surr_result.fun))

        # Collect all DE optima + diverse pool around them
        de_optima = np.array([r[0] for r in de_results])
        n_around = 200  # perturbations around each DE optimum
        perturbations = []
        for opt in de_optima:
            sigma = 0.02 * (ub - lb)
            pts = opt + rng.normal(0, 1, size=(n_around, N_VARS)) * sigma
            perturbations.append(np.clip(pts, lb, ub))

        pool = np.vstack([de_optima] + perturbations)

        # --- Phase 2: Rank pool with surrogate + uncertainty penalty ---
        pool_prims = surrogate.predict(pool)
        pool_results = reconstruct_aero(pool_prims, pool, mission, feasibility)
        pool_obj = -pool_results["L_over_D"] + pool_results["penalty"] * 10.0

        # Add uncertainty penalty (designs where ensemble disagrees)
        if hasattr(surrogate, 'predict_with_uncertainty'):
            _, pool_std = surrogate.predict_with_uncertainty(pool)
            # Sum std across all targets as proxy for prediction uncertainty
            total_std = sum(pool_std[k] for k in pool_std)
            pool_obj = pool_obj + 2.0 * total_std  # penalize uncertain designs

        # Select top-K unique candidates for AVL validation
        top_idx = np.argsort(pool_obj)
        # De-duplicate: skip candidates too close to already-selected ones
        selected = []
        selected_X = []
        min_dist = 0.01 * np.linalg.norm(ub - lb)  # 1% of diagonal

        for idx in top_idx:
            if len(selected) >= n_infill:
                break
            x_cand = pool[idx]
            if selected_X:
                dists = np.linalg.norm(np.array(selected_X) - x_cand, axis=1)
                if dists.min() < min_dist:
                    continue
            selected.append(idx)
            selected_X.append(x_cand)

        candidates = pool[np.array(selected)]

        if verbose:
            best_surr_ld = pool_results["L_over_D"][selected[0]]
            best_surr_pen = pool_results["penalty"][selected[0]]
            print(f"  Surrogate best: L/D={best_surr_ld:.2f}, penalty={best_surr_pen:.2f}")
            print(f"  Validating {len(candidates)} candidates with AVL...")

        # --- Phase 3: Validate with AVL ---
        infill_results = _evaluate_batch(
            candidates, mission, feasibility=feasibility, verbose=False,
        )

        n_feas = 0
        for i, (x, result) in enumerate(zip(candidates, infill_results)):
            db.add(x, result)
            total_vlm_evals += 1
            is_feas = result.get("is_feasible", False)
            if is_feas:
                n_feas += 1

            if verbose:
                ld = result["L_over_D"]
                surr_ld = pool_results["L_over_D"][selected[i]]
                feas = "OK" if is_feas else "X"
                print(f"    infill {i+1:2d}/{len(candidates)}  "
                      f"L/D={ld:6.2f} (surr={surr_ld:.2f}) [{feas}]",
                      flush=True)

        # --- Phase 4: Convergence check ---
        prev_best_ld = best_ld
        best_x, best_r = db.best_feasible()
        best_ld = best_r["L_over_D"] if best_r else 0.0
        improvement = best_ld - prev_best_ld

        cycle_history.append({
            "cycle": cycle + 1,
            "best_ld": best_ld,
            "improvement": improvement,
            "n_feasible_infill": n_feas,
            "total_vlm_evals": total_vlm_evals,
            "db_size": len(db),
        })

        if verbose:
            print(f"  Feasible: {n_feas}/{len(candidates)}")
            print(f"  Best feasible L/D: {best_ld:.2f} "
                  f"({'+' if improvement >= 0 else ''}{improvement:.2f})")
            print(f"  Total AVL evals: {total_vlm_evals}")

        if improvement < convergence_tol and cycle >= 1:
            if verbose:
                print(f"  Converged (improvement {improvement:.3f} < {convergence_tol})")
            break

        # Note: we do NOT retrain the surrogate here. The pre-trained model
        # (60k samples) is far superior to anything we could train on the
        # handful of infill points. Instead, each cycle uses a different DE
        # seed to explore different basins of the surrogate landscape.

    # Final result
    best_x, best_r = db.best_feasible()
    best_params = params_from_vector(best_x) if best_x is not None else BWBParams()

    if verbose:
        print(f"\n=== Surrogate Optimization Complete ===")
        print(f"Total AVL evaluations: {total_vlm_evals}")
        print(f"Database size: {len(db)}")
        if best_r:
            print(f"Best feasible L/D: {best_r['L_over_D']:.2f}")
            print(f"  Span     = {2*best_params.half_span:.2f} m")
            print(f"  AR       = {best_r['AR']:.1f}")
            print(f"  SM       = {best_r['static_margin']:.3f}")
            print(f"  Mass     = {best_r['struct_mass']:.3f} kg")
            print(f"  Feasible = {best_r['is_feasible']}")

    return {
        "best_params": best_params,
        "best_result": best_r or {},
        "best_x": best_x,
        "database": db,
        "cycle_history": cycle_history,
        "total_vlm_evals": total_vlm_evals,
    }


# Default entry point
def run_optimization(
    mission: MissionCondition | None = None,
    feasibility: FeasibilityConfig | None = None,
    method: str = "de",
    **kwargs,
) -> dict:
    """Run optimization with specified method ('de' or 'surrogate')."""
    if method == "de":
        return run_differential_evolution(mission=mission, feasibility=feasibility, **kwargs)
    elif method == "surrogate":
        return run_surrogate_assisted(mission=mission, feasibility=feasibility, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'de' or 'surrogate'.")
