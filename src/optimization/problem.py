"""BWB optimization using Differential Evolution.

Supports two methods:
- 'de': Single-objective DE with VLM evaluation (Neuronautics approach)
- 'surrogate': Surrogate-assisted DE -- MLP predicts 7 VLM primitives,
  analytical reconstruction for ranking, VLM validation on top candidates.
"""

import os
import time
import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor, as_completed

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
# Batch VLM evaluation (parallel + sequential fallback)
# ---------------------------------------------------------------------------

# Per-worker evaluator (initialized once per process via _init_worker)
_worker_evaluator = None


def _init_worker():
    """Initialize evaluator once per worker process (avoids re-import overhead)."""
    global _worker_evaluator
    from src.aero.evaluator import AeroEvaluator
    from src.config import load_all
    cfg = load_all()
    _worker_evaluator = AeroEvaluator(
        mission=cfg['mission'], feasibility=cfg['feasibility'],
        controls=cfg['controls'], cg_config=cfg['cg'],
        avl_command=cfg['avl_command'],
        aero_model=cfg['aero_model'], structure_config=cfg['structure'],
        use_cg=True,
    )


def _evaluate_single(x):
    """Worker: evaluate one design using the pre-initialized evaluator."""
    global _worker_evaluator
    if _worker_evaluator is None:
        _init_worker()
    try:
        params = params_from_vector(x)
        return _worker_evaluator.evaluate(params)
    except Exception as e:
        return {"L_over_D": 0.0, "penalty": 999.0, "is_feasible": False,
                "error": str(e)}


# Persistent pool (created once, reused across calls)
_pool = None
_pool_n_workers = 0


def _get_pool(n_workers):
    """Get or create a persistent ProcessPoolExecutor."""
    global _pool, _pool_n_workers
    if _pool is None or _pool_n_workers != n_workers:
        if _pool is not None:
            _pool.shutdown(wait=False)
        _pool = ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker,
        )
        _pool_n_workers = n_workers
    return _pool


def _evaluate_batch(
    X: np.ndarray,
    mission: MissionCondition,
    feasibility: FeasibilityConfig | None = None,
    verbose: bool = True,
    n_workers: int | None = None,
) -> list[dict]:
    """Evaluate a batch of designs, in parallel if n_workers > 1."""
    n_designs = len(X)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 8)

    if n_workers <= 1 or n_designs <= 2:
        if verbose:
            print(f"  [sequential mode: n_workers={n_workers}, n_designs={n_designs}]")
        return _evaluate_batch_sequential(X, mission, feasibility, verbose)

    # --- Parallel evaluation with persistent pool ---
    if verbose:
        print(f"  [parallel mode: {n_workers} workers, {n_designs} designs]")
    pool = _get_pool(n_workers)
    results = [None] * n_designs
    t0 = time.time()
    done = 0

    future_to_idx = {
        pool.submit(_evaluate_single, X[i]): i
        for i in range(n_designs)
    }
    for future in as_completed(future_to_idx):
        idx = future_to_idx[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            results[idx] = {"L_over_D": 0.0, "penalty": 999.0,
                            "is_feasible": False, "error": str(e)}
        done += 1
        if verbose and (done % 10 == 0 or done == n_designs):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            n_feas = sum(1 for r in results if r and r.get("is_feasible", False))
            print(f"  evaluated {done:4d}/{n_designs} "
                  f"({rate:.1f}/s, {n_feas} feasible)", flush=True)

    return results


def _evaluate_batch_sequential(
    X: np.ndarray,
    mission: MissionCondition,
    feasibility: FeasibilityConfig | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Evaluate a batch of designs sequentially (fallback)."""
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
            n_feas = sum(1 for r in results if r.get("is_feasible", False))
            print(f"  evaluated {i+1:4d}/{n_designs} "
                  f"({rate:.1f}/s, {n_feas} feasible)", flush=True)

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
    surrogate_path: str = "models/surrogate_v2_ctrl",
    n_restarts: int = 3,
    n_infill: int = 40,
    max_cycles: int = 5,
    surrogate_de_maxiter: int = 200,
    surrogate_de_popsize: int = 50,
    convergence_tol: float = 0.1,
    trust_region_shrink: float = 0.5,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Surrogate-assisted optimization with trust-region refinement.

    Cycle 1: global DE on surrogate over full design space.
    Cycles 2+: trust region centered on best feasible design,
    bounds shrink by trust_region_shrink each cycle.

    1. Load pre-trained surrogate
    2. Multi-restart DE on surrogate (fast)
    3. Rank top candidates by surrogate objective + uncertainty penalty
    4. Validate top candidates with AVL
    5. Shrink bounds around best design, repeat
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
    best_x_global = None
    current_lb = lb.copy()
    current_ub = ub.copy()

    for cycle in range(max_cycles):
        if verbose:
            print(f"\n--- Cycle {cycle+1}/{max_cycles} ---")

        # --- Trust region: shrink bounds around best design after Cycle 1 ---
        if cycle > 0 and best_x_global is not None:
            radius = (ub - lb) * trust_region_shrink ** cycle
            current_lb = np.maximum(lb, best_x_global - radius)
            current_ub = np.minimum(ub, best_x_global + radius)
            if verbose:
                pct = np.mean((current_ub - current_lb) / (ub - lb)) * 100
                print(f"  Trust region: {pct:.0f}% of global bounds "
                      f"(shrink={trust_region_shrink}^{cycle})")

        cycle_bounds = list(zip(current_lb, current_ub))

        # --- Phase 1: Multi-restart DE on surrogate (cheap) ---
        if verbose:
            print(f"  Running {n_restarts} DE restarts on surrogate "
                  f"(pop={surrogate_de_popsize}, iter={surrogate_de_maxiter})...")

        rng = np.random.default_rng(seed + cycle * 100)
        de_results = []

        for restart in range(n_restarts):
            surr_result = differential_evolution(
                _surrogate_objective_vectorized,
                bounds=cycle_bounds,
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
            sigma = 0.02 * (current_ub - current_lb)
            pts = opt + rng.normal(0, 1, size=(n_around, N_VARS)) * sigma
            perturbations.append(np.clip(pts, current_lb, current_ub))

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
            candidates, mission, feasibility=feasibility, verbose=verbose,
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
        if best_x is not None:
            best_x_global = best_x.copy()

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


# ---------------------------------------------------------------------------
# Multi-objective: NSGA-II via pymoo
# ---------------------------------------------------------------------------

def run_nsga2(
    mission: MissionCondition | None = None,
    feasibility: FeasibilityConfig | None = None,
    surrogate_path: str = "models/surrogate_v2_ctrl",
    n_gen: int = 200,
    pop_size: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Multi-objective optimization using NSGA-II on surrogate.

    Objectives: minimize [-L/D, struct_mass] (bi-objective)
    Constraints: all FeasibilityConfig constraints (penalty <= 0)

    Returns dict with 'pareto_X', 'pareto_F', 'pareto_results', 'pymoo_result'.
    """
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize as pymoo_minimize

    mission = mission or MissionCondition()
    feasibility = feasibility or DEFAULT_FEASIBILITY
    surrogate = SurrogateModel.load(surrogate_path)
    lb, ub = get_bounds_arrays()

    if verbose:
        print(f"=== NSGA-II Multi-Objective Optimization ===")
        print(f"Objectives: minimize [-L/D, struct_mass]")
        print(f"Constraints: penalty <= 0 (all feasibility constraints)")
        print(f"Population: {pop_size}, Generations: {n_gen}")

    class BWBProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=N_VARS,
                n_obj=2,
                n_ieq_constr=1,
                xl=lb,
                xu=ub,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            prim = surrogate.predict(x)
            r = reconstruct_aero(prim, x, mission, feasibility)

            # Objectives: minimize both
            out["F"] = [-r["L_over_D"], r["struct_mass"]]
            # Constraint: penalty <= 0 means feasible
            out["G"] = [r["penalty"] - 0.01]  # small tolerance

    problem = BWBProblem()

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    res = pymoo_minimize(
        problem, algorithm,
        termination=("n_gen", n_gen),
        seed=seed,
        verbose=verbose,
    )

    # Extract Pareto front
    pareto_X = res.X  # (n_pareto, 30)
    pareto_F = res.F  # (n_pareto, 2) = [-L/D, mass]

    # Reconstruct full results for Pareto designs
    pareto_results = []
    if pareto_X is not None:
        for x in pareto_X:
            prim = surrogate.predict(x)
            r = reconstruct_aero(prim, x, mission, feasibility)
            # Convert arrays to scalars
            r_scalar = {k: (float(v) if hasattr(v, '__len__') and len(v) > 0 else float(v))
                        for k, v in r.items()
                        if not isinstance(v, (bool, np.bool_))}
            r_scalar["is_feasible"] = bool(r.get("is_feasible", False))
            r_scalar["duct_fits"] = bool(r.get("duct_fits", False))
            pareto_results.append(r_scalar)

    n_feasible = sum(1 for r in pareto_results if r.get("is_feasible", False))

    if verbose and pareto_X is not None:
        print(f"\nPareto front: {len(pareto_X)} solutions ({n_feasible} feasible)")
        # Best L/D on Pareto front
        ld_vals = [-f[0] for f in pareto_F]
        mass_vals = [f[1] for f in pareto_F]
        print(f"  L/D range:  [{min(ld_vals):.2f}, {max(ld_vals):.2f}]")
        print(f"  Mass range: [{min(mass_vals)*1000:.0f}, {max(mass_vals)*1000:.0f}] g")

    return {
        "pareto_X": pareto_X,
        "pareto_F": pareto_F,
        "pareto_results": pareto_results,
        "pymoo_result": res,
        "n_pareto": len(pareto_X) if pareto_X is not None else 0,
        "n_feasible": n_feasible,
    }


# Default entry point
def run_optimization(
    mission: MissionCondition | None = None,
    feasibility: FeasibilityConfig | None = None,
    method: str = "de",
    **kwargs,
) -> dict:
    """Run optimization with specified method ('de', 'surrogate', or 'nsga2')."""
    if method == "de":
        return run_differential_evolution(mission=mission, feasibility=feasibility, **kwargs)
    elif method == "surrogate":
        return run_surrogate_assisted(mission=mission, feasibility=feasibility, **kwargs)
    elif method == "nsga2":
        return run_nsga2(mission=mission, feasibility=feasibility, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'de', 'surrogate', or 'nsga2'.")
