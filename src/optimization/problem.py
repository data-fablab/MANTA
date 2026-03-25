"""BWB optimization pipeline.

- run_surrogate_assisted: mono-objectif (max L/D) with trust-region DE
- run_nsga2_assisted: multi-objectif (Pareto front) with NSGA-II + AVL validation
"""

import os
import time
import numpy as np
from scipy.optimize import differential_evolution
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
# Surrogate-assisted optimization (mono-objectif)
# ---------------------------------------------------------------------------

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
# Multi-objective: NSGA-II surrogate-assisted with AVL validation
# ---------------------------------------------------------------------------

AVAILABLE_OBJECTIVES = {
    "L_over_D":      {"key": "L_over_D",      "sense": "max", "label": "L/D"},
    "struct_mass":   {"key": "struct_mass",    "sense": "min", "label": "Structural mass [kg]"},
    "endurance_min": {"key": "endurance_min",  "sense": "max", "label": "Endurance [min]"},
    "manuf_score":   {"key": "manufacturability_score", "sense": "max", "label": "Manufacturability"},
    "range_km":      {"key": "range_km",       "sense": "max", "label": "Range [km]"},
    "drag_force":    {"key": "drag_force",     "sense": "min", "label": "Drag [N]"},
}


def run_nsga2_assisted(
    mission: MissionCondition | None = None,
    feasibility: FeasibilityConfig | None = None,
    surrogate_path: str = "models/surrogate_v2_ctrl",
    objectives: tuple[str, ...] = ("L_over_D", "struct_mass"),
    n_gen: int = 200,
    pop_size: int = 100,
    n_validate: int = 50,
    n_workers: int = 8,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Multi-objective NSGA-II on surrogate + AVL validation of Pareto front.

    Phase 1: NSGA-II on surrogate (fast, ~20k evals in seconds)
    Phase 2: Select top-K designs from Pareto front
    Phase 3: Validate with AVL (parallel)
    Phase 4: Build validated Pareto front + identify best per objective & knee

    Parameters
    ----------
    objectives : tuple of str
        Keys from AVAILABLE_OBJECTIVES (e.g. ("L_over_D", "struct_mass")).
    n_validate : int
        Number of Pareto designs to validate with AVL. 0 = skip validation.
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

    # Validate objectives
    obj_defs = []
    for obj_name in objectives:
        if obj_name not in AVAILABLE_OBJECTIVES:
            raise ValueError(f"Unknown objective '{obj_name}'. "
                             f"Available: {list(AVAILABLE_OBJECTIVES.keys())}")
        obj_defs.append(AVAILABLE_OBJECTIVES[obj_name])

    n_obj = len(obj_defs)
    obj_labels = [d["label"] for d in obj_defs]

    if verbose:
        print(f"=== NSGA-II Multi-Objective Optimization ===")
        print(f"Objectives ({n_obj}): {', '.join(obj_labels)}")
        print(f"Population: {pop_size}, Generations: {n_gen}")
        print(f"AVL validation: {n_validate} designs")
        print(f"Surrogate: {surrogate.architecture_str}, {surrogate.n_ensemble} ensemble")
        print(flush=True)

    # --- Phase 1: NSGA-II on surrogate ---
    _constr_keys = [
        "g_sm_lo", "g_sm_hi", "g_td", "g_endurance", "g_vs",
        "g_elevon_defl", "g_cn_beta", "g_ar", "g_volume",
        "g_manufacturability", "g_servo_torque",
        "g_bump", "g_dutch_roll",
    ]
    _n_constr = len(_constr_keys)

    class BWBProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=N_VARS, n_obj=n_obj, n_ieq_constr=_n_constr,
                xl=lb, xu=ub,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            prim = surrogate.predict(x)
            r = reconstruct_aero(prim, x, mission, feasibility)
            f = []
            for d in obj_defs:
                val = r[d["key"]]
                f.append(-val if d["sense"] == "max" else val)
            out["F"] = f
            # Individual constraints: g_i <= 0 means satisfied
            constraints = r.get("constraints", {})
            out["G"] = [constraints.get(k, 0.0) for k in _constr_keys]

    if verbose:
        print("Phase 1: Running NSGA-II on surrogate...", flush=True)

    t0 = time.time()
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    res = pymoo_minimize(
        BWBProblem(), algorithm,
        termination=("n_gen", n_gen),
        seed=seed, verbose=False,
    )
    t_surr = time.time() - t0

    pareto_X = res.X
    pareto_F = res.F
    n_pareto = len(pareto_X) if pareto_X is not None else 0

    if verbose:
        print(f"  Surrogate Pareto: {n_pareto} designs in {t_surr:.1f}s")
        if pareto_F is not None:
            for i, d in enumerate(obj_defs):
                vals = -pareto_F[:, i] if d["sense"] == "max" else pareto_F[:, i]
                print(f"    {d['label']:>25}: [{vals.min():.3f}, {vals.max():.3f}]")

    if n_pareto == 0:
        if verbose:
            print("  No Pareto solutions found!")
        return {"pareto_X": None, "pareto_F": None,
                "avl_X": None, "avl_F": None,
                "avl_results": [], "avl_feasible_mask": np.array([], dtype=bool),
                "best_per_objective": {},
                "objectives": objectives, "obj_defs": obj_defs,
                "n_pareto": 0, "n_feasible": 0}

    # --- Phase 2: Select top-K for AVL validation ---
    if n_validate <= 0:
        if verbose:
            print("Skipping AVL validation (n_validate=0)")
        return {"pareto_X": pareto_X, "pareto_F": pareto_F,
                "avl_results": [], "best_per_objective": {},
                "objectives": objectives, "obj_defs": obj_defs,
                "n_pareto": n_pareto, "n_feasible": 0, "pymoo_result": res}

    if n_validate >= n_pareto:
        validate_idx = np.arange(n_pareto)
    else:
        validate_idx = set()
        for i in range(n_obj):
            validate_idx.add(int(np.argmin(pareto_F[:, i])))
        remaining = n_validate - len(validate_idx)
        if remaining > 0:
            step = max(1, n_pareto // remaining)
            for j in range(0, n_pareto, step):
                validate_idx.add(j)
                if len(validate_idx) >= n_validate:
                    break
        validate_idx = np.array(sorted(validate_idx))

    pareto_sample = pareto_X[validate_idx]

    # Enrich with generate_candidates: combine features of best Pareto designs
    # to explore beyond the surrogate's Pareto front
    n_enriched = max(5, n_validate // 3)
    surr_feasible = np.array([
        reconstruct_aero(surrogate.predict(x), x, mission, feasibility)["penalty"] < 0.01
        for x in pareto_sample
    ])
    enriched = generate_candidates(
        top_k=pareto_sample,
        n_candidates=n_enriched,
        bounds=(lb, ub),
        rng=np.random.default_rng(seed + 1),
        feasible_mask=surr_feasible if surr_feasible.any() and not surr_feasible.all() else None,
    )
    X_validate = np.vstack([pareto_sample, enriched])
    n_val = len(X_validate)

    if verbose:
        print(f"\nPhase 2: Validating {len(pareto_sample)} Pareto + "
              f"{n_enriched} enriched = {n_val} designs with AVL...")

    # --- Phase 3: AVL validation (parallel) ---
    avl_results = _evaluate_batch(X_validate, mission, feasibility,
                                  verbose=verbose, n_workers=n_workers)

    # --- Phase 4: Build validated Pareto front ---
    avl_F = np.zeros((n_val, n_obj))
    for i, r in enumerate(avl_results):
        for j, d in enumerate(obj_defs):
            val = r.get(d["key"], 0.0)
            avl_F[i, j] = -val if d["sense"] == "max" else val

    feasible_mask = np.array([r.get("is_feasible", False) for r in avl_results])
    n_feasible = int(feasible_mask.sum())

    best_per_objective = {}
    if n_feasible > 0:
        feas_idx = np.where(feasible_mask)[0]
        for j, (obj_name, d) in enumerate(zip(objectives, obj_defs)):
            feas_vals = avl_F[feas_idx, j]
            best_local = feas_idx[np.argmin(feas_vals)]
            best_per_objective[obj_name] = {
                "x": X_validate[best_local],
                "result": avl_results[best_local],
                "index": int(best_local),
            }

        # Knee point: closest to utopia (normalized)
        feas_F = avl_F[feas_idx]
        f_min = feas_F.min(axis=0)
        f_max = feas_F.max(axis=0)
        f_range = np.maximum(f_max - f_min, 1e-10)
        f_norm = (feas_F - f_min) / f_range
        dist_to_utopia = np.linalg.norm(f_norm, axis=1)
        knee_local = feas_idx[np.argmin(dist_to_utopia)]
        best_per_objective["knee"] = {
            "x": X_validate[knee_local],
            "result": avl_results[knee_local],
            "index": int(knee_local),
        }

    if verbose:
        print(f"\n=== Results ===")
        print(f"Pareto (surrogate): {n_pareto} designs")
        print(f"Validated (AVL):    {n_val} designs")
        print(f"Feasible (AVL):     {n_feasible}")
        if n_feasible > 0:
            print(f"\nBest per objective (feasible):")
            for name, info in best_per_objective.items():
                r = info["result"]
                parts = []
                for obj_name, d in zip(objectives, obj_defs):
                    parts.append(f"{d['label']}={r.get(d['key'], 0):.3f}")
                print(f"  {name:>15}: {', '.join(parts)}")

    return {
        "pareto_X": pareto_X,
        "pareto_F": pareto_F,
        "validate_idx": validate_idx,
        "avl_X": X_validate,
        "avl_F": avl_F,
        "avl_results": avl_results,
        "avl_feasible_mask": feasible_mask,
        "best_per_objective": best_per_objective,
        "objectives": objectives,
        "obj_defs": obj_defs,
        "n_pareto": n_pareto,
        "n_feasible": n_feasible,
        "pymoo_result": res,
    }
