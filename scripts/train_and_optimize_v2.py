"""Train surrogate v2 (10 targets) on regenerated database, then optimize.

Usage:
    python scripts/train_and_optimize_v2.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATABASE = Path("data/eval_database_v2.json")
SURROGATE_PATH = "models/surrogate_v2_ctrl"


def train_surrogate():
    """Train 10-target ensemble surrogate."""
    from src.surrogate.model import SurrogateModel

    print("=" * 60)
    print("PHASE 1: Train 10-target surrogate")
    print("=" * 60)

    print(f"Loading {DATABASE}...")
    with open(DATABASE) as f:
        db = json.load(f)
    X = np.array(db["X"])
    results = db["results"]

    # Check data quality
    n_ctrl = sum(1 for r in results if "Cmd01" in r and "c_ref" in r)
    print(f"  {len(results)} samples, {n_ctrl} with new c_ref primitives")

    print("\nTraining surrogate (5 ensemble)...")
    t0 = time.time()
    surr = SurrogateModel(n_ensemble=5)
    ok = surr.fit(X, results, min_samples=30)
    elapsed = time.time() - t0
    print(f"  Success: {ok}, time: {elapsed:.0f}s, arch: {surr.architecture_str}")

    if not ok:
        print("FAILED - not enough valid samples")
        return None

    surr.save(SURROGATE_PATH)
    print(f"  Saved to {SURROGATE_PATH}/")

    # Validation
    rng = np.random.default_rng(42)
    test_idx = rng.choice(len(X), 500)
    preds = surr.predict(X[test_idx])

    Y_rows = []
    for i in test_idx:
        r = results[i]
        row = [
            r.get("CL_0", 0), r.get("CL_alpha", 0),
            r.get("CM_0", 0), r.get("CM_alpha", 0),
            r.get("CD0_wing", 0.005), r.get("CD0_body", 0.003),
            r.get("Cn_beta", 0), r.get("Cl_beta", 0),
            r.get("CLd01", 0), r.get("Cmd01", 0),
        ]
        Y_rows.append(row)
    Y_true = np.array(Y_rows)

    print("\n  Validation (500 samples):")
    for j, key in enumerate(surr.target_keys):
        y_t, y_p = Y_true[:, j], preds[key]
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((y_p - y_t) ** 2))
        print(f"    {key:12s}: R2={r2:.4f}  RMSE={rmse:.5f}")

    return surr


def verify_cg_correction():
    """Verify CG correction coherence between surrogate and AVL."""
    from src.surrogate.model import SurrogateModel
    from src.surrogate.reconstruct import reconstruct_aero
    from src.aero.evaluator import AeroEvaluator
    from src.parameterization.design_variables import params_from_vector

    print("\n" + "=" * 60)
    print("PHASE 2: Verify CG correction coherence")
    print("=" * 60)

    surr = SurrogateModel.load(SURROGATE_PATH)
    best_x = np.load("output/best_x.npy")
    params = params_from_vector(best_x)

    # Surrogate path
    prim = surr.predict(best_x)
    r_surr = reconstruct_aero(prim, best_x)

    # AVL path (ground truth)
    evaluator = AeroEvaluator(use_cg=True)
    r_avl = evaluator.evaluate(params)

    print(f"\n  {'Metric':20s} {'Surrogate':>12s} {'AVL':>12s} {'Delta':>10s}")
    print("  " + "-" * 56)
    for key in ["static_margin", "L_over_D", "CM", "elevon_deflection"]:
        v_surr = r_surr.get(key, 0)
        v_avl = r_avl.get(key, 0)
        if key == "static_margin":
            print(f"  {key:20s} {v_surr*100:11.1f}% {v_avl*100:11.1f}% {(v_surr-v_avl)*100:9.1f}%")
        else:
            print(f"  {key:20s} {v_surr:12.2f} {v_avl:12.2f} {v_surr-v_avl:10.2f}")

    return abs(r_surr["static_margin"] - r_avl["static_margin"]) < 0.05  # within 5% MAC


def optimize():
    """Run surrogate-assisted optimization with CG + controls."""
    from src.optimization.problem import run_surrogate_assisted
    from src.aero.mission import MissionCondition

    print("\n" + "=" * 60)
    print("PHASE 3: Optimize with real CG + controls")
    print("=" * 60)

    result = run_surrogate_assisted(
        mission=MissionCondition(),
        surrogate_path=SURROGATE_PATH,
        n_restarts=5,
        n_infill=50,
        max_cycles=8,
        surrogate_de_maxiter=300,
        surrogate_de_popsize=60,
        convergence_tol=0.05,
        seed=42,
        verbose=True,
    )

    if result["best_result"]:
        r = result["best_result"]
        p = result["best_params"]
        print(f"\n{'='*60}")
        print(f"BEST FEASIBLE DESIGN:")
        print(f"  Span:          {2*p.half_span:.2f} m")
        print(f"  Wing area:     {r['S_ref']*10000:.0f} cm2")
        print(f"  AR:            {r['AR']:.1f}")
        print(f"  L/D:           {r['L_over_D']:.2f}")
        print(f"  SM:            {r['static_margin']*100:.1f}% MAC")
        print(f"  Elevon defl:   {r.get('elevon_deflection', 0):.1f} deg")
        print(f"  T/D:           {r['T_over_D']:.2f}")
        print(f"  Endurance:     {r['endurance_min']:.1f} min")
        print(f"  Struct mass:   {r['struct_mass']:.3f} kg")
        print(f"  Feasible:      {r['is_feasible']}")

        np.save("output/best_x_v2.npy", result["best_x"])
        print(f"\n  Saved: output/best_x_v2.npy")
    else:
        print("\nNo feasible design found.")
        print(f"  Total AVL evals: {result['total_vlm_evals']}")
        print(f"  Database size: {len(result['database'])}")

    return result


if __name__ == "__main__":
    surr = train_surrogate()
    if surr is None:
        sys.exit(1)

    coherent = verify_cg_correction()
    if not coherent:
        print("\nWARNING: CG correction mismatch > 5% MAC. Proceeding anyway.")

    optimize()
