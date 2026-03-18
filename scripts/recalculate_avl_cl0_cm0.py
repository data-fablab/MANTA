"""Recalculate CL_0 and CM_0 using AVL at alpha=0.

Adds consistent CL_0 and CM_0 from AVL to match the CL_alpha, CM_alpha,
Cn_beta already computed by AVL. This eliminates the solver mismatch
between AeroSandbox VLM (used for CL_0/CM_0) and AVL (used for derivatives).

Usage:
    python scripts/recalculate_avl_cl0_cm0.py [--workers N] [--batch-size N]

Reads:  data/eval_database_avl.json
Writes: data/eval_database_avl.json (in-place update)
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parameterization.design_variables import params_from_vector
from src.parameterization.bwb_aircraft import build_airplane
from src.aero.avl_runner import run_avl_stability
from src.aero.mission import MissionCondition

DATABASE = Path("data/eval_database_avl.json")
CHECKPOINT = Path("data/_avl_cl0_checkpoint.json")
AVL_CMD = "d:/UAV/tools/avl/avl.exe"
MISSION = MissionCondition()


def process_sample(args):
    """Run AVL at alpha=0 and return CL_0, CM_0."""
    idx, x_vec = args
    try:
        params = params_from_vector(np.array(x_vec))
        airplane = build_airplane(params)

        res = run_avl_stability(
            airplane,
            alpha=0.0,
            velocity=MISSION.velocity,
            avl_command=AVL_CMD,
            timeout=8.0,
            chordwise_res=6,
            spanwise_res=8,
        )

        if res is None or "CL" not in res or "Cm" not in res:
            return idx, None

        return idx, {
            "CL_0_avl": float(res["CL"]),
            "CM_0_avl": float(res["Cm"]),
        }

    except Exception:
        return idx, None


def main():
    parser = argparse.ArgumentParser(description="Recalculate CL_0/CM_0 with AVL")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    print(f"Loading {DATABASE}...")
    with open(DATABASE) as f:
        db = json.load(f)

    X = db["X"]
    results = db["results"]
    n_total = len(results)
    print(f"  {n_total} samples loaded")

    # Checkpoint
    done_set = set()
    checkpoint_data = {}
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            checkpoint_data = json.load(f)
        for idx_str, update in checkpoint_data.items():
            idx = int(idx_str)
            if update is not None:
                results[idx].update(update)
            done_set.add(idx)
        print(f"  Resuming: {len(done_set)} already done")

    # Already has CL_0_avl?
    already = sum(1 for r in results if "CL_0_avl" in r)
    if already > 0:
        print(f"  {already} samples already have CL_0_avl")

    work = [(i, X[i]) for i in range(n_total) if i not in done_set]
    n_remaining = len(work)
    print(f"  {n_remaining} samples to process")

    if n_remaining == 0:
        _finalize(db, results)
        return

    t0 = time.time()
    n_done = n_ok = n_fail = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sample, item): item[0]
                   for item in work}

        for future in as_completed(futures):
            idx, update = future.result()
            if update is not None:
                results[idx].update(update)
                checkpoint_data[str(idx)] = update
                n_ok += 1
            else:
                checkpoint_data[str(idx)] = None
                n_fail += 1
            n_done += 1

            if n_done % 100 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (n_remaining - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{n_remaining}] "
                      f"ok={n_ok} fail={n_fail} "
                      f"rate={rate:.1f}/s ETA={eta/60:.1f}min")

            if n_done % args.batch_size == 0:
                with open(CHECKPOINT, "w") as f:
                    json.dump(checkpoint_data, f)

    with open(CHECKPOINT, "w") as f:
        json.dump(checkpoint_data, f)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min: {n_ok} ok, {n_fail} failed")

    _finalize(db, results)


def _finalize(db, results):
    n_replaced = 0
    for r in results:
        if "CL_0_avl" in r:
            r["CL_0_vlm"] = r.get("CL_0", r.get("CL_required", 0.0) -
                                   r.get("CL_alpha_vlm", r.get("CL_alpha", 0.0)) *
                                   np.radians(r.get("alpha_eq", 0.0)))
            r["CL_0"] = r["CL_0_avl"]
            n_replaced += 1
        if "CM_0_avl" in r:
            r["CM_0_vlm"] = r.get("CM_0", r.get("CM", 0.0) -
                                   r.get("CM_alpha_vlm", r.get("CM_alpha", 0.0)) *
                                   np.radians(r.get("alpha_eq", 0.0)))
            r["CM_0"] = r["CM_0_avl"]

    db["results"] = results

    print(f"\nSaving {DATABASE} ({n_replaced} samples with AVL CL_0/CM_0)...")
    with open(DATABASE, "w") as f:
        json.dump(db, f)
    print("Done.")

    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("Checkpoint cleaned up.")


if __name__ == "__main__":
    main()
