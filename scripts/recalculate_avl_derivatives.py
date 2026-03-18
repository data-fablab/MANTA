"""Recalculate CL_alpha, CM_alpha, Cn_beta using AVL direct stability derivatives.

Replaces the noisy finite-difference values from AeroSandbox VLM with
analytical derivatives from AVL's `st` command. Runs one AVL call per sample
at the stored trim alpha.

Usage:
    python scripts/recalculate_avl_derivatives.py [--workers N] [--batch-size N]

Outputs:
    data/eval_database_avl.json  — updated database with AVL derivatives
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parameterization.design_variables import params_from_vector
from src.parameterization.bwb_aircraft import build_airplane
from src.aero.avl_runner import run_avl_stability
from src.aero.mission import MissionCondition

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_IN = Path("data/eval_database.json")
DATABASE_OUT = Path("data/eval_database_avl.json")
CHECKPOINT = Path("data/_avl_checkpoint.json")
AVL_CMD = "d:/UAV/tools/avl/avl.exe"
MISSION = MissionCondition()


def process_sample(args):
    """Process a single sample: run AVL and return derivatives.

    Returns (index, dict_update) or (index, None) on failure.
    """
    idx, x_vec, alpha_eq = args

    try:
        params = params_from_vector(np.array(x_vec))
        airplane = build_airplane(params)

        res = run_avl_stability(
            airplane,
            alpha=alpha_eq,
            velocity=MISSION.velocity,
            avl_command=AVL_CMD,
            timeout=8.0,
            chordwise_res=6,
            spanwise_res=8,
        )

        if res is None or "CLa" not in res or "Cnb" not in res:
            return idx, None

        return idx, {
            "CL_alpha_avl": float(res["CLa"]),
            "CM_alpha_avl": float(res["Cma"]),
            "Cn_beta_avl": float(res["Cnb"]),
            "Cl_beta_avl": float(res.get("Clb", 0.0)),
            "CY_beta_avl": float(res.get("CYb", 0.0)),
        }

    except Exception:
        return idx, None


def main():
    parser = argparse.ArgumentParser(description="Recalculate derivatives with AVL")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Checkpoint every N samples (default: 500)")
    args = parser.parse_args()

    # --- Load database ---
    print(f"Loading {DATABASE_IN}...")
    with open(DATABASE_IN) as f:
        db = json.load(f)

    X = db["X"]  # list of lists
    results = db["results"]  # list of dicts
    n_total = len(results)
    print(f"  {n_total} samples loaded")

    # --- Load checkpoint if exists ---
    done_set = set()
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            ckpt = json.load(f)
        # Apply previous results
        for idx_str, update in ckpt.items():
            idx = int(idx_str)
            if update is not None:
                results[idx].update(update)
            done_set.add(idx)
        print(f"  Resuming from checkpoint: {len(done_set)} already done")

    # --- Prepare work items ---
    work = []
    for i in range(n_total):
        if i in done_set:
            continue
        r = results[i]
        alpha_eq = r.get("alpha_eq", 3.0)
        # Clip extreme alpha values
        alpha_eq = float(np.clip(alpha_eq, -5, 15))
        work.append((i, X[i], alpha_eq))

    n_remaining = len(work)
    print(f"  {n_remaining} samples to process")

    if n_remaining == 0:
        print("Nothing to do.")
        _finalize(db, results)
        return

    # --- Process ---
    t0 = time.time()
    n_done = 0
    n_ok = 0
    n_fail = 0
    checkpoint_data = {}

    # Load existing checkpoint data
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            checkpoint_data = json.load(f)

    if args.workers <= 1:
        # Single-threaded (simpler, easier to debug)
        for item in work:
            idx, update = process_sample(item)
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
                _save_checkpoint(checkpoint_data)
    else:
        # Multi-process
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
                    _save_checkpoint(checkpoint_data)

    # --- Save final checkpoint ---
    _save_checkpoint(checkpoint_data)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min: {n_ok} ok, {n_fail} failed")

    # --- Finalize: replace VLM derivatives with AVL where available ---
    _finalize(db, results)


def _save_checkpoint(data):
    """Save checkpoint to disk."""
    with open(CHECKPOINT, "w") as f:
        json.dump(data, f)


def _finalize(db, results):
    """Replace VLM derivatives with AVL values and save final database."""
    n_replaced = 0
    for r in results:
        if "Cn_beta_avl" in r:
            r["Cn_beta_vlm"] = r.get("Cn_beta", 0.0)  # backup
            r["Cn_beta"] = r["Cn_beta_avl"]
            n_replaced += 1
        if "CL_alpha_avl" in r:
            r["CL_alpha_vlm"] = r.get("CL_alpha", 0.0)
            r["CL_alpha"] = r["CL_alpha_avl"]
        if "CM_alpha_avl" in r:
            r["CM_alpha_vlm"] = r.get("CM_alpha", 0.0)
            r["CM_alpha"] = r["CM_alpha_avl"]

    db["results"] = results

    print(f"\nSaving {DATABASE_OUT} ({n_replaced} samples with AVL derivatives)...")
    with open(DATABASE_OUT, "w") as f:
        json.dump(db, f)
    print("Done.")

    # Cleanup checkpoint
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("Checkpoint cleaned up.")


if __name__ == "__main__":
    main()
