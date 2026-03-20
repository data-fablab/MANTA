"""Regenerate ALL AVL-dependent primitives with consistent c_ref normalization.

Runs AVL at alpha=0 with control surfaces for each sample and extracts:
  CL_0, CL_alpha, CM_0, CM_alpha, Cn_beta, Cl_beta, CLd01, Cmd01

CD0_wing and CD0_body (analytical/NeuralFoil) are kept from the existing database
since they do not depend on AVL's reference chord normalization.

Usage:
    python scripts/regenerate_avl_primitives.py [--workers N] [--batch-size N]
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
from src.aero.avl_runner import run_avl_stability, ControlConfig
from src.aero.mission import MissionCondition

# ---------------------------------------------------------------------------
DATABASE_IN = Path("data/eval_database_avl.json")
DATABASE_OUT = Path("data/eval_database_v2.json")
CHECKPOINT = Path("data/_regen_checkpoint.json")
AVL_CMD = "d:/UAV/tools/avl/avl.exe"
MISSION = MissionCondition()
CONTROLS = ControlConfig.default_bwb()
CONTROLS.trim_elevon = False  # just derivatives, no trim


def process_sample(args):
    """Single AVL call at alpha=0 with controls -> all 8 AVL primitives."""
    idx, x_vec = args
    try:
        params = params_from_vector(np.array(x_vec))
        airplane = build_airplane(params)

        res = run_avl_stability(
            airplane, alpha=0.0,
            velocity=MISSION.velocity,
            avl_command=AVL_CMD,
            timeout=8.0,
            chordwise_res=6, spanwise_res=8,
            controls=CONTROLS,
        )
        if res is None or "CLa" not in res:
            return idx, None

        return idx, {
            "CL_0": float(res["CL"]),
            "CL_alpha": float(res["CLa"]),
            "CM_0": float(res["Cm"]),
            "CM_alpha": float(res["Cma"]),
            "Cn_beta": float(res.get("Cnb", 0.0)),
            "Cl_beta": float(res.get("Clb", 0.0)),
            "CLd01": float(res.get("CLd01", 0.0)),
            "Cmd01": float(res.get("Cmd01", 0.0)),
            # Also store aileron derivatives
            "Cld02": float(res.get("Cld02", 0.0)),
            "Cnd02": float(res.get("Cnd02", 0.0)),
            # Store c_ref for verification
            "c_ref": float(airplane.c_ref),
            "s_ref": float(airplane.s_ref),
        }
    except Exception:
        return idx, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading {DATABASE_IN}...")
    with open(DATABASE_IN) as f:
        db = json.load(f)

    X = db["X"]
    results = db["results"]
    n_total = len(results)
    print(f"  {n_total} samples")

    # Verify new c_ref
    p0 = params_from_vector(np.array(X[0]))
    a0 = build_airplane(p0)
    print(f"  New c_ref = {a0.c_ref:.4f}m  s_ref = {a0.s_ref:.4f}m2  b_ref = {a0.b_ref:.4f}m")

    # Load checkpoint
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
        print(f"  Checkpoint: {len(done_set)} done")

    # Prepare work
    work = []
    for i in range(n_total):
        if i not in done_set:
            work.append((i, X[i]))
    if args.max_samples > 0:
        work = work[:args.max_samples]

    n_remaining = len(work)
    print(f"  {n_remaining} to process")

    if n_remaining == 0:
        _finalize(db, results)
        return

    t0 = time.time()
    n_done = n_ok = n_fail = 0

    if args.workers <= 1:
        for item in work:
            idx, update = process_sample(item)
            _handle_result(idx, update, results, checkpoint_data)
            n_done += 1
            n_ok += 1 if update else 0
            n_fail += 0 if update else 1
            if n_done % 200 == 0:
                _progress(n_done, n_remaining, n_ok, n_fail, t0)
            if n_done % args.batch_size == 0:
                _save_checkpoint(checkpoint_data)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_sample, item): item[0] for item in work}
            for future in as_completed(futures):
                idx, update = future.result()
                _handle_result(idx, update, results, checkpoint_data)
                n_done += 1
                n_ok += 1 if update else 0
                n_fail += 0 if update else 1
                if n_done % 200 == 0:
                    _progress(n_done, n_remaining, n_ok, n_fail, t0)
                if n_done % args.batch_size == 0:
                    _save_checkpoint(checkpoint_data)

    _save_checkpoint(checkpoint_data)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min: {n_ok} ok, {n_fail} failed")
    _finalize(db, results)


def _handle_result(idx, update, results, checkpoint_data):
    if update is not None:
        results[idx].update(update)
        checkpoint_data[str(idx)] = update
    else:
        checkpoint_data[str(idx)] = None


def _progress(n_done, n_remaining, n_ok, n_fail, t0):
    elapsed = time.time() - t0
    rate = n_done / elapsed if elapsed > 0 else 0
    eta = (n_remaining - n_done) / rate if rate > 0 else 0
    print(f"  [{n_done}/{n_remaining}] ok={n_ok} fail={n_fail} "
          f"rate={rate:.1f}/s ETA={eta/60:.1f}min")


def _save_checkpoint(data):
    with open(CHECKPOINT, "w") as f:
        json.dump(data, f)


def _finalize(db, results):
    db["results"] = results

    # Stats
    n_ok = sum(1 for r in results if "Cmd01" in r and "c_ref" in r)
    if n_ok > 0:
        crefs = [r["c_ref"] for r in results if "c_ref" in r]
        cmd01s = [r["Cmd01"] for r in results if "Cmd01" in r]
        print(f"\nStats ({n_ok} samples):")
        print(f"  c_ref: [{min(crefs):.4f}, {max(crefs):.4f}]")
        print(f"  Cmd01: mean={np.mean(cmd01s):.5f} std={np.std(cmd01s):.5f}")

    print(f"\nSaving {DATABASE_OUT}...")
    with open(DATABASE_OUT, "w") as f:
        json.dump(db, f)
    print("Done.")

    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("Checkpoint cleaned up.")


if __name__ == "__main__":
    main()
