"""
Sensitivity of the CHU calibration to the assumed workload (Prof. Wang comment 8.2).
======================================================================================
The 100 scans/day figure used to calibrate the filament evaporation rate is an
estimate; comparable Moroccan hospitals measure 25-34/day. This script re-runs the
SAME calibration procedure described in the paper (manual bisection on
evaporation_rate_coeff until the N=150-tube batch mean lifespan matches the real
CHU target of 404 days, within 1%) at several assumed CHU workloads spanning that
range, and reports, for each: the recalibrated coefficient, the resulting CHU fit,
the dominant CHU failure mode, and what that same recalibrated simulator predicts
at Tetouan's workload (13/day) using the SAME, untouched, Tetouan-calibrated aging
parameters.

Usage: python scripts/workload_sensitivity.py
Output: results/workload_sensitivity.csv + printed table
"""
import sys, os, copy, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import pandas as pd
import yaml

from src.simulation.degradation_engine import DegradationEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BATCH = 150          # matches the paper's stated convergence batch size
TARGET_CHU_DAYS = 404.0
TETOUAN_WORKLOAD = 13
TOL = 0.01              # 1% convergence tolerance, matches the paper


def load_base_config():
    with open(os.path.join(ROOT, "configs", "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_batch(config, workload, n=N_BATCH, seed_offset=0):
    cfg = copy.deepcopy(config)
    cfg["simulation"]["scans_per_day_mean"] = workload
    engine = DegradationEngine(cfg)
    lifespans, modes = [], []
    for i in range(n):
        rows = engine.simulate_single_tube(tube_id=i, seed=seed_offset + i * 42)
        if not rows:
            continue
        last = rows[-1]
        lifespan_days = last["scan_cycle"] / float(workload)
        lifespans.append(lifespan_days)
        modes.append(last["failure_mode"])
    return np.array(lifespans), pd.Series(modes)


def calibrate_filament_rate(base_config, workload, target_days=TARGET_CHU_DAYS,
                             tol=TOL, max_iters=9, search_n=40, confirm_n=N_BATCH):
    """Bisection on evaporation_rate_coeff so the batch mean at `workload`
    matches `target_days` within `tol`. Rate is inversely related to lifespan.
    Searches with a smaller batch (search_n) for speed, then confirms the
    converged value with a full-size batch (confirm_n)."""
    lo, hi = 2.0e-11, 6.0e-10
    coeff = base_config["physics"]["filament"]["evaporation_rate_coeff"]
    for it in range(max_iters):
        cfg = copy.deepcopy(base_config)
        cfg["physics"]["filament"]["evaporation_rate_coeff"] = coeff
        life, modes = run_batch(cfg, workload, n=search_n)
        mean_life = life.mean()
        err = (mean_life - target_days) / target_days
        print(f"    iter {it}: coeff={coeff:.3e}  mean(n={search_n})={mean_life:.1f}d  err={err*100:+.1f}%", flush=True)
        if abs(err) <= tol:
            break
        if mean_life < target_days:
            hi = coeff   # too short -> rate too high -> lower it
        else:
            lo = coeff   # too long -> rate too low -> raise it
        coeff = (lo + hi) / 2.0

    # Final confirmation at full batch size
    cfg = copy.deepcopy(base_config)
    cfg["physics"]["filament"]["evaporation_rate_coeff"] = coeff
    life, modes = run_batch(cfg, workload, n=confirm_n)
    mean_life = life.mean()
    err = (mean_life - target_days) / target_days
    print(f"    CONFIRM n={confirm_n}: coeff={coeff:.3e}  mean={mean_life:.1f}d  err={err*100:+.1f}%", flush=True)
    dominant_mode = modes.value_counts(normalize=True)
    return coeff, mean_life, err, dominant_mode, cfg


def main():
    base_config = load_base_config()
    workloads = [25, 34, 50, 70, 100, 150]
    rows = []
    t0 = time.time()
    for w in workloads:
        print(f"\n=== Recalibrating for assumed CHU workload = {w}/day ===", flush=True)
        coeff, chu_mean, chu_err, chu_modes, cfg = calibrate_filament_rate(base_config, w)
        print(f"  CHU dominant mode(s): {dict(chu_modes.round(3))}")

        tet_life, tet_modes = run_batch(cfg, TETOUAN_WORKLOAD)
        tet_mean = tet_life.mean()
        tet_modes_frac = tet_modes.value_counts(normalize=True)
        print(f"  -> Tetouan @13/day with this coeff: {tet_mean:.1f}d "
              f"(real 854, {(tet_mean-854)/854*100:+.1f}%)  modes: {dict(tet_modes_frac.round(3))}")

        rows.append(dict(
            assumed_chu_workload=w,
            evaporation_rate_coeff=coeff,
            chu_mean_days=chu_mean,
            chu_pct_err=chu_err * 100,
            chu_dominant_mode=chu_modes.idxmax(),
            chu_dominant_mode_frac=chu_modes.max(),
            tetouan_mean_days=tet_mean,
            tetouan_pct_err=(tet_mean - 854) / 854 * 100,
            tetouan_dominant_mode=tet_modes_frac.idxmax(),
            tetouan_dominant_mode_frac=tet_modes_frac.max(),
        ))

    df = pd.DataFrame(rows)
    out_path = os.path.join(ROOT, "results", "workload_sensitivity.csv")
    df.to_csv(out_path, index=False)

    print("\n\n=== SUMMARY: calibration sensitivity to assumed CHU workload ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_path}")
    print(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
