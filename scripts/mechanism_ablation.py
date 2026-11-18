"""
Mechanism ablation (Prof. Wang comment 9.10).
================================================
Checks that the simulator's dominant-mode story is driven by the mechanisms it is
supposed to be driven by, not an artefact: disable each of the four config-tunable
failure-triggering mechanisms (filament, bearing, vacuum/arcing, calendar age) one
at a time, at both CHU (100/day) and Tetouan (13/day) workloads, and report how the
batch mean lifespan and dominant failure mode shift relative to the full model.
If filament rupture is really what limits CHU's life, disabling it (and only it)
should noticeably lengthen CHU's simulated life while leaving Tetouan's roughly
alone; the mirror should hold for calendar age at Tetouan. Bearing and vacuum are
included for completeness even though they are not the dominant mode at either real
anchor, so disabling them is expected to barely move the mean.

Note: sustained overheating is NOT config-tunable (its threshold is hardcoded in
degradation_engine.py), so it is reported qualitatively rather than ablated
numerically; it has not been the dominant mode in any run so far.

Usage: python scripts/mechanism_ablation.py [smoke]
Output: results/mechanism_ablation.csv + printed table
"""
import sys, os, copy, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np, pandas as pd, yaml

from src.simulation.degradation_engine import DegradationEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
N = 10 if SMOKE else 150
HOSPITALS = {"CHU": 100, "Tetouan": 13}

ABLATIONS = {
    "full_model": {},
    "disable_filament": {("filament", "critical_diameter"): -1.0},
    "disable_bearing":  {("rotor", "vibration_failure_rms"): 1.0e6},
    "disable_vacuum":   {("vacuum", "failure_pressure"): 1.0e6},
    "disable_age":      {("aging", "characteristic_life_days"): 1.0e7},
}


def load_base_config():
    with open(os.path.join(ROOT, "configs", "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_batch(config, workload, n=N):
    cfg = copy.deepcopy(config)
    cfg["simulation"]["scans_per_day_mean"] = workload
    engine = DegradationEngine(cfg)
    lifespans, modes = [], []
    for i in range(n):
        rows = engine.simulate_single_tube(tube_id=i, seed=i * 42)
        if not rows:
            continue
        last = rows[-1]
        lifespans.append(last["scan_cycle"] / float(workload))
        modes.append(last["failure_mode"])
    return np.array(lifespans), pd.Series(modes)


def main():
    base_config = load_base_config()
    rows = []
    t0 = time.time()
    for hosp, workload in HOSPITALS.items():
        for ablation_name, overrides in ABLATIONS.items():
            cfg = copy.deepcopy(base_config)
            for (section, key), val in overrides.items():
                cfg["physics"][section][key] = val
            life, modes = run_batch(cfg, workload)
            mean_life = life.mean()
            mode_frac = modes.value_counts(normalize=True)
            top_mode = mode_frac.idxmax()
            rows.append(dict(
                hospital=hosp, ablation=ablation_name, mean_days=mean_life,
                dominant_mode=top_mode, dominant_mode_frac=mode_frac.max(),
            ))
            print(f"  [{hosp} workload={workload}] {ablation_name:<20} "
                  f"mean={mean_life:7.1f}d  top_mode={top_mode} "
                  f"({mode_frac.max()*100:.0f}%)  {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "mechanism_ablation.csv")
    df.to_csv(out, index=False)

    print("\n=== SUMMARY: lifespan shift (days) relative to full model ===")
    for hosp in HOSPITALS:
        base = df[(df.hospital == hosp) & (df.ablation == "full_model")]["mean_days"].values[0]
        print(f"\n-- {hosp} (full model = {base:.1f}d) --")
        for ablation_name in ABLATIONS:
            if ablation_name == "full_model":
                continue
            sub = df[(df.hospital == hosp) & (df.ablation == ablation_name)].iloc[0]
            delta = sub["mean_days"] - base
            print(f"  {ablation_name:<20} {sub['mean_days']:7.1f}d  "
                  f"({delta:+7.1f}d)  now dominated by {sub['dominant_mode']}")
    print(f"\nSaved -> {out}")
    print(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
