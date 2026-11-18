"""
Return-on-synthetic-data curve at 10 seeds, with significance tests (Prof. Wang 9.7).
=========================================================================================
The original curve (Fig. 3 / scarcity_curve.png) used 3 seeds per level. This reruns
PINN-LSTM (lambda=0.3) at the same 5 tube-count levels (2, 6, 14, 29, 42, via the same
train_fraction values CTLoader converts to those exact counts from the same 42-tube
trainable pool) with 10 seeds each, on the same honest 100 scans/day data. Reports
mean +/- SD, a 95% CI, and a Welch's t-test between each pair of adjacent levels to
show where the downward trend is statistically real versus where it plateaus into
noise.

Usage: python scripts/scarcity_seeds10.py [smoke]
Output: results/ct_tubes/scarcity_seeds10_raw.csv + printed summary
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np, pandas as pd, torch, yaml
from scipy import stats

import train_ct_tubes as tct
from src.preprocessing.ct_loader import CTLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
SPD = 100
DATA = os.path.join(ROOT, "data", "synthetic", "ct_tubes_honest.csv")
with open(os.path.join(ROOT, "configs", "config.yaml"), encoding="utf-8") as f:
    base_config = yaml.safe_load(f)
device = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = base_config["data"]["sequence_length"]
EPOCHS = 3 if SMOKE else 120
# train_fraction -> n_train_tubes (of the fixed 42-tube pool): matches the original
# 3-seed run exactly (2, 6, 14, 29, 42 tubes).
LEVELS = [(0.05, 2)] if SMOKE else [(0.05, 2), (0.15, 6), (0.35, 14), (0.7, 29), (1.0, 42)]
SEEDS = [0] if SMOKE else list(range(10))
CKPT = os.path.join(ROOT, "results", "ct_tubes")
os.makedirs(CKPT, exist_ok=True)
LAMBDA = 0.3


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    rows, t0 = [], time.time()
    for frac, n_tubes in LEVELS:
        dls = CTLoader(csv_path=DATA, seq_len=SEQ, batch_size=32, max_tubes=60,
                        stride=100, train_fraction=frac).prepare()
        cfg = dict(base_config)
        cfg["training"] = dict(base_config["training"])
        cfg["training"]["physics_loss_weight"] = LAMBDA
        dlc = tct.make_config(cfg, n_features=dls["n_features"], epochs=EPOCHS)
        for seed in SEEDS:
            set_seed(seed)
            sp = os.path.join(CKPT, "scarcity10_tmp.pt")
            m = tct.run_deep(f"PINN-LSTM n={n_tubes}", tct.LSTMPredictor, True,
                              dls, dlc, device, sp, scans_per_day=SPD)
            rows.append(dict(n_train_tubes=n_tubes, seed=seed, rmse_days=m["rmse"]))
            print(f"  [n={n_tubes} seed={seed}] rmse={m['rmse']:.2f}d  "
                  f"{time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CKPT, "scarcity_seeds10_raw.csv"), index=False)

    print("\n=== SUMMARY: RUL RMSE (days) vs training tube count, 10 seeds ===")
    print(f"{'n_tubes':<10}{'mean':<10}{'SD':<10}{'95% CI':<22}")
    levels_n = [n for _, n in LEVELS]
    group_vals = {}
    for n in levels_n:
        vals = df[df.n_train_tubes == n]["rmse_days"].values
        group_vals[n] = vals
        mean, sd = vals.mean(), vals.std(ddof=1)
        se = sd / np.sqrt(len(vals))
        ci = stats.t.interval(0.95, len(vals) - 1, loc=mean, scale=se) if len(vals) > 1 else (mean, mean)
        print(f"{n:<10}{mean:<10.2f}{sd:<10.2f}[{ci[0]:.2f}, {ci[1]:.2f}]")

    print("\n=== Welch's t-test, adjacent levels ===")
    for (n1), (n2) in zip(levels_n[:-1], levels_n[1:]):
        t, p = stats.ttest_ind(group_vals[n1], group_vals[n2], equal_var=False)
        sig = "significant (p<0.05)" if p < 0.05 else "NOT significant"
        print(f"  n={n1} vs n={n2}: t={t:.2f}, p={p:.4f}  [{sig}]")

    print(f"\nDONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
