"""
Sensitivity of PINN-LSTM to the physics-loss weight lambda (Prof. Wang comment 9.12.2).
=========================================================================================
The paper uses lambda=0.3 with no stated justification. This reruns PINN-LSTM at
lambda in {0.0 (baseline LSTM), 0.1, 0.3 (paper value), 0.5, 1.0}, at full (100%) and
scarce (20%) training data, 3 seeds each, on the same honest 100 scans/day data used
for Table 2/3/4. Reports RMSE (days) and monotonicity violations (%, 20-window
separation) at each lambda, so the paper can state whether 0.3 is a sensitive choice.

Usage: python scripts/lambda_sensitivity.py [smoke]
Output: results/ct_tubes/lambda_sensitivity_raw.csv + printed summary table
"""
import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np, pandas as pd, torch, yaml

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
LAMBDAS = [0.3] if SMOKE else [0.0, 0.1, 0.3, 0.5, 1.0]
FRACS = [1.0] if SMOKE else [1.0, 0.2]
SEEDS = [0] if SMOKE else [0, 1, 2]
CKPT = os.path.join(ROOT, "results", "ct_tubes")
os.makedirs(CKPT, exist_ok=True)


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def monotonicity_violation(net, tpt, step):
    vv = tt = 0
    with torch.no_grad():
        for seqs in tpt:
            if len(seqs) < step + 1:
                continue
            pr = net(torch.FloatTensor(np.array(seqs))).numpy().flatten()
            d = pr[step:] - pr[:-step]
            vv += int((d > 0).sum()); tt += len(d)
    return vv / tt * 100 if tt else float("nan")


def main():
    rows, t0 = [], time.time()
    for frac in FRACS:
        pct = int(frac * 100)
        dls = CTLoader(csv_path=DATA, seq_len=SEQ, batch_size=32, max_tubes=60,
                        stride=100, train_fraction=frac).prepare()
        tpt = dls["test_per_tube"]
        for lam in LAMBDAS:
            cfg = copy.deepcopy(base_config)
            cfg["training"]["physics_loss_weight"] = lam
            dlc = tct.make_config(cfg, n_features=dls["n_features"], epochs=EPOCHS)
            for seed in SEEDS:
                set_seed(seed)
                use_pinn = lam > 0.0
                sp = os.path.join(CKPT, "lambda_tmp.pt")
                m = tct.run_deep(f"PINN-LSTM lam={lam}", tct.LSTMPredictor, use_pinn,
                                  dls, dlc, device, sp, scans_per_day=SPD)

                mcfg = dict(cfg["model"]); mcfg["input_features"] = dls["n_features"]
                mcfg.setdefault("output_size", 1)
                net = tct.LSTMPredictor({"model": mcfg})
                net.load_state_dict(torch.load(sp, map_location="cpu", weights_only=True))
                net.eval()
                viol20 = monotonicity_violation(net, tpt, 20)
                viol5 = monotonicity_violation(net, tpt, 5)

                rows.append(dict(frac=frac, lam=lam, seed=seed, rmse_days=m["rmse"],
                                  viol_sep5=viol5, viol_sep20=viol20))
                print(f"  [frac {pct}% lam={lam} seed {seed}] rmse={m['rmse']:.2f}d "
                      f"viol5={viol5:.1f}% viol20={viol20:.1f}%  "
                      f"{time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CKPT, "lambda_sensitivity_raw.csv"), index=False)

    print("\n=== LAMBDA SENSITIVITY SUMMARY (mean +/- std over seeds) ===")
    for frac in FRACS:
        print(f"\n-- {int(frac*100)}% data --")
        print(f"{'lambda':<8}{'RMSE (days)':<18}{'viol sep=5 (%)':<18}{'viol sep=20 (%)':<18}")
        for lam in LAMBDAS:
            sub = df[(df.frac == frac) & (df.lam == lam)]
            print(f"{lam:<8}{sub.rmse_days.mean():>6.2f}+/-{sub.rmse_days.std():<9.2f}"
                  f"{sub.viol_sep5.mean():>6.1f}+/-{sub.viol_sep5.std():<10.1f}"
                  f"{sub.viol_sep20.mean():>6.1f}+/-{sub.viol_sep20.std():<10.1f}")
    print(f"\nDONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
