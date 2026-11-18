"""
Rerun the six-model scarcity comparison (paper Table 2) + monotonicity (Table 3)
on the HONEST-calibrated data, at 100 cycles/day.
================================================================================
Why: the conference tables were computed at 300 cycles/day and on the old
calibration. This regenerates them consistently with the journal paper
(scans_per_day = 100, ct_tubes_honest.csv), reusing the tested functions in
scripts/train_ct_tubes.py.

Inputs:
  data/synthetic/ct_tubes_honest.csv   (60 honest-calibrated tubes @ ~100/day;
                                         regenerate with gen_honest_trajectories
                                         if missing)
Outputs:
  results/ct_tubes/honest_scarcity_raw.csv   (per model x fraction x seed RMSE)
  prints Table 2 (RMSE days, mean+/-std, 5 seeds) and Table 3 (monotonicity %)
  saves LSTM/PINN checkpoints (f100, f20, seed 0) for the monotonicity table

Usage:
  python scripts/rerun_honest_comparison.py          # full run (~1.5-2 h)
  python scripts/rerun_honest_comparison.py smoke     # 1 fraction, 1 seed, 3 epochs
"""
import sys, os, time
import numpy as np, pandas as pd, torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import train_ct_tubes as tct
from src.models.baselines import CNNLSTMPredictor, TransformerPredictor
from src.preprocessing.ct_loader import CTLoader

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
SPD = 100  # honest scans/cycles per day (matches configs/config.yaml)
DATA = os.path.join(ROOT, "data", "synthetic", "ct_tubes_honest.csv")
with open(os.path.join(ROOT, "configs", "config.yaml"), encoding="utf-8") as f:
    config = yaml.safe_load(f)
device = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = config["data"]["sequence_length"]
EPOCHS = 3 if SMOKE else 120
FRACS = [1.0] if SMOKE else [1.0, 0.5, 0.2]
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
CKPT = os.path.join(ROOT, "results", "ct_tubes"); os.makedirs(CKPT, exist_ok=True)


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    rows, t0, last_dls = [], time.time(), None
    for frac in FRACS:
        pct = int(frac * 100)
        dls = CTLoader(csv_path=DATA, seq_len=SEQ, batch_size=32, max_tubes=60,
                       stride=100, train_fraction=frac).prepare()
        last_dls = dls
        dlc = tct.make_config(config, n_features=dls["n_features"], epochs=EPOCHS)
        for lab, mk in [("SVR", lambda: SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")),
                        ("Random Forest", lambda: RandomForestRegressor(
                            n_estimators=100, max_depth=15, n_jobs=-1, random_state=42))]:
            m = tct.run_sklearn(lab, mk(), dls, scans_per_day=SPD)
            rows.append(dict(frac=frac, model=lab, seed=-1, rmse_days=m["rmse"]))
        for seed in SEEDS:
            set_seed(seed)
            for lab, cls, pinn in [("LSTM", tct.LSTMPredictor, False),
                                   ("CNN-LSTM", CNNLSTMPredictor, False),
                                   ("Transformer", TransformerPredictor, False),
                                   ("PINN-LSTM (ours)", tct.LSTMPredictor, True)]:
                if seed == 0 and frac in (1.0, 0.2) and lab in ("LSTM", "PINN-LSTM (ours)"):
                    tag = "f100" if frac == 1.0 else "f20"
                    name = "LSTM" if lab == "LSTM" else "PINN-LSTM_ours"
                    sp = os.path.join(CKPT, f"{name}_{tag}.pt")
                else:
                    sp = os.path.join(CKPT, "tmp.pt")
                m = tct.run_deep(lab, cls, pinn, dls, dlc, device, sp, scans_per_day=SPD)
                rows.append(dict(frac=frac, model=lab, seed=seed, rmse_days=m["rmse"]))
            print(f"  [frac {pct}% seed {seed} done  {time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CKPT, "honest_scarcity_raw.csv"), index=False)

    print("\n=== TABLE 2  RMSE (days), mean+/-std over seeds, 100 cycles/day ===")
    order = ["SVR", "Random Forest", "LSTM", "CNN-LSTM", "Transformer", "PINN-LSTM (ours)"]
    for model in order:
        line = f"{model:<20}"
        for frac in FRACS:
            sub = df[(df.model == model) & (df.frac == frac)]["rmse_days"]
            line += (f"  {sub.mean():6.2f}+/-{sub.std():.2f}" if len(sub) > 1 else f"  {sub.values[0]:6.2f}")
        print(line)

    tags = ["f100"] if SMOKE else ["f100", "f20"]
    if all(os.path.exists(os.path.join(CKPT, f"{n}_{t}.pt"))
           for n in ["LSTM", "PINN-LSTM_ours"] for t in tags):
        tpt = last_dls["test_per_tube"]
        mcfg = dict(config["model"]); mcfg["input_features"] = last_dls["n_features"]; mcfg.setdefault("output_size", 1)

        def load(p):
            net = tct.LSTMPredictor({"model": mcfg})
            net.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)); net.eval(); return net

        def viol(net, step):
            vv = tt = 0
            with torch.no_grad():
                for seqs in tpt:
                    if len(seqs) < step + 1:
                        continue
                    pr = net(torch.FloatTensor(np.array(seqs))).numpy().flatten()
                    d = pr[step:] - pr[:-step]; vv += int((d > 0).sum()); tt += len(d)
            return vv / tt * 100 if tt else float("nan")

        print("\n=== TABLE 3  monotonicity violations (%) ===")
        print("sep".ljust(6) + "".join(f"{'LSTM '+t:>12}{'PINN '+t:>12}" for t in tags))
        for step in [5, 10, 20]:
            line = f"{step:<6}"
            for t in tags:
                L = load(os.path.join(CKPT, f"LSTM_{t}.pt")); P = load(os.path.join(CKPT, f"PINN-LSTM_ours_{t}.pt"))
                line += f"{viol(L, step):>12.1f}{viol(P, step):>12.1f}"
            print(line)
    print(f"\nDONE  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
