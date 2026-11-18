"""
Classical baselines for the model comparison (Prof. Wang comment 9.13).
=========================================================================
Adds three classical models to the six-model comparison (Table 3 of the paper) on the
SAME data, tubes, splits and test windows, at the same 100% / 50% / 20% training fractions:

  1. Random forest on handcrafted window features (45 statistics per window).
  2. Wiener-process RUL, pooled drift.
  3. Wiener-process RUL, covariate-dependent (log-linear) drift.

Wiener baseline (health-index approach, as in the PHM literature):
  - Health index h = life fraction (age / lifespan), predicted from the 45 handcrafted
    window features by ridge regression fitted on the training tubes.
  - Each training tube's predicted-h path is modelled as a Wiener process
    X(t) = mu t + sigma B(t). Pooled variant: one drift mu (pooled MLE over tubes).
    Covariate variant: log mu is a ridge-linear function of operating-stress covariates
    (window-mean anode temperature, tube voltage, tube current), the standard
    accelerated-degradation form; each training tube's drift is the OLS slope of its path.
  - RUL point prediction = mean first-passage time to the failure level D = 1,
    E[T] = (D - x) / mu, capped at the same 50,000-cycle RUL cap as every other model.

No baseline is tuned: all use fixed default hyper-parameters, like the deep models.
The script also checks that its tube selection / splits / scaling reproduce the original
CTLoader test windows exactly, so the new rows are directly comparable.

Usage: python scripts/classical_baselines.py [smoke]
Output: results/ct_tubes/classical_baselines_raw.csv + printed summary table
"""
import sys, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.preprocessing.ct_loader import CTLoader, FEATURE_COLS, RUL_CAP

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
SPD = 100                       # honest scans per day (RMSE reported in days)
SEQ, STRIDE, MAX_TUBES = 30, 100, 60
DATA = os.path.join(ROOT, "data", "synthetic", "ct_tubes_honest.csv")
FRACS = [1.0] if SMOKE else [1.0, 0.5, 0.2]
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
COV_IDX = (0, 2, 3)             # anode temperature, tube voltage, tube current
OUT = os.path.join(ROOT, "results", "ct_tubes")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Data: replicates CTLoader's selection / split / scaling, but keeps tube metadata
# ---------------------------------------------------------------------------
class MetaData:
    def __init__(self, csv_path, seq_len=SEQ, stride=STRIDE, max_tubes=MAX_TUBES,
                 rul_cap=RUL_CAP, val_fraction=0.15, test_fraction=0.15):
        self.csv_path, self.seq_len, self.stride = csv_path, seq_len, stride
        self.max_tubes, self.rul_cap = max_tubes, rul_cap
        self.val_fraction, self.test_fraction = val_fraction, test_fraction

    def load(self):
        counts = {}
        for chunk in pd.read_csv(self.csv_path, usecols=["tube_id", "scan_cycle"],
                                  chunksize=500_000):
            for tid, cnt in chunk.groupby("tube_id")["scan_cycle"].count().items():
                counts[tid] = counts.get(tid, 0) + cnt
        valid_ids = [t for t, c in counts.items() if c >= self.seq_len]
        if self.max_tubes and len(valid_ids) > self.max_tubes:
            np.random.seed(42)
            valid_ids = list(np.random.choice(valid_ids, self.max_tubes, replace=False))
        valid_set = set(valid_ids)
        usecols = ["tube_id", "scan_cycle", "rul"] + FEATURE_COLS
        chunks = []
        for chunk in pd.read_csv(self.csv_path, usecols=usecols, chunksize=500_000):
            f = chunk[chunk["tube_id"].isin(valid_set)]
            if len(f):
                chunks.append(f)
        df = pd.concat(chunks, ignore_index=True)
        df["rul_capped"] = df["rul"].clip(upper=self.rul_cap)
        self.df = df
        all_ids = df["tube_id"].unique()
        np.random.seed(42)
        np.random.shuffle(all_ids)
        n_test = max(1, int(len(all_ids) * self.test_fraction))
        n_val = max(1, int(len(all_ids) * self.val_fraction))
        self.test_ids = list(all_ids[:n_test])
        self.val_ids = list(all_ids[n_test:n_test + n_val])
        self.full_train_ids = list(all_ids[n_test + n_val:])
        print(f"  tubes: {len(self.full_train_ids)} train-pool / {len(self.val_ids)} val / "
              f"{len(self.test_ids)} test", flush=True)

    def _windows(self, sub):
        Xs, ys, tube, aend, Ls = [], [], [], [], []
        for uid, g in sub.groupby("tube_id", sort=False):
            g = g.sort_values("scan_cycle")
            feats = g[FEATURE_COLS].values
            rul = g["rul_capped"].values
            cyc = g["scan_cycle"].values
            T = len(feats)
            if T < self.seq_len:
                continue
            L = cyc.max()
            for t in range(0, T - self.seq_len + 1, self.stride):
                Xs.append(feats[t:t + self.seq_len])
                ys.append(rul[t + self.seq_len - 1])
                tube.append(uid)
                aend.append(cyc[t + self.seq_len - 1])
                Ls.append(L)
        return dict(X=np.array(Xs, dtype=np.float32), y=np.array(ys, dtype=np.float64),
                    tube=np.array(tube), a_end=np.array(aend, dtype=np.float64),
                    L=np.array(Ls, dtype=np.float64))

    def split(self, train_fraction):
        full = self.full_train_ids
        n_keep = len(full) if train_fraction >= 1.0 else max(1, int(len(full) * train_fraction))
        train_ids = full[:n_keep]
        tr_rows = self.df[self.df["tube_id"].isin(train_ids)]
        scaler = MinMaxScaler().fit(tr_rows[FEATURE_COLS].values)
        out = {}
        for name, ids in (("train", train_ids), ("test", self.test_ids)):
            sub = self.df[self.df["tube_id"].isin(ids)].copy()
            sub[FEATURE_COLS] = scaler.transform(sub[FEATURE_COLS].values)
            out[name] = self._windows(sub)
        return out


def verify_against_ctloader(splits, frac):
    """The new baselines must see exactly the windows the original comparison saw."""
    dls = CTLoader(csv_path=DATA, seq_len=SEQ, batch_size=32, max_tubes=MAX_TUBES,
                   stride=STRIDE, train_fraction=frac).prepare()
    Xo = np.concatenate([x.numpy() for x, _ in dls["test"]])
    yo = np.concatenate([y.numpy() for _, y in dls["test"]]) * RUL_CAP
    ok_shape = Xo.shape == splits["test"]["X"].shape
    ok_x = ok_shape and np.allclose(Xo, splits["test"]["X"], atol=1e-6)
    ok_y = ok_shape and np.allclose(yo, splits["test"]["y"], atol=0.05)
    n_tr_orig = len(dls["train"].dataset)
    ok_tr = n_tr_orig == len(splits["train"]["X"])
    print(f"  [verify frac={frac}] test windows {Xo.shape[0]} | X match={ok_x} "
          f"| y match={ok_y} | train windows match={ok_tr} ({n_tr_orig})", flush=True)
    if not (ok_x and ok_y and ok_tr):
        raise RuntimeError("MetaData does not reproduce CTLoader windows; baselines would not "
                           "be comparable. Aborting.")


# ---------------------------------------------------------------------------
# Handcrafted features and models
# ---------------------------------------------------------------------------
def handcrafted(X):
    """45 features per window: mean, std, min, max, slope of each of the 9 channels."""
    X = np.asarray(X, dtype=np.float64)
    T = X.shape[1]
    t = np.arange(T) - (T - 1) / 2.0
    slope = (X * t[None, :, None]).sum(1) / (t ** 2).sum()
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1), slope], axis=1)


def rmse_days(pred_cycles, true_cycles):
    return float(np.sqrt(np.mean((np.asarray(pred_cycles) - np.asarray(true_cycles)) ** 2)) / SPD)


class WienerRUL:
    def __init__(self, covariate_drift, ridge=1.0, D=1.0, cap=RUL_CAP):
        self.cov, self.ridge, self.D, self.cap = covariate_drift, ridge, D, cap

    def fit(self, tr):
        F = handcrafted(tr["X"])
        self.fsc = StandardScaler().fit(F)
        Fz = self.fsc.transform(F)
        h = tr["a_end"] / tr["L"]                     # life fraction (health index target)
        self.hi = Ridge(alpha=self.ridge).fit(Fz, h)
        hh = self.hi.predict(Fz)

        num = den = 0.0
        slopes, zbar, res_sq, res_n = [], [], 0.0, 0
        for uid in np.unique(tr["tube"]):
            idx = np.where(tr["tube"] == uid)[0]
            a, x = tr["a_end"][idx], hh[idx]
            if len(idx) < 3:
                continue
            num += x[-1] - x[0]
            den += a[-1] - a[0]
            slopes.append(np.polyfit(a, x, 1)[0])
            zbar.append(tr["X"][idx][:, :, COV_IDX].astype(np.float64).mean((0, 1)))
        self.mu_pooled = num / den
        # diffusion estimate from pooled increments (diagnostic; point RUL needs only mu)
        for uid in np.unique(tr["tube"]):
            idx = np.where(tr["tube"] == uid)[0]
            if len(idx) < 3:
                continue
            da = np.diff(tr["a_end"][idx]); dx = np.diff(hh[idx])
            res_sq += np.sum((dx - self.mu_pooled * da) ** 2 / da); res_n += len(da)
        self.sigma = float(np.sqrt(res_sq / max(res_n, 1)))
        if self.cov:
            logmu = np.log(np.clip(np.array(slopes), 1e-9, None))
            Z = np.array(zbar)
            self.zsc = StandardScaler().fit(Z)
            self.drift = Ridge(alpha=self.ridge).fit(self.zsc.transform(Z), logmu)
        return self

    def predict(self, X):
        hh = self.hi.predict(self.fsc.transform(handcrafted(X)))
        if self.cov:
            z = np.asarray(X, dtype=np.float64)[:, :, COV_IDX].mean(1)
            mu = np.exp(self.drift.predict(self.zsc.transform(z)))
        else:
            mu = np.full(len(hh), self.mu_pooled)
        rul = (self.D - hh) / np.clip(mu, 1e-12, None)
        return np.clip(rul, 0.0, self.cap), hh


def main():
    t0 = time.time()
    meta = MetaData(DATA)
    meta.load()
    rows = []
    for frac in FRACS:
        pct = int(frac * 100)
        sp = meta.split(frac)
        verify_against_ctloader(sp, frac)
        tr, te = sp["train"], sp["test"]
        Ftr, Fte = handcrafted(tr["X"]), handcrafted(te["X"])
        print(f"\n--- fraction {pct}%: {len(tr['X'])} train windows from "
              f"{len(np.unique(tr['tube']))} tubes, {len(te['X'])} test windows ---", flush=True)

        # Reference: the paper's existing flattened-window random forest (sanity check)
        rf0 = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        rf0.fit(tr["X"].reshape(len(tr["X"]), -1), tr["y"])
        r = rmse_days(rf0.predict(te["X"].reshape(len(te["X"]), -1)), te["y"])
        rows.append(dict(frac=frac, model="RF flattened (reference)", seed=-1, rmse_days=r))
        print(f"  RF flattened (reference, paper Table 3 = 8.62/20.49/42.61): {r:.2f} d", flush=True)

        for seed in SEEDS:
            rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=seed)
            rf.fit(Ftr, tr["y"])
            r = rmse_days(rf.predict(Fte), te["y"])
            rows.append(dict(frac=frac, model="RF + handcrafted features", seed=seed, rmse_days=r))
            print(f"  RF + handcrafted     seed {seed}: {r:.2f} d", flush=True)

        for name, cov in (("Wiener (pooled drift)", False), ("Wiener (covariate drift)", True)):
            w = WienerRUL(covariate_drift=cov).fit(tr)
            pred, hh = w.predict(te["X"])
            r = rmse_days(pred, te["y"])
            hi_r2 = 1 - np.sum((hh - te["a_end"] / te["L"]) ** 2) / np.sum(
                (te["a_end"] / te["L"] - (te["a_end"] / te["L"]).mean()) ** 2)
            rows.append(dict(frac=frac, model=name, seed=-1, rmse_days=r))
            print(f"  {name:<26}: {r:6.2f} d   (health-index R2 on test = {hi_r2:.3f}, "
                  f"pooled mu = {w.mu_pooled*SPD:.5f}/day, sigma = {w.sigma:.4f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "classical_baselines_raw.csv"), index=False)

    print("\n=== SUMMARY: RMSE (days), same test set as the paper's model comparison ===")
    print(f"{'model':<30}" + "".join(f"{int(f*100):>14}%" for f in FRACS))
    for model in ["RF flattened (reference)", "RF + handcrafted features",
                  "Wiener (pooled drift)", "Wiener (covariate drift)"]:
        line = f"{model:<30}"
        for f in FRACS:
            v = df[(df.model == model) & (df.frac == f)]["rmse_days"]
            line += f"{v.mean():>8.2f}+/-{(v.std() if len(v) > 1 else 0):<5.2f}"
        print(line)
    print(f"\nDONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
