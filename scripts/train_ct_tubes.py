"""
CT-Tube Main Experiment
========================
Step 1 — Generate 500 synthetic tube histories (physics simulation)
Step 2 — Validate degradation model against hospital targets
Step 3 — Train 6 models: SVR, RF, LSTM, CNN-LSTM, Transformer, PINN-LSTM
Step 4 — Print full comparison table (thesis Chapter 4, Table 4.3)

Usage:
    cd "C:\\Users\\admin\\Desktop\\Master 2027"
    python scripts/train_ct_tubes.py
    python scripts/train_ct_tubes.py --skip_generate   # if data already exists
    python scripts/train_ct_tubes.py --epochs 100      # faster run
"""

import sys, os, argparse, random
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.simulation.degradation_engine import DegradationEngine
from src.preprocessing.ct_loader import CTLoader
from src.utils.trainer import Trainer
from src.models.baselines import CNNLSTMPredictor, TransformerPredictor


# ---------------------------------------------------------------------------
# Reproducibility — fix every RNG so one run = one stable result table.
# (The data split is already seeded in ct_loader.py; this covers PyTorch
#  weight initialisation and the training DataLoader shuffle.)
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class LSTMPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        m = config["model"]
        self.lstm = nn.LSTM(input_size=m["input_features"],
                            hidden_size=m["hidden_size"],
                            num_layers=m["num_layers"],
                            dropout=m["dropout"] if m["num_layers"] > 1 else 0.0,
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(m["hidden_size"], 64), nn.ReLU(),
            nn.Dropout(m["dropout"]),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, m["output_size"]), nn.Sigmoid(),
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_config(base_config: dict, n_features: int, epochs: int) -> dict:
    """Build the per-model config from the YAML single source of truth.

    Only runtime values are overridden: n_features (from the data loader)
    and epochs (from the --epochs CLI arg). Architecture and training
    hyperparameters all come from configs/config.yaml, so there is exactly
    one authoritative definition and the script can no longer silently
    diverge from the config file.
    """
    m = dict(base_config["model"])
    t = dict(base_config["training"])
    m["input_features"] = n_features   # authoritative at runtime (from loader)
    m.setdefault("output_size", 1)
    t["epochs"] = epochs               # from --epochs CLI arg
    return {"model": m, "training": t}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# CT tube RUL is expressed in scan cycles (0 – ~150,000).
# 1 day ≈ 300 scan cycles: 80 patient exams × ~3-4 tube activations per exam.
# NASA score was designed for C-MAPSS where RUL ≤ 125 cycles; applying it
# to CT-scale RUL causes exp(150000/10) → inf overflow.  For CT tubes we
# therefore report RMSE / MAE converted to **days** (clinically meaningful)
# and omit NASA score.  NASA score is kept only for the C-MAPSS benchmark.
CT_SCANS_PER_DAY = 300  # scan cycles per day (documented in configs/config.yaml)

def nasa_score(y_true, y_pred):
    """C-MAPSS NASA score. Only valid when RUL is in the 0-125 cycle range."""
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)))

def compute_metrics(y_true, y_pred, scans_per_day=None):
    """
    Compute RUL prediction metrics.

    Parameters
    ----------
    y_true, y_pred : arrays in scan cycles
    scans_per_day  : if provided, RMSE/MAE are converted to days before
                     reporting, and NASA score is omitted (CT-tube mode).
                     Pass None to get raw-cycle metrics + NASA score (C-MAPSS mode).
    """
    if scans_per_day is not None:
        # Convert to days for clinical interpretability
        y_true = y_true / scans_per_day
        y_pred = y_pred / scans_per_day

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }
    if scans_per_day is None:
        # NASA score is only meaningful at C-MAPSS scale (~0-125 cycles)
        metrics["nasa_score"] = nasa_score(y_true, y_pred)
    else:
        metrics["nasa_score"] = float("nan")   # N/A for CT tubes
    return metrics

def evaluate_torch(model, loader, device, rul_cap, scans_per_day=None):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.extend(model(X.to(device)).cpu().numpy().flatten())
            trues.extend(y.numpy().flatten())
    return compute_metrics(np.array(trues)*rul_cap, np.array(preds)*rul_cap,
                           scans_per_day=scans_per_day)

def collect_flat(loader, rul_cap):
    Xs, ys = [], []
    for X, y in loader:
        Xs.append(X.numpy()); ys.append(y.numpy())
    X = np.concatenate(Xs).reshape(sum(len(x) for x in Xs), -1)
    y = np.concatenate(ys).flatten() * rul_cap
    return np.concatenate([x.reshape(len(x),-1) for x in Xs]), y

def _print_m(label, m, unit="cycles"):
    nasa_str = f"{m['nasa_score']:.1f}" if not np.isnan(m['nasa_score']) else "N/A"
    print(f"\n  Test Results ({label}):")
    print(f"    RMSE  : {m['rmse']:.1f} {unit}  |  MAE: {m['mae']:.1f}  |  R2: {m['r2']:.4f}  |  NASA: {nasa_str}")


# ---------------------------------------------------------------------------
# Step 1 — Generate dataset
# ---------------------------------------------------------------------------
def generate_data(config: dict, save_path: str):
    print("\n" + "="*60)
    print("  STEP 1 — Generating synthetic CT-tube dataset")
    print("="*60)
    import csv, pandas as pd
    from collections import Counter
    engine = DegradationEngine(config)
    num_tubes = config["simulation"]["num_tubes"]

    COLS = ["tube_id","scan_cycle","anode_temperature","oil_temperature",
            "tube_voltage_kv","tube_current_ma","filament_current_a",
            "filament_diameter_m","vacuum_pressure_pa","rotor_vibration_rms",
            "thermal_stress_pa","cumulative_stress_pa","ambient_temperature_k",
            "arcing_events","health_indicator","rul","failure_mode"]

    lifespans, failure_modes = [], []
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        for i in range(num_tubes):
            records = engine.simulate_single_tube(tube_id=i, seed=i * 42)
            if records:
                writer.writerows(records)
                lifespans.append(records[-1]["scan_cycle"])
                failure_modes.append(records[-1]["failure_mode"])
            if (i + 1) % 50 == 0:
                print(f"  Generated {i+1}/{num_tubes} tubes")

    lifespans = np.array(lifespans)
    scans_per_day = config["simulation"]["scans_per_day_mean"]
    lifespan_days = lifespans / scans_per_day
    print(f"\n  Lifespan stats:")
    print(f"    Mean : {lifespans.mean():.0f} cycles = {lifespan_days.mean():.0f} days")
    print(f"    Std  : {lifespans.std():.0f} cycles = {lifespan_days.std():.0f} days")
    print(f"    Min  : {lifespans.min():.0f} cycles = {lifespan_days.min():.0f} days")
    print(f"    Max  : {lifespans.max():.0f} cycles = {lifespan_days.max():.0f} days")
    print(f"  Failure modes: {dict(Counter(failure_modes))}")
    print(f"  Dataset saved -> {save_path}")

    return pd.DataFrame({
        "tube_id": list(range(len(lifespans))),
        "scan_cycle": lifespans,
        "failure_mode": failure_modes,
    })


# ---------------------------------------------------------------------------
# Step 2 — Validate degradation model
# ---------------------------------------------------------------------------
def validate_model(config: dict, dataset):
    print("\n" + "="*60)
    print("  STEP 2 — Degradation Model Validation")
    print("="*60)

    scans_per_day = config["simulation"]["scans_per_day_mean"]
    target_days   = config["simulation"]["target_lifespan_days"]   # 395
    tolerance     = config["evaluation"]["calibration"]["tolerance_days"]  # 60

    # Defensive: collapse to one row per tube (max scan_cycle = lifespan).
    # Guards against being passed per-row data, which previously made the mean
    # average every cycle index (~half the true lifespan: 193 days instead of
    # 379) and count failure modes as ROWS (millions) instead of tubes.
    if dataset.groupby("tube_id").size().max() > 1:
        dataset = (dataset.groupby("tube_id")
                   .agg(scan_cycle=("scan_cycle", "max"),
                        failure_mode=("failure_mode", "last"))
                   .reset_index())
    lifespans = dataset["scan_cycle"]
    mean_days = (lifespans / scans_per_day).mean()
    std_days  = (lifespans / scans_per_day).std()

    error_pct = abs(mean_days - target_days) / target_days * 100
    passed = abs(mean_days - target_days) <= tolerance

    print(f"\n  Hospital calibration target : {target_days} days (CHU Ibn Rochd, Zakaria 2023)")
    print(f"  Simulated mean lifespan     : {mean_days:.1f} days  (std = {std_days:.1f} days)")
    print(f"  Calibration error           : {error_pct:.1f}%")
    print(f"  Tolerance window            : +/- {tolerance} days")
    print(f"  Validation result           : {'PASSED' if passed else 'FAILED'}")

    # Failure mode distribution
    modes = dataset["failure_mode"].value_counts()
    print(f"\n  Failure mode distribution:")
    for mode, count in modes.items():
        print(f"    {mode:<25} : {count:>4} tubes  ({count/modes.sum()*100:.1f}%)")

    print(f"\n  THESIS NOTE (Chapter 3 — Model Validation):")
    if passed:
        print(f"  The physics simulation reproduces the observed CHU Ibn Rochd mean")
        print(f"  tube lifespan of {target_days} days within {error_pct:.1f}% error, confirming")
        print(f"  the calibration of the thermal, filament, and mechanical sub-models.")
    else:
        print(f"  WARNING: Simulation mean ({mean_days:.0f} days) deviates {error_pct:.1f}% from")
        print(f"  hospital target ({target_days} days). Physics sub-model may need recalibration.")

    return {"mean_days": mean_days, "std_days": std_days,
            "error_pct": error_pct, "passed": passed}


# ---------------------------------------------------------------------------
# Step 3 — Train models
# ---------------------------------------------------------------------------
def run_sklearn(label, model_obj, dataloaders, scans_per_day=None):
    print(f"\n{'='*60}\n  Training: {label}\n{'='*60}")
    rul_cap = dataloaders["rul_cap"]

    Xs_tr, ys_tr, Xs_te, ys_te = [], [], [], []
    for X, y in dataloaders["train"]:
        Xs_tr.append(X.numpy()); ys_tr.append(y.numpy())
    for X, y in dataloaders["test"]:
        Xs_te.append(X.numpy()); ys_te.append(y.numpy())

    X_tr = np.concatenate(Xs_tr).reshape(sum(len(x) for x in Xs_tr), -1)
    y_tr = np.concatenate(ys_tr).flatten() * rul_cap
    X_te = np.concatenate(Xs_te).reshape(sum(len(x) for x in Xs_te), -1)
    y_te = np.concatenate(ys_te).flatten() * rul_cap

    # Cap training samples for the classical baselines. SVR (RBF) scales
    # ~O(n^2) in samples, so the ~160k windows from 200 tubes would make it
    # intractable/OOM (this is what crushed the 500-tube run). A 30k random
    # subsample is a representative fit and keeps SVR/RF fast. Deep models are
    # unaffected - they train in mini-batches on the full set.
    MAX_SK_SAMPLES = 30000
    if len(X_tr) > MAX_SK_SAMPLES:
        _rng = np.random.RandomState(42)
        _idx = _rng.choice(len(X_tr), MAX_SK_SAMPLES, replace=False)
        X_tr, y_tr = X_tr[_idx], y_tr[_idx]
        print(f"  Subsampled train to {MAX_SK_SAMPLES:,} windows for {label} (SVR/RF scaling).")

    pipe = Pipeline([("sc", StandardScaler()), ("m", model_obj)])
    print(f"  Fitting on {len(X_tr):,} samples x {X_tr.shape[1]} features...")
    pipe.fit(X_tr, y_tr)
    metrics = compute_metrics(y_te, pipe.predict(X_te), scans_per_day=scans_per_day)
    unit = "days" if scans_per_day else "cycles"
    _print_m(label, metrics, unit=unit)
    return metrics


def run_deep(label, model_class, use_pinn, dataloaders, config, device, save_path,
             scans_per_day=None):
    print(f"\n{'='*60}\n  Training: {label}\n{'='*60}")
    model = model_class(config)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    trainer = Trainer(model=model, config=config, device=device, use_pair_loss=use_pinn)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trainer.fit(dataloaders["train"], dataloaders["val"],
                pair_loader=dataloaders.get("train_pairs") if use_pinn else None,
                save_path=save_path)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    metrics = evaluate_torch(model, dataloaders["test"], device, dataloaders["rul_cap"],
                             scans_per_day=scans_per_day)
    unit = "days" if scans_per_day else "cycles"
    _print_m(label, metrics, unit=unit)
    return metrics


# ---------------------------------------------------------------------------
# Step 4 — Print full table
# ---------------------------------------------------------------------------
def print_table(results, rul_cap, scans_per_day=None):
    unit = "days" if scans_per_day else "cycles"
    cap_str = (f"{rul_cap:,} cycles = {rul_cap//scans_per_day:,} days"
               if scans_per_day else f"{rul_cap:,} cycles")
    print("\n" + "="*80)
    print(f"  FULL COMPARISON — CT X-Ray Tubes (RUL cap = {cap_str})")
    print("="*80)
    lstm_rmse = next((r["m"]["rmse"] for r in results if r["label"]=="LSTM"), None)
    use_nasa = not any(np.isnan(r["m"]["nasa_score"]) for r in results)
    if use_nasa:
        print(f"  {'Model':<22} {'RMSE':>12} {'MAE':>10} {'R2':>8} {'NASA':>10}  {'vs LSTM':>9}")
        print(f"  {'-'*76}")
    else:
        print(f"  {'Model':<22} {'RMSE ('+unit+')':>14} {'MAE ('+unit+')':>12} {'R2':>8}  {'vs LSTM':>9}")
        print(f"  {'-'*76}")
    for r in results:
        m = r["m"]
        rel = f"{(lstm_rmse - m['rmse'])/lstm_rmse*100:>+8.1f}%" if lstm_rmse and r["label"]!="LSTM" else "  baseline"
        if use_nasa:
            print(f"  {r['label']:<22} {m['rmse']:>12.1f} {m['mae']:>10.1f} {m['r2']:>8.4f} {m['nasa_score']:>10.1f}  {rel}")
        else:
            print(f"  {r['label']:<22} {m['rmse']:>14.1f} {m['mae']:>12.1f} {m['r2']:>8.4f}  {rel}")
    print("="*80)
    if use_nasa:
        print("  NASA score: lower is better.\n")
    else:
        print(f"  RMSE/MAE in {unit} (1 day = {scans_per_day} scan cycles). NASA score omitted — N/A at CT-tube RUL scale.\n")

    rmse_hdr = f"RMSE ({unit})"
    mae_hdr  = f"MAE ({unit})"
    print(f"  PAPER TABLE — CT-Tube Experiment:")
    print(f"  +------------------------+-------------+-------------+---------+")
    print(f"  | Model                  | {rmse_hdr:<11}  | {mae_hdr:<11}  |   R2    |")
    print(f"  +------------------------+-------------+-------------+---------+")
    for r in results:
        m = r["m"]
        print(f"  | {r['label']:<22} | {m['rmse']:>11.1f}  | {m['mae']:>11.1f}  | {m['r2']:>7.4f} |")
    print(f"  +------------------------+-------------+-------------+---------+")


# ---------------------------------------------------------------------------
# Data-scarcity experiment
# ---------------------------------------------------------------------------
def run_data_scarce(config, DATA_PATH, device, epochs, fractions=(1.0, 0.5, 0.2), seed=42):
    """
    Train LSTM, CNN-LSTM, Transformer, and PINN-LSTM at multiple training-data
    fractions.  Test set is always the same 15 tubes (fixed by seed=42 in CTLoader).
    Prints a table showing how each model degrades as training data shrinks.
    """
    import pandas as pd

    MODELS = [
        ("LSTM",             LSTMPredictor,    False),
        ("CNN-LSTM",         CNNLSTMPredictor, False),
        ("Transformer",      TransformerPredictor, False),
        ("PINN-LSTM (ours)", LSTMPredictor,    True),
    ]

    # Point 8: classical baselines now run at every data fraction too, so they
    # appear in the scarcity table - not just the full comparison. Lambdas build
    # a fresh estimator per fraction. run_sklearn caps samples internally (SVR
    # RBF scaling), so these stay fast even at 200 tubes.
    SK_MODELS = [
        ("SVR",           lambda: SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")),
        ("Random Forest", lambda: RandomForestRegressor(n_estimators=100, max_depth=15,
                                                         n_jobs=-1, random_state=42)),
    ]

    all_rows = []   # list of dicts for CSV export

    for frac in fractions:
        pct = int(frac * 100)
        print(f"\n{'='*70}")
        print(f"  DATA FRACTION: {pct}%")
        print(f"{'='*70}")

        loader = CTLoader(
            csv_path=DATA_PATH,
            seq_len=config["data"]["sequence_length"],
            batch_size=32,
            max_tubes=200,
            stride=100,
            train_fraction=frac,
        )
        dls       = loader.prepare()
        n_feat    = dls["n_features"]
        rul_cap   = dls["rul_cap"]
        dl_config = make_config(config, n_features=n_feat, epochs=epochs)

        frac_results = []

        # Classical baselines first (run_sklearn caps samples internally).
        for sk_label, sk_factory in SK_MODELS:
            m = run_sklearn(sk_label, sk_factory(), dls, scans_per_day=CT_SCANS_PER_DAY)
            frac_results.append({"label": sk_label, "m": m})
            all_rows.append({
                "train_fraction": frac, "model": sk_label,
                "rmse_days": m["rmse"], "mae_days": m["mae"], "r2": m["r2"],
            })

        for label, model_cls, use_pinn in MODELS:
            save_path = f"results/ct_tubes/{label.replace(' ', '_').replace('(', '').replace(')', '')}_f{pct}.pt"
            m = run_deep(label, model_cls, use_pinn, dls, dl_config, device,
                         save_path, scans_per_day=CT_SCANS_PER_DAY)
            frac_results.append({"label": label, "m": m})
            all_rows.append({
                "train_fraction": frac,
                "model": label,
                "rmse_days": m["rmse"],
                "mae_days":  m["mae"],
                "r2":        m["r2"],
            })

        print_table(frac_results, rul_cap, scans_per_day=CT_SCANS_PER_DAY)

    # ── Summary table across fractions ───────────────────────────────────────
    results_df = pd.DataFrame(all_rows)
    print("\n" + "="*80)
    print("  DATA-SCARCITY SUMMARY — RMSE (days) by model and training data fraction")
    print("="*80)
    model_order = [m[0] for m in SK_MODELS] + [m[0] for m in MODELS]
    header = f"  {'Model':<22}" + "".join(f" {int(f*100):>8}%" for f in fractions)
    print(header)
    print("  " + "-"*70)
    for label in model_order:
        row_str = f"  {label:<22}"
        for frac in fractions:
            val = results_df[(results_df["model"] == label) &
                             (results_df["train_fraction"] == frac)]["rmse_days"]
            row_str += f" {val.values[0]:>9.2f}"
        print(row_str)
    print("="*80)

    # Print PINN advantage vs CNN-LSTM at each fraction
    print("\n  PINN-LSTM vs CNN-LSTM advantage (RMSE reduction, lower is better):")
    for frac in fractions:
        pinn = results_df[(results_df["model"] == "PINN-LSTM (ours)") &
                          (results_df["train_fraction"] == frac)]["rmse_days"].values[0]
        cnn  = results_df[(results_df["model"] == "CNN-LSTM") &
                          (results_df["train_fraction"] == frac)]["rmse_days"].values[0]
        delta = cnn - pinn
        pct_gain = delta / cnn * 100
        status = "PINN wins" if delta > 0 else "CNN-LSTM wins"
        print(f"    {int(frac*100):>3}% data: PINN={pinn:.2f}d  CNN-LSTM={cnn:.2f}d  "
              f"Δ={delta:+.2f}d ({pct_gain:+.1f}%)  [{status}]")

    out_csv = f"results/ct_tubes/data_scarce_seed{seed}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n  Saved -> {out_csv}")
    return results_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int, default=150)
    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument("--skip_sklearn",  action="store_true")
    parser.add_argument("--data_scarce",   action="store_true",
                        help="Run data-scarcity experiment (100%%, 50%%, 20%% training data)")
    parser.add_argument("--config",        default="configs/config.yaml")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)   # reproducibility — must be called before any model is built

    print("="*60)
    print("  CT-Tube Experiment: Physics Simulation + 6-Model Comparison")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs("data/synthetic", exist_ok=True)
    os.makedirs("results/ct_tubes", exist_ok=True)
    DATA_PATH = "data/synthetic/ct_tubes_small.csv"

    # ── Step 1: Generate ──────────────────────────────────────────────────
    if args.skip_generate and os.path.exists(DATA_PATH):
        print(f"\n  Skipping generation — loading {DATA_PATH}")
        import pandas as pd
        full_df = pd.read_csv(DATA_PATH)
        # validate_model expects one row per tube (tube_id, scan_cycle=lifespan, failure_mode)
        dataset = (
            full_df.groupby("tube_id")
            .agg(scan_cycle=("scan_cycle", "max"), failure_mode=("failure_mode", "last"))
            .reset_index()
        )
    else:
        dataset = generate_data(config, DATA_PATH)

    # ── Step 2: Validate ──────────────────────────────────────────────────
    val_result = validate_model(config, dataset)

    # ── Data-scarcity experiment (--data_scarce) ──────────────────────────
    if args.data_scarce:
        run_data_scarce(config, DATA_PATH, device, epochs=args.epochs, seed=args.seed)
        return

    # ── Load sequences ────────────────────────────────────────────────────
    print("\n  Preparing sequences...")
    loader = CTLoader(
        csv_path=DATA_PATH,
        seq_len=config["data"]["sequence_length"],   # from config.yaml (single source)
        batch_size=32,
        max_tubes=200,
        stride=100,       # one window every 100 cycles
    )
    dataloaders = loader.prepare()
    n_features = dataloaders["n_features"]
    rul_cap    = dataloaders["rul_cap"]
    dl_config  = make_config(config, n_features=n_features, epochs=args.epochs)

    results = []

    # ── Step 3: Train all models ──────────────────────────────────────────
    if not args.skip_sklearn:
        results.append({"label": "SVR", "m": run_sklearn(
            "SVR", SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale"), dataloaders,
            scans_per_day=CT_SCANS_PER_DAY)})
        results.append({"label": "Random Forest", "m": run_sklearn(
            "Random Forest", RandomForestRegressor(n_estimators=100, max_depth=15,
                                                   n_jobs=-1, random_state=42), dataloaders,
            scans_per_day=CT_SCANS_PER_DAY)})

    results.append({"label": "LSTM", "m": run_deep(
        "LSTM", LSTMPredictor, False, dataloaders, dl_config, device,
        "results/ct_tubes/lstm.pt", scans_per_day=CT_SCANS_PER_DAY)})
    results.append({"label": "CNN-LSTM", "m": run_deep(
        "CNN-LSTM", CNNLSTMPredictor, False, dataloaders, dl_config, device,
        "results/ct_tubes/cnn_lstm.pt", scans_per_day=CT_SCANS_PER_DAY)})
    results.append({"label": "Transformer", "m": run_deep(
        "Transformer", TransformerPredictor, False, dataloaders, dl_config, device,
        "results/ct_tubes/transformer.pt", scans_per_day=CT_SCANS_PER_DAY)})
    results.append({"label": "PINN-LSTM (ours)", "m": run_deep(
        "PINN-LSTM (ours)", LSTMPredictor, True, dataloaders, dl_config, device,
        "results/ct_tubes/pinn_lstm.pt", scans_per_day=CT_SCANS_PER_DAY)})

    # ── Step 4: Report ────────────────────────────────────────────────────
    print_table(results, rul_cap, scans_per_day=CT_SCANS_PER_DAY)

    import pandas as pd
    rows = []
    for r in results:
        row = {"model": r["label"],
               "rmse_days": r["m"]["rmse"],
               "mae_days":  r["m"]["mae"],
               "r2":        r["m"]["r2"]}
        # nasa_score is nan for CT tubes — omit column rather than write nan
        rows.append(row)
    pd.DataFrame(rows).to_csv("results/ct_tubes/full_comparison.csv", index=False)
    print("  Report saved -> results/ct_tubes/full_comparison.csv")
    print("  NOTE: RMSE/MAE stored in days (÷300 scan cycles/day). NASA score omitted for CT tubes.")


if __name__ == "__main__":
    main()
