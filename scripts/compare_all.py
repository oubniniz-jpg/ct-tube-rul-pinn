"""
Comprehensive Model Comparison — C-MAPSS FD001
===============================================
Trains and evaluates 6 methods on the NASA C-MAPSS benchmark:

  1. SVR            — Support Vector Regression (sklearn, flattened sequences)
  2. Random Forest  — Ensemble tree method     (sklearn, flattened sequences)
  3. LSTM           — Baseline LSTM            (PyTorch, MSE loss only)
  4. CNN-LSTM       — Conv1D + LSTM hybrid     (PyTorch, MSE loss only)
  5. Transformer    — Self-attention encoder   (PyTorch, MSE loss only)
  6. PINN-LSTM      — LSTM + physics loss      (PyTorch, pair monotonicity)

This is the comparative experiment table required by the thesis supervisor.
Results are saved to results/cmapss/full_comparison.csv

Usage:
    cd "C:\\Users\\admin\\Desktop\\Master 2027"
    python scripts/compare_all.py
    python scripts/compare_all.py --epochs 100   # faster run
"""

import sys
import os
import copy
import argparse
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

from src.preprocessing.cmapss_loader import CMAPSSLoader
from src.utils.trainer import Trainer
from src.models.baselines import CNNLSTMPredictor, TransformerPredictor


# ---------------------------------------------------------------------------
# Inline LSTM (same as train_cmapss.py — avoids circular imports)
# ---------------------------------------------------------------------------
class LSTMPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        m = config["model"]
        self.lstm = nn.LSTM(
            input_size=m["input_features"], hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
            dropout=m["dropout"] if m["num_layers"] > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(m["hidden_size"], 64), nn.ReLU(),
            nn.Dropout(m["dropout"]),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, m["output_size"]), nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------
def make_config(n_features: int, epochs: int = 150) -> dict:
    return {
        "model": {
            "type": "lstm",
            "input_features": n_features,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "output_size": 1,
        },
        "training": {
            "epochs": epochs,
            "batch_size": 256,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "physics_loss_weight": 0.3,
            "early_stopping_patience": 20,
            "scheduler_step_size": 50,
            "scheduler_gamma": 0.5,
        },
    }


# ---------------------------------------------------------------------------
# Evaluation (shared)
# ---------------------------------------------------------------------------
def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def compute_metrics(y_true, y_pred):
    return {
        "rmse":      float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":       float(mean_absolute_error(y_true, y_pred)),
        "r2":        float(r2_score(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
    }


def evaluate_torch(model, dataloader, device, rul_cap):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device)).cpu().numpy().flatten()
            preds.extend(pred)
            trues.extend(y.numpy().flatten())
    return compute_metrics(
        np.array(trues) * rul_cap,
        np.array(preds) * rul_cap,
    )


# ---------------------------------------------------------------------------
# Sklearn helpers — flatten (N, seq_len, F) → (N, seq_len*F)
# ---------------------------------------------------------------------------
def collect_arrays(dataloader, rul_cap):
    """Extract numpy arrays from a DataLoader."""
    Xs, ys = [], []
    for X, y in dataloader:
        Xs.append(X.numpy())
        ys.append(y.numpy())
    X = np.concatenate(Xs, axis=0)                       # (N, seq_len, F)
    y = np.concatenate(ys, axis=0).flatten() * rul_cap   # (N,) in cycles
    X_flat = X.reshape(len(X), -1)                        # (N, seq_len*F)
    return X_flat, y


# ---------------------------------------------------------------------------
# SVR
# ---------------------------------------------------------------------------
def run_svr(dataloaders):
    print("\n" + "=" * 60)
    print("  Training: SVR")
    print("=" * 60)
    rul_cap = dataloaders["rul_cap"]
    X_tr, y_tr = collect_arrays(dataloaders["train"], rul_cap)
    X_te, y_te = collect_arrays(dataloaders["test"],  rul_cap)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")),
    ])
    print(f"  Fitting SVR on {len(X_tr):,} samples x {X_tr.shape[1]} features...")
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    metrics = compute_metrics(y_te, y_pred)
    _print_metrics("SVR", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
def run_random_forest(dataloaders):
    print("\n" + "=" * 60)
    print("  Training: Random Forest")
    print("=" * 60)
    rul_cap = dataloaders["rul_cap"]
    X_tr, y_tr = collect_arrays(dataloaders["train"], rul_cap)
    X_te, y_te = collect_arrays(dataloaders["test"],  rul_cap)

    rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                               n_jobs=-1, random_state=42)
    print(f"  Fitting Random Forest on {len(X_tr):,} samples...")
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    metrics = compute_metrics(y_te, y_pred)
    _print_metrics("Random Forest", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Deep learning runner (LSTM / CNN-LSTM / Transformer / PINN-LSTM)
# ---------------------------------------------------------------------------
def run_deep(label, model_class, use_pinn, dataloaders, config, device, save_path):
    print("\n" + "=" * 60)
    print(f"  Training: {label}")
    print("=" * 60)

    model = model_class(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    trainer = Trainer(model=model, config=config, device=device,
                      use_pair_loss=use_pinn)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trainer.fit(
        dataloaders["train"],
        dataloaders["val"],
        pair_loader=dataloaders.get("train_pairs") if use_pinn else None,
        save_path=save_path,
    )

    model.load_state_dict(torch.load(save_path, weights_only=True))
    metrics = evaluate_torch(model, dataloaders["test"], device, dataloaders["rul_cap"])
    _print_metrics(label, metrics)
    return metrics


def _print_metrics(label, m):
    print(f"\n  Test Results ({label}):")
    print(f"    RMSE       : {m['rmse']:.3f} cycles")
    print(f"    MAE        : {m['mae']:.3f} cycles")
    print(f"    R2         : {m['r2']:.4f}")
    print(f"    NASA Score : {m['nasa_score']:.1f}")


# ---------------------------------------------------------------------------
# Final comparison table
# ---------------------------------------------------------------------------
def print_final_table(results: list, rul_cap: float):
    print("\n" + "=" * 80)
    print(f"  FULL COMPARISON — C-MAPSS FD001 (RUL cap = {rul_cap} cycles)")
    print("=" * 80)
    print(f"  {'Model':<22} {'RMSE':>10} {'MAE':>10} {'R2':>8} {'NASA':>10}  {'vs LSTM':>9}")
    print(f"  {'-'*75}")

    # Find LSTM baseline for relative comparison
    lstm_rmse = next((r["metrics"]["rmse"] for r in results if r["label"] == "LSTM"), None)

    for r in results:
        m = r["metrics"]
        rel = ""
        if lstm_rmse and r["label"] != "LSTM":
            imp = (lstm_rmse - m["rmse"]) / lstm_rmse * 100
            rel = f"{imp:>+8.1f}%"
        else:
            rel = "  baseline"
        print(f"  {r['label']:<22} {m['rmse']:>10.3f} {m['mae']:>10.3f} "
              f"{m['r2']:>8.4f} {m['nasa_score']:>10.1f}  {rel}")

    print("=" * 80)
    print("  NASA score: lower is better (late predictions penalised more).")

    # Thesis table
    print(f"""
  THESIS TABLE (Chapter 4):
  +------------------------+-----------+-----------+---------+------------+
  | Model                  |  RMSE     |  MAE      |   R2    | NASA Score |
  +------------------------+-----------+-----------+---------+------------+""")
    for r in results:
        m = r["metrics"]
        print(f"  | {r['label']:<22} | {m['rmse']:>9.3f} | {m['mae']:>9.3f} | {m['r2']:>7.4f} | {m['nasa_score']:>10.1f} |")
    print("  +------------------------+-----------+-----------+---------+------------+")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=150)
    parser.add_argument("--subset",   default="FD001",
                        choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--data_dir", default="data/cmapss")
    parser.add_argument("--seq_len",  type=int, default=30)
    parser.add_argument("--skip_sklearn", action="store_true",
                        help="Skip SVR and Random Forest (much faster)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Full Model Comparison: 6 Methods on C-MAPSS")
    print(f"  Dataset: {args.subset} | Epochs: {args.epochs}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    # Load data
    data_dir = args.data_dir
    if not os.path.isfile(os.path.join(data_dir, f"train_{args.subset}.txt")):
        candidate = os.path.join(data_dir, "CMaps")
        if os.path.isfile(os.path.join(candidate, f"train_{args.subset}.txt")):
            data_dir = candidate
    print(f"\n  Loading C-MAPSS {args.subset}...")
    loader = CMAPSSLoader(data_dir=data_dir, subset=args.subset,
                          seq_len=args.seq_len, batch_size=256)
    dataloaders = loader.prepare()
    n_features = dataloaders["n_features"]
    rul_cap    = dataloaders["rul_cap"]

    config = make_config(n_features=n_features, epochs=args.epochs)
    results = []

    # 1. SVR
    if not args.skip_sklearn:
        results.append({"label": "SVR",
                        "metrics": run_svr(dataloaders)})

    # 2. Random Forest
    if not args.skip_sklearn:
        results.append({"label": "Random Forest",
                        "metrics": run_random_forest(dataloaders)})

    # 3. LSTM baseline
    results.append({"label": "LSTM",
                    "metrics": run_deep(
                        "LSTM", LSTMPredictor, False,
                        dataloaders, config, device,
                        "results/cmapss/cmp_lstm.pt")})

    # 4. CNN-LSTM
    results.append({"label": "CNN-LSTM",
                    "metrics": run_deep(
                        "CNN-LSTM", CNNLSTMPredictor, False,
                        dataloaders, config, device,
                        "results/cmapss/cmp_cnnlstm.pt")})

    # 5. Transformer
    results.append({"label": "Transformer",
                    "metrics": run_deep(
                        "Transformer", TransformerPredictor, False,
                        dataloaders, config, device,
                        "results/cmapss/cmp_transformer.pt")})

    # 6. PINN-LSTM
    results.append({"label": "PINN-LSTM (ours)",
                    "metrics": run_deep(
                        "PINN-LSTM (ours)", LSTMPredictor, True,
                        dataloaders, config, device,
                        "results/cmapss/cmp_pinn_lstm.pt")})

    # Print full comparison
    print_final_table(results, rul_cap)

    # Save CSV
    import pandas as pd
    rows = [{"model": r["label"], "dataset": args.subset, **r["metrics"]}
            for r in results]
    os.makedirs("results/cmapss", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/cmapss/full_comparison.csv", index=False)
    print("\n  Report saved -> results/cmapss/full_comparison.csv")


if __name__ == "__main__":
    main()
