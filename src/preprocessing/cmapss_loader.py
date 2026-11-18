"""
C-MAPSS Data Loader
====================
Loads and preprocesses the NASA C-MAPSS turbofan degradation dataset
for RUL prediction benchmarking.

Dataset: CMAPSS (Commercial Modular Aero-Propulsion System Simulation)
Source: NASA Prognostic Data Repository
Reference: Saxena, A. et al. (2008). "Damage propagation modeling for aircraft
           engine run-to-failure simulation." Proc. ICAS 2008.

Sub-datasets:
  FD001 — 1 operating condition, 1 fault mode (HPC degradation)  ← PRIMARY
  FD002 — 6 operating conditions, 1 fault mode
  FD003 — 1 operating condition, 2 fault modes
  FD004 — 6 operating conditions, 2 fault modes

FD001 is the standard benchmark used in most RUL papers
(Shamim 2024, Pomsar 2025 comparison literature).

Feature set: 14 sensors (drop 7 near-zero-variance sensors for FD001) + 1 cycle counter
= 15 features total.

The cycle counter (time_in_cycles, index 0 after unit_number is dropped) is the
key feature for the physics-informed gradient penalty:
  L_physics = mean(relu( d(RUL_pred) / d(cycle) )^2)
This enforces: "predicted RUL must decrease as the cycle count increases."
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.preprocessing import MinMaxScaler


# Standard column names for C-MAPSS (26 columns, no header in files)
CMAPSS_COLUMNS = (
    ["unit_number", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors with near-zero variance in FD001 — standard to drop these
# Reference: most C-MAPSS literature (Heimes 2008, Li 2018, etc.)
DROP_SENSORS_FD001 = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                      "sensor_16", "sensor_18", "sensor_19"]

# Piecewise linear RUL cap (standard benchmark: cap high early-life RUL)
# This prevents the model from being penalized for not predicting exact RUL
# when the engine is operating normally (early degradation is flat).
RUL_CAP = 125  # cycles — widely used in C-MAPSS literature


class CMAPSSDataset(Dataset):
    """PyTorch Dataset for C-MAPSS sequences."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.X = torch.FloatTensor(sequences)
        self.y = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CMAPSSPairDataset(Dataset):
    """Same-engine pairs for the PINN monotonicity constraint.

    Each item is (X_early, X_late) — two non-overlapping windows from the
    SAME engine where X_late comes strictly AFTER X_early.

    Physical law enforced: pred(X_early) > pred(X_late)
    (remaining life must decrease as the engine ages).

    A standard LSTM trained on shuffled independent sequences CAN violate
    this across inference calls. This dataset provides the supervision signal
    to penalise those violations during training.
    """

    def __init__(self, sequences_by_engine: list, step: int = None):
        """
        Args:
            sequences_by_engine: list of np.ndarray, each (T_i, seq_len, n_features)
                                  one array per engine
            step: minimum gap between early and late window start indices.
                  Defaults to seq_len (no temporal overlap between windows).
        """
        self.pairs_early: list = []
        self.pairs_late: list = []

        for eng_seqs in sequences_by_engine:
            T = len(eng_seqs)
            if T < 2:
                continue
            gap = step if step is not None else 1
            for i in range(T - gap):
                j = i + gap  # late window is gap steps ahead
                self.pairs_early.append(eng_seqs[i])
                self.pairs_late.append(eng_seqs[j])

        self.early = torch.FloatTensor(np.array(self.pairs_early))
        self.late  = torch.FloatTensor(np.array(self.pairs_late))

    def __len__(self):
        return len(self.early)

    def __getitem__(self, idx):
        return self.early[idx], self.late[idx]


class CMAPSSLoader:
    """
    Loads C-MAPSS data and prepares train/val/test splits.

    Feature layout after preprocessing (FD001, 15 features):
      Index 0: time_in_cycles  ← cycle counter (used for physics gradient penalty)
      Index 1: op_setting_1
      Index 2: op_setting_2
      Index 3: op_setting_3
      Index 4-17: sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9,
                  sensor_11, sensor_12, sensor_13, sensor_14, sensor_15,
                  sensor_17, sensor_20, sensor_21

    CYCLE_FEATURE_IDX = 0 → passed to PhysicsInformedLoss as cycle_feature_idx
    """

    CYCLE_FEATURE_IDX = 0  # position of time_in_cycles after preprocessing

    def __init__(self, data_dir: str = "data/cmapss",
                 subset: str = "FD001",
                 seq_len: int = 30,
                 rul_cap: int = RUL_CAP,
                 val_fraction: float = 0.15,
                 batch_size: int = 256,
                 drop_sensors: list = None):
        self.data_dir = data_dir
        self.subset = subset
        self.seq_len = seq_len
        self.rul_cap = rul_cap
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.drop_sensors = drop_sensors if drop_sensors is not None else DROP_SENSORS_FD001
        self.scaler = MinMaxScaler()
        self.n_features = None  # set after loading

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def _load_file(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(path, sep=r"\s+", header=None,
                         names=CMAPSS_COLUMNS, engine="python")
        return df

    # ------------------------------------------------------------------
    # RUL labelling
    # ------------------------------------------------------------------
    def _add_rul_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RUL column: counts down from max_cycle to 0 per engine."""
        max_cycles = df.groupby("unit_number")["time_in_cycles"].max()
        df = df.merge(max_cycles.rename("max_cycle"), on="unit_number")
        df["rul"] = df["max_cycle"] - df["time_in_cycles"]
        df.drop(columns=["max_cycle"], inplace=True)
        # Apply piecewise-linear cap
        df["rul"] = df["rul"].clip(upper=self.rul_cap)
        return df

    # ------------------------------------------------------------------
    # Sequence creation
    # ------------------------------------------------------------------
    def _build_sequences(self, df: pd.DataFrame, feature_cols: list,
                         return_per_engine: bool = False) -> tuple:
        """Build sliding-window sequences per engine.

        Returns:
            X: (N, seq_len, n_features)
            y: (N,)  — normalised RUL in [0, 1]
            per_engine (optional): list of (T_i, seq_len, n_features) arrays,
                                   one per engine — used to build PairDataset.
        """
        X_list, y_list, per_engine = [], [], []
        for uid in df["unit_number"].unique():
            eng = df[df["unit_number"] == uid].sort_values("time_in_cycles")
            feats = eng[feature_cols].values   # (T, F)
            rul = eng["rul"].values             # (T,)
            T = len(feats)
            if T < self.seq_len:
                continue
            eng_seqs = []
            for t in range(T - self.seq_len + 1):
                X_list.append(feats[t: t + self.seq_len])
                y_list.append(rul[t + self.seq_len - 1])
                eng_seqs.append(feats[t: t + self.seq_len])
            per_engine.append(np.array(eng_seqs, dtype=np.float32))

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        if return_per_engine:
            return X, y, per_engine
        return X, y

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def prepare(self) -> dict:
        """Load, preprocess, and split C-MAPSS data.

        Returns:
            dict with keys 'train', 'val', 'test' → TorchDataLoader instances
            Also sets self.n_features and self.feature_names for model config.
        """
        # Load raw files
        train_df = self._load_file(f"train_{self.subset}.txt")
        test_df = self._load_file(f"test_{self.subset}.txt")
        rul_test = pd.read_csv(
            os.path.join(self.data_dir, f"RUL_{self.subset}.txt"),
            header=None, names=["rul_true"]
        )

        # Feature columns (drop unit_number; keep time_in_cycles)
        feature_cols = [c for c in CMAPSS_COLUMNS
                        if c != "unit_number" and c not in self.drop_sensors]
        self.feature_names = feature_cols
        self.n_features = len(feature_cols)

        print(f"  C-MAPSS {self.subset} — {self.n_features} features: {feature_cols}")

        # Add RUL labels to training data
        train_df = self._add_rul_labels(train_df)

        # Fit scaler on training features only (no data leakage)
        all_train_feats = train_df[feature_cols].values
        self.scaler.fit(all_train_feats)

        # Scale training features
        train_df[feature_cols] = self.scaler.transform(train_df[feature_cols].values)

        # Normalise RUL to [0, 1]
        train_df["rul_norm"] = train_df["rul"] / self.rul_cap

        # Split training engines into train / val
        engine_ids = train_df["unit_number"].unique()
        np.random.seed(42)
        np.random.shuffle(engine_ids)
        n_val = max(1, int(len(engine_ids) * self.val_fraction))
        val_ids = set(engine_ids[:n_val])
        train_ids = set(engine_ids[n_val:])

        tr_df = train_df[train_df["unit_number"].isin(train_ids)]
        va_df = train_df[train_df["unit_number"].isin(val_ids)]

        # Build sequences — also get per-engine arrays for PairDataset
        X_tr, y_tr, per_engine_tr = self._build_sequences(tr_df, feature_cols,
                                                            return_per_engine=True)
        X_va, y_va = self._build_sequences(va_df, feature_cols)

        # Prepare test set
        # For test: each engine has one final sequence (last seq_len cycles)
        test_df[feature_cols] = self.scaler.transform(test_df[feature_cols].values)
        X_te, y_te_raw = [], []
        for i, uid in enumerate(sorted(test_df["unit_number"].unique())):
            eng = test_df[test_df["unit_number"] == uid].sort_values("time_in_cycles")
            feats = eng[feature_cols].values
            if len(feats) < self.seq_len:
                # Pad with first row if sequence too short
                pad = np.tile(feats[0], (self.seq_len - len(feats), 1))
                feats = np.vstack([pad, feats])
            X_te.append(feats[-self.seq_len:])
            # True RUL from RUL_FD001.txt (already capped)
            true_rul = min(float(rul_test.iloc[i]["rul_true"]), self.rul_cap)
            y_te_raw.append(true_rul / self.rul_cap)

        X_te = np.array(X_te, dtype=np.float32)
        y_te = np.array(y_te_raw, dtype=np.float32)

        # Replace y_tr with normalised RUL
        y_tr = y_tr / self.rul_cap
        y_va = y_va / self.rul_cap

        # Build PairDataset from training engines (step=5: late window is 5 positions ahead)
        pair_ds = CMAPSSPairDataset(per_engine_tr, step=5)

        print(f"  Train sequences : {len(X_tr):,}")
        print(f"  Val   sequences : {len(X_va):,}")
        print(f"  Test  sequences : {len(X_te):,}")
        print(f"  Same-engine pairs (physics): {len(pair_ds):,}")

        def make_loader(X, y, shuffle):
            ds = CMAPSSDataset(X, y)
            return TorchDataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

        pair_loader = TorchDataLoader(pair_ds, batch_size=self.batch_size,
                                      shuffle=True, drop_last=True)

        return {
            "train":       make_loader(X_tr, y_tr, shuffle=True),
            "val":         make_loader(X_va, y_va, shuffle=False),
            "test":        make_loader(X_te, y_te, shuffle=False),
            "train_pairs": pair_loader,      # used by PINN-LSTM only
            "n_features":  self.n_features,
            "feature_names": self.feature_names,
            "rul_cap":     self.rul_cap,
        }
