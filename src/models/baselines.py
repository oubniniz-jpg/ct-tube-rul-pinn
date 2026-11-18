"""
Baseline models for comparative experiments.

Models implemented:
  1. CNNLSTMPredictor  — 1D-Conv feature extractor + LSTM temporal model
  2. TransformerPredictor — Multi-head self-attention encoder for sequences

Both share the same interface as LSTMPredictor:
  forward(x) where x is (batch, seq_len, n_features) → (batch, 1) normalised RUL
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CNN-LSTM
# ---------------------------------------------------------------------------
class CNNLSTMPredictor(nn.Module):
    """
    1D-Convolutional feature extractor followed by an LSTM.

    Architecture:
      Conv1d(n_features → 64, kernel=3) → ReLU → MaxPool(2)
      → LSTM(64, hidden=64, layers=2)
      → FC head → Sigmoid → RUL in [0,1]

    The CNN extracts local temporal patterns; the LSTM models long-range
    temporal dependencies across the compressed sequence.
    """

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        n_feat = m["input_features"]
        hidden = m["hidden_size"]       # reuse same hidden size as LSTM
        dropout = m["dropout"]

        # CNN block — operates on (batch, channels, seq_len)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_feat, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),   # halves seq_len
            nn.Dropout(dropout),
        )

        # LSTM on the CNN output
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )

        # FC head (identical to LSTMPredictor)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, m["output_size"]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)          # → (batch, n_features, seq_len)
        x = self.cnn(x)                  # → (batch, 64, seq_len//2)
        x = x.permute(0, 2, 1)          # → (batch, seq_len//2, 64)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])    # last timestep → RUL


# ---------------------------------------------------------------------------
# Positional Encoding (standard sinusoidal)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)             # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
class TransformerPredictor(nn.Module):
    """
    Transformer encoder for RUL sequence regression.

    Architecture:
      Linear projection (n_features → d_model)
      → Positional Encoding
      → N × TransformerEncoderLayer (multi-head self-attention + FFN)
      → Mean pooling over sequence
      → FC head → Sigmoid → RUL in [0,1]

    Reference: Vaswani et al. (2017), adapted for regression.
    """

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        n_feat   = m["input_features"]
        d_model  = 64          # embedding dimension
        nhead    = 4           # attention heads (d_model must be divisible by nhead)
        num_layers = 2
        dim_ff   = 128         # feedforward inner dimension
        dropout  = m["dropout"]

        self.input_proj = nn.Linear(n_feat, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, m["output_size"]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)           # → (batch, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)          # → (batch, seq_len, d_model)
        x = x.mean(dim=1)               # mean pooling over sequence
        return self.fc(x)
