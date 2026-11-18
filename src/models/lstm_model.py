"""
LSTM Model for Remaining Useful Life (RUL) Prediction.
Core architecture from the research proposal.
"""

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """Stacked LSTM network for RUL prediction of CT-scanner X-ray tubes."""

    def __init__(self, config: dict):
        super().__init__()
        model_config = config["model"]

        self.hidden_size = model_config["hidden_size"]
        self.num_layers = model_config["num_layers"]

        self.lstm = nn.LSTM(
            input_size=model_config["input_features"],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=model_config["dropout"],
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(model_config["dropout"]),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, model_config["output_size"]),
            nn.Sigmoid(),  # RUL normalized to [0, 1]
        )

    def forward(self, x):
        """Forward pass.
        Args:
            x: tensor of shape (batch, seq_len, input_features)
        Returns:
            RUL prediction of shape (batch, 1)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)
