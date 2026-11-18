"""
CNN-LSTM Hybrid Model for RUL Prediction.
CNN extracts local patterns, LSTM captures temporal dependencies.
"""

import torch
import torch.nn as nn


class CNNLSTMPredictor(nn.Module):
    """CNN-LSTM hybrid for RUL prediction with spatial-temporal learning."""

    def __init__(self, config: dict):
        super().__init__()
        model_config = config["model"]
        cnn_config = model_config["cnn"]

        input_features = model_config["input_features"]
        channels = cnn_config["channels"]
        kernel_size = cnn_config["kernel_size"]

        # CNN layers for local feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, channels[0], kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(channels[0], channels[1], kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=channels[1],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(model_config["hidden_size"], 64),
            nn.ReLU(),
            nn.Dropout(model_config["dropout"]),
            nn.Linear(64, model_config["output_size"]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass.
        Args:
            x: tensor of shape (batch, seq_len, input_features)
        Returns:
            RUL prediction of shape (batch, 1)
        """
        # CNN expects (batch, channels, seq_len)
        x_cnn = x.permute(0, 2, 1)
        cnn_out = self.cnn(x_cnn)
        # Back to (batch, seq_len, features)
        cnn_out = cnn_out.permute(0, 2, 1)

        lstm_out, _ = self.lstm(cnn_out)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)
