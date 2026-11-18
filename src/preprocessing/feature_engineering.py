"""
Feature engineering for CT-scanner degradation data.
Extracts derived features: temperature gradients, stress rates, usage intensity.
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Extracts and adds derived features to degradation data."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all derived features to the dataset."""
        df = df.copy()

        for tube_id in df["tube_id"].unique():
            mask = df["tube_id"] == tube_id
            tube_data = df.loc[mask].sort_values("cycle")

            # Temperature gradient (rate of change)
            df.loc[mask, "temp_gradient"] = tube_data["temperature"].diff().fillna(0)

            # Rolling average temperature (thermal load indicator)
            df.loc[mask, "temp_rolling_mean"] = (
                tube_data["temperature"].rolling(self.window_size, min_periods=1).mean()
            )

            # Stress accumulation rate
            df.loc[mask, "stress_rate"] = tube_data["cumulative_stress"].diff().fillna(0)

            # Electrical power (kV * mA)
            df.loc[mask, "power_kw"] = (
                tube_data["voltage_kv"] * tube_data["current_ma"] / 1000
            )

            # Filament wear rate
            df.loc[mask, "filament_wear_rate"] = (
                tube_data["filament_diameter"].diff().fillna(0).abs()
            )

        return df
