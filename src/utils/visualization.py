"""
Visualization utilities for degradation data and model results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_degradation(df: pd.DataFrame, tube_id: int = 0, save_path: str = None):
    """Plot degradation trajectory for a single tube."""
    tube = df[df["tube_id"] == tube_id].sort_values("cycle")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"CT X-Ray Tube Degradation - Tube {tube_id}", fontsize=14)

    # Temperature
    axes[0, 0].plot(tube["cycle"], tube["temperature"], color="red", alpha=0.7)
    axes[0, 0].set_title("Anode Temperature (K)")
    axes[0, 0].set_xlabel("Scan Cycle")

    # Filament diameter
    axes[0, 1].plot(tube["cycle"], tube["filament_diameter"] * 1e6, color="blue")
    axes[0, 1].set_title("Filament Diameter (μm)")
    axes[0, 1].set_xlabel("Scan Cycle")

    # Cumulative stress
    axes[1, 0].plot(tube["cycle"], tube["cumulative_stress"], color="green")
    axes[1, 0].set_title("Cumulative Thermal Stress (Pa)")
    axes[1, 0].set_xlabel("Scan Cycle")

    # Health indicator
    axes[1, 1].plot(tube["cycle"], tube["health_indicator"], color="purple")
    axes[1, 1].set_title("Health Indicator")
    axes[1, 1].set_xlabel("Scan Cycle")
    axes[1, 1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_rul_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """Plot predicted vs actual RUL values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10)
    axes[0].plot([0, 1], [0, 1], "r--", label="Perfect prediction")
    axes[0].set_xlabel("Actual RUL")
    axes[0].set_ylabel("Predicted RUL")
    axes[0].set_title("Predicted vs Actual RUL")
    axes[0].legend()

    # Error distribution
    errors = y_pred - y_true
    axes[1].hist(errors, bins=50, alpha=0.7, color="steelblue")
    axes[1].set_xlabel("Prediction Error")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("RUL Prediction Error Distribution")
    axes[1].axvline(x=0, color="red", linestyle="--")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
