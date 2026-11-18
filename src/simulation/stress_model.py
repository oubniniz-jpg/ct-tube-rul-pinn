"""
Thermal Stress Model for CT-Scanner X-Ray Tube
Implements: σ = E·α·ΔT  (Equation 3 from proposal)
Models cumulative mechanical fatigue from repeated thermal cycling.
"""

import numpy as np


class StressModel:
    """Models thermal stress accumulation in the anode."""

    def __init__(self, config: dict):
        anode = config["physics"]["anode"]
        self.E = anode["youngs_modulus"]
        self.alpha = anode["thermal_expansion_coeff"]
        self.T0 = anode["initial_temperature"]
        self.cumulative_stress = 0.0
        self.cycle_count = 0
        self.peak_stress_history = []

    def compute_stress(self, temperature: float) -> float:
        """Compute instantaneous thermal stress: σ = E·α·(T - T0)"""
        delta_T = temperature - self.T0
        return self.E * self.alpha * delta_T

    def accumulate_cycle(self, peak_temperature: float):
        """Record one thermal cycle and accumulate fatigue damage.
        Uses simplified Miner's rule for cumulative damage.
        """
        stress = self.compute_stress(peak_temperature)
        self.peak_stress_history.append(stress)
        self.cumulative_stress += stress
        self.cycle_count += 1

    def get_fatigue_indicator(self, failure_threshold: float = 1.0e15) -> float:
        """Return fatigue damage as fraction [0, 1].
        0.0 = new, 1.0 = failure threshold reached
        """
        return np.clip(self.cumulative_stress / failure_threshold, 0.0, 1.0)

    def reset(self):
        """Reset to initial conditions."""
        self.cumulative_stress = 0.0
        self.cycle_count = 0
        self.peak_stress_history = []
