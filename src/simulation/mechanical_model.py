"""
Mechanical / Rotor Vibration Model for CT-Scanner X-Ray Tube
Shamim 2024 uses `rotor_vibration` as one of four primary sensor features.
CT tube anodes rotate at ~9000 rpm (150 Hz) on ball bearings in vacuum.
Bearing wear is a documented failure mode and produces a measurable increase
in the vibration RMS amplitude over time.

Model:
    rms(t) = baseline + wear_rate · cycle_count + thermal_factor(T) + noise
where thermal_factor captures the fact that extreme temperatures accelerate
lubricant degradation and bearing clearance growth.

Failure threshold follows ISO 10816-3 class criteria: 4.5 mm/s RMS is the
"unsatisfactory for long-term operation" threshold for rigid-mounted machines
of this size class.
"""

import numpy as np


class MechanicalModel:
    """Rotor/bearing vibration model for the CT tube anode assembly."""

    def __init__(self, config: dict):
        rotor = config["physics"].get("rotor", {})
        self.nominal_freq = rotor.get("nominal_frequency", 150.0)          # Hz
        self.baseline_rms = rotor.get("nominal_vibration_rms", 0.5)        # mm/s
        self.wear_rate = rotor.get("bearing_wear_rate", 1.0e-7)            # per cycle
        self.failure_rms = rotor.get("vibration_failure_rms", 4.5)         # mm/s

        anode = config["physics"]["anode"]
        self.T_ref = anode["initial_temperature"]
        self.T_max = anode["max_temperature"]

        self.bearing_wear = 0.0      # accumulated wear [0, 1+]
        self.current_rms = self.baseline_rms

    def update(self, cycle: int, temperature: float, dt: float = 1.0) -> float:
        """Advance one simulation step and return the updated RMS vibration (mm/s).

        Wear accumulates linearly in cycles but is amplified by thermal stress on
        the bearings (lubricant breakdown above nominal operating temperature).
        """
        # Thermal acceleration factor — normalized [1.0, ~3.0]
        thermal_factor = 1.0 + 2.0 * max(
            0.0, (temperature - self.T_ref) / (self.T_max - self.T_ref)
        )

        # Accumulate wear
        self.bearing_wear += self.wear_rate * thermal_factor * dt

        # Vibration RMS grows with wear (linear) + small stochastic noise
        wear_contribution = self.bearing_wear * (self.failure_rms - self.baseline_rms)
        noise = np.random.normal(0.0, 0.05)
        self.current_rms = max(0.0, self.baseline_rms + wear_contribution + noise)
        return self.current_rms

    def get_vibration_rms(self) -> float:
        """Return current rotor vibration RMS (mm/s)."""
        return self.current_rms

    def get_health_indicator(self) -> float:
        """Return bearing health as fraction of remaining life [0, 1]."""
        usable_range = self.failure_rms - self.baseline_rms
        remaining = self.failure_rms - self.current_rms
        return float(np.clip(remaining / usable_range, 0.0, 1.0))

    def is_failed(self) -> bool:
        """True when vibration RMS exceeds ISO 10816 failure threshold."""
        return self.current_rms >= self.failure_rms

    def reset(self) -> None:
        """Reset to initial conditions."""
        self.bearing_wear = 0.0
        self.current_rms = self.baseline_rms
