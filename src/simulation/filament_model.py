"""
Filament Degradation Model for CT-Scanner X-Ray Tube
Implements Richardson-Dushman thermionic emission law (Equation 2 from proposal):
    J = A * T² * exp(-W / (k*T))
Models filament thinning due to evaporation at high temperatures.
"""

import numpy as np


class FilamentModel:
    """Models cathode filament degradation through evaporation."""

    def __init__(self, config: dict):
        filament = config["physics"]["filament"]
        self.A = filament["richardson_constant"]
        self.W = filament["work_function"]  # eV
        self.diameter = filament["initial_diameter"]
        self.initial_diameter = filament["initial_diameter"]
        self.critical_diameter = filament["critical_diameter"]
        self.evap_coeff = filament["evaporation_rate_coeff"]
        self.k_boltzmann = 8.617e-5  # eV/K

    def emission_current_density(self, temperature: float) -> float:
        """Richardson-Dushman equation: J = A·T²·exp(-W/(k·T))
        Returns current density in A/m².
        """
        if temperature < 300:
            return 0.0
        exponent = -self.W / (self.k_boltzmann * temperature)
        return self.A * temperature**2 * np.exp(exponent)

    def emission_current(self, temperature: float) -> float:
        """Total filament emission current (A), derived from Richardson-Dushman.
        Shamim (2024) uses filament_current as one of four primary sensor features.
        I = J · A_filament,  where A_filament ≈ π · (d/2)²  (cross-section area).
        This is a physics-consistent derivation rather than a free parameter.
        """
        J = self.emission_current_density(temperature)
        radius = self.diameter / 2.0
        area = np.pi * radius * radius
        return J * area

    def evaporation_step(self, temperature: float, dt: float) -> float:
        """Model filament diameter reduction due to thermal evaporation.
        Rate scales as coeff * exp(T/1000) — exponential in temperature.
        No hard threshold: even at low temperatures evaporation is non-zero
        (tungsten slowly erodes via ion bombardment and sublimation at any
        operating temperature). At room temperature the rate is negligible;
        at 2000 K it dominates. The exp scaling handles this naturally.
        """
        evap_rate = self.evap_coeff * np.exp(temperature / 1000.0)
        self.diameter -= evap_rate * dt
        self.diameter = max(self.diameter, 0.0)
        return self.diameter

    def get_health_indicator(self) -> float:
        """Return filament health as fraction of remaining life [0, 1].
        1.0 = new, 0.0 = failed
        """
        usable_range = self.initial_diameter - self.critical_diameter
        remaining = self.diameter - self.critical_diameter
        return np.clip(remaining / usable_range, 0.0, 1.0)

    def is_failed(self) -> bool:
        """Check if filament has degraded below critical threshold."""
        return self.diameter <= self.critical_diameter

    def reset(self):
        """Reset to initial conditions."""
        self.diameter = self.initial_diameter
