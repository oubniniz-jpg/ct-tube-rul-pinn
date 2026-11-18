"""
Thermal Model for CT-Scanner X-Ray Tube Anode
Implements: ρ·cp·(∂T/∂t) = ∇·(k·∇T) + Q  (Equation 1 from proposal)

Simplified to a 2-node lumped-parameter model:
  - Node 1: Anode body (focal-track heated zone)
  - Node 2: Oil jacket (Zhou 2024: oil_temperature is the #1 predictive feature)

Key physical insight: a CT anode disc rotates at ~9000 rpm so the focal spot
sweeps the full track circumference. Only the small focal zone is being heated
at any instant, but the WHOLE disc radiates continuously. The model therefore
uses two separate geometric parameters:
  - thermal_volume  → sets the effective thermal mass (focal track zone)
  - radiating_area  → sets the Stefan-Boltzmann cooling surface (full disc)
Radiation cooling is ALWAYS active, regardless of whether a scan is in progress.
This is the physically correct treatment and prevents the unrealistic runaway
heating seen when cooling is only applied between scans.
"""

import numpy as np


class ThermalModel:
    """Models anode + oil-jacket temperature evolution under repeated scan cycles."""

    def __init__(self, config: dict):
        anode = config["physics"]["anode"]
        sim = config.get("simulation", {})

        self.density = anode["density"]
        self.cp = anode["specific_heat"]
        self.T0 = anode["initial_temperature"]
        self.T_max = anode["max_temperature"]

        # Separated geometry (see module docstring)
        self.volume = anode.get("thermal_volume", 1.0e-4)   # m³ — thermal mass
        self.area = anode.get("radiating_area", 0.05)        # m² — radiating surface
        self.emissivity = anode.get("emissivity", 0.6)
        self.mass = self.density * self.volume               # kg

        # Cooling efficiency from CHU Ibn Rochd conditions (Waid 2023):
        # air-cooled system, 100% of technicians report cooling failures.
        self.cooling_efficiency = sim.get("cooling_efficiency", 1.0)
        self.ambient_temperature = sim.get("ambient_temperature", self.T0)

        # Oil jacket — Zhou 2024 (West China Hospital): oil temperature is the
        # strongest single predictive feature for arcing and overheating events.
        oil_cfg = config["physics"].get("oil_jacket", {})
        self.oil_T0 = oil_cfg.get("initial_temperature", self.T0)
        self.oil_coupling = oil_cfg.get("coupling_coefficient", 0.02)
        self.oil_cooling = oil_cfg.get("cooling_coefficient", 0.005)
        self.oil_T_max = oil_cfg.get("max_temperature", 373.0)

        # State
        self.temperature = self.T0
        self.oil_temperature = self.oil_T0
        self.peak_temperature = self.T0
        self._sigma = 5.67e-8  # Stefan-Boltzmann constant

    # ------------------------------------------------------------------
    # Heat input
    # ------------------------------------------------------------------
    def heat_input(self, voltage_kv: float, current_ma: float, duration_s: float) -> float:
        """Q = V·I·t  (electrical power converted to heat in Joules)."""
        power_w = voltage_kv * 1000.0 * current_ma / 1000.0
        return power_w * duration_s

    # ------------------------------------------------------------------
    # Temperature evolution
    # ------------------------------------------------------------------
    def update_temperature(self, Q: float, dt: float) -> float:
        """Advance the thermal state by one time-step dt (seconds).

        Radiation cooling is always active (the disc is always hot and always
        spinning). The caller passes Q > 0 during a scan and Q = 0 during the
        inter-scan cooling interval.

        Energy balance: ρ·cp·V·(dT/dt) = Q/dt − σ·ε·A·(T⁴ − T_amb⁴)
        """
        # Radiative cooling — always on, reduced by cooling_efficiency
        Q_rad = (
            self._sigma
            * self.emissivity
            * self.area
            * (self.temperature ** 4 - self.ambient_temperature ** 4)
        )  # W (instantaneous power)
        Q_rad_eff = Q_rad * self.cooling_efficiency  # reduced by degraded cooling

        # Net temperature change: Q_in_per_dt − Q_rad_per_dt
        dT = (Q / dt - Q_rad_eff) * dt / (self.mass * self.cp)
        # = (Q_in - Q_rad * dt) / (m * cp) — keeps units consistent

        self.temperature += dT
        self.temperature = float(np.clip(
            self.temperature, self.ambient_temperature, self.T_max * 1.1
        ))

        # Track running peak (used by Tang 2026 overheating failure mode)
        if self.temperature > self.peak_temperature:
            self.peak_temperature = self.temperature

        # Propagate heat to oil jacket (2-node coupling)
        self._update_oil_temperature(dt)

        return self.temperature

    def _update_oil_temperature(self, dt: float) -> None:
        """2-node lumped thermal coupling: anode → oil → ambient.

        dT_oil/dt ≈ α·(T_anode − T_oil) − β·(T_oil − T_ambient)
        """
        anode_to_oil = self.oil_coupling * (self.temperature - self.oil_temperature)
        oil_to_ambient = self.oil_cooling * self.cooling_efficiency * (
            self.oil_temperature - self.ambient_temperature
        )
        self.oil_temperature += (anode_to_oil - oil_to_ambient) * dt
        self.oil_temperature = float(np.clip(
            self.oil_temperature, self.ambient_temperature, self.oil_T_max
        ))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_normalized_temperature(self) -> float:
        """Anode temperature normalised to [0, 1] between T0 and T_max."""
        return (self.temperature - self.T0) / (self.T_max - self.T0)

    def get_oil_temperature(self) -> float:
        """Current oil jacket temperature (K)."""
        return self.oil_temperature

    def get_peak_temperature(self) -> float:
        """Running peak anode temperature since last reset (K)."""
        return self.peak_temperature

    def is_overheating(self, threshold_fraction: float = 0.85) -> bool:
        """True when anode temperature exceeds threshold_fraction of T_max range.
        (Tang 2026 'overheating' failure mode.)
        """
        threshold = self.T0 + threshold_fraction * (self.T_max - self.T0)
        return self.temperature >= threshold

    def reset(self) -> None:
        """Reset to initial conditions."""
        self.temperature = self.T0
        self.oil_temperature = self.oil_T0
        self.peak_temperature = self.T0
