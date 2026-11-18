"""
Calendar-Age Degradation Model for CT-Scanner X-Ray Tube
=========================================================
INNOVATION 1 (thesis / journal extension of the AFRICAI conference paper).

The other degradation sub-models (filament evaporation, thermal fatigue, bearing
wear) are all USAGE-driven: they only advance while the tube is scanning. That is
correct for a busy scanner, but it makes a lightly-used tube effectively immortal
(fixed scans-to-failure / a very low scan rate = decades of calendar life). Real
X-ray tubes do NOT behave that way. They also age with WALL-CLOCK time whether or
not they are used, through time-dependent mechanisms:

  - vacuum / gas-load rise (continuous outgassing and gas permeation),
  - bearing-lubricant oxidation and dry-out,
  - high-voltage insulation (oil, epoxy) and seal aging,
  - getter saturation.

This sub-model adds that missing calendar-age pathway. It competes with the
usage-driven modes (weakest-link / competing risks): a tube fails by whichever
pathway reaches its limit first.

  - High-workload hospitals (e.g. CHU Ibn Rochd, ~300 scans/day): the tube dies
    by USAGE (filament) at ~395 days before calendar age matters.
  - Low-workload hospitals (e.g. Hopital Civil Tetouan, ~12 scans/day): usage
    alone would give decades, so calendar AGE dominates and the tube fails after
    a few years, matching the observed ~931-day inter-failure interval.

A single calibrated simulator can then reproduce real tube lifetimes across the
whole workload range, which the usage-only model of the conference paper could
not (it over-predicted low-workload life by ~15x).

Damage model (competing risks):
    d_age(t) = integral over calendar time of  (1 / L_tube) * accel(T) dt
  where
    L_tube      = per-tube characteristic calendar life (sampled around
                  `characteristic_life_days` with `variability_std` spread),
    accel(T)    = 1 + thermal_acceleration * T_norm  (hotter operation ages
                  seals, oil and vacuum faster; T_norm in [0, 1]).
  The tube fails by age when the accumulated age-damage reaches 1.0.
"""

import numpy as np


class AgingModel:
    """Calendar-time (idle-inclusive) degradation of the X-ray tube."""

    def __init__(self, config: dict):
        aging = config["physics"].get("aging", {})
        self.characteristic_life_days = aging.get("characteristic_life_days", 950.0)
        self.thermal_acceleration = aging.get("thermal_acceleration", 0.4)
        self.variability_std = aging.get("variability_std", 0.15)
        self.min_life_fraction = aging.get("min_life_fraction", 0.4)

        anode = config["physics"]["anode"]
        self.T_ref = anode["initial_temperature"]
        self.T_max = anode["max_temperature"]

        self.tube_life_days = self.characteristic_life_days
        self.age_damage = 0.0

    def reset(self) -> None:
        """Resample this tube's characteristic calendar life and clear damage.

        Per-tube variability makes a hospital show a realistic spread of ages
        rather than every tube failing on exactly the same day.
        """
        factor = np.random.normal(1.0, self.variability_std)
        factor = max(self.min_life_fraction, factor)
        self.tube_life_days = self.characteristic_life_days * factor
        self.age_damage = 0.0

    def step(self, temperature: float, days_elapsed: float) -> None:
        """Advance calendar-age damage by `days_elapsed` real calendar days.

        `days_elapsed` is the real wall-clock time represented by one scan cycle,
        i.e. 1 / scans_per_day, so this accrues whether the tube is busy or idle.
        """
        t_norm = max(0.0, (temperature - self.T_ref) / (self.T_max - self.T_ref))
        accel = 1.0 + self.thermal_acceleration * t_norm
        self.age_damage += (days_elapsed / self.tube_life_days) * accel

    def get_health_indicator(self) -> float:
        """Return calendar-age health as a fraction [0, 1]. 1.0 = new, 0.0 = aged out."""
        return float(np.clip(1.0 - self.age_damage, 0.0, 1.0))

    def is_failed(self) -> bool:
        """True when accumulated calendar-age damage has reached the failure limit."""
        return self.age_damage >= 1.0
