"""
Degradation Engine - Orchestrates all physics models to generate
synthetic degradation trajectories for CT-scanner X-ray tubes.

Combines:
  - ThermalModel       (anode body + oil jacket — Zhou 2024)
  - FilamentModel      (Richardson-Dushman emission — filament evaporation)
  - StressModel        (thermal fatigue: σ = E·α·ΔT — Equation 3)
  - MechanicalModel    (rotor bearing wear — Shamim 2024)
  - Vacuum / arcing    (Tang 2026 arcing failure mode)

Output feature set (12 channels) is a SUPERSET of all competing literature:
  anode_temperature, oil_temperature, tube_voltage_kv, tube_current_ma,
  filament_current_a, filament_diameter_m, vacuum_pressure_pa,
  rotor_vibration_rms, thermal_stress_pa, cumulative_stress_pa,
  scan_cycle, ambient_temperature_k

Per-tube output also tags the failure mode (Tang 2026 taxonomy):
  overheating / arcing / filament_failure / rotor_bearing_failure
"""

import numpy as np
import pandas as pd

from .thermal_model import ThermalModel
from .filament_model import FilamentModel
from .stress_model import StressModel
from .mechanical_model import MechanicalModel
from .aging_model import AgingModel


class DegradationEngine:
    """Main simulation engine that generates synthetic tube degradation data."""

    def __init__(self, config: dict):
        self.config = config
        self.sim_config = config["simulation"]
        self.physics_config = config["physics"]

        self.thermal = ThermalModel(config)
        self.filament = FilamentModel(config)
        self.stress = StressModel(config)
        self.mechanical = MechanicalModel(config)
        self.aging = AgingModel(config)

        vacuum_cfg = self.physics_config["vacuum"]
        self.vacuum_initial = vacuum_cfg["initial_pressure"]
        self.vacuum_failure = vacuum_cfg["failure_pressure"]
        self.outgassing_rate = vacuum_cfg["outgassing_rate"]
        self.arcing_threshold = vacuum_cfg.get("arcing_threshold", 1.0e-4)
        self.arcing_base_prob = vacuum_cfg.get("arcing_base_prob", 1.0e-5)

        self.ambient_temperature = self.sim_config.get("ambient_temperature", 305.0)

    # ------------------------------------------------------------------
    # Single tube simulation
    # ------------------------------------------------------------------
    def simulate_single_tube(self, tube_id: int = 0, seed: int | None = None) -> pd.DataFrame:
        """Simulate the full lifecycle of one X-ray tube.

        Returns a DataFrame with columns matching `model.feature_names` in config.yaml
        plus metadata columns: tube_id, scan_cycle, rul, health_indicator, failure_mode.
        """
        if seed is not None:
            np.random.seed(seed)

        self.thermal.reset()
        self.filament.reset()
        self.stress.reset()
        self.mechanical.reset()
        self.aging.reset()

        max_cycles = self.sim_config["max_cycles"]
        scan_duration = self.sim_config["scan_duration"]
        cooling_duration = self.sim_config["cooling_duration"]
        # Real calendar time represented by one scan cycle = 1 / scans_per_day.
        # Drives the calendar-age model so idle time ages the tube too (Innovation 1).
        scans_per_day = self.sim_config.get("scans_per_day_mean", 300.0)
        days_per_cycle = 1.0 / max(float(scans_per_day), 1e-6)

        # Per-tube variability: randomize operating regime
        # Clinical range (CHU Ibn Rochd workload — chest/abdomen/head protocols).
        # 100 kV floor avoids non-physical low-power regimes where evaporation
        # would never accumulate to failure within realistic hospital timescales.
        base_voltage = np.random.uniform(100, 140)  # kV
        base_current = np.random.uniform(150, 400)  # mA

        # Per-tube ambient temperature jitter (Waid 2023: 20–30 °C at CHU Ibn Rochd)
        ambient = self.ambient_temperature + np.random.uniform(-5.0, 5.0)

        vacuum_pressure = self.vacuum_initial
        arcing_events = 0
        failure_mode: str | None = None

        # Sustained-overheating counter: overheating failure only declared after
        # the temperature exceeds the threshold for many consecutive cycles.
        # A single hot cycle does not destroy the tube — accumulated damage does.
        # 1000 cycles ≈ 3.3 days at 300 scans/day — realistic for thermal runaway.
        overheat_consecutive = 0
        OVERHEAT_FAILURE_CYCLES = 1000

        records: list[dict] = []
        for cycle in range(max_cycles):
            # ---- Scan parameters with realistic jitter ----
            voltage = np.clip(base_voltage + np.random.normal(0, 5), 60, 150)
            current = np.clip(base_current + np.random.normal(0, 20), 50, 500)

            # ---- Scan: heating input + simultaneous radiation (always-on) ----
            # Radiation cooling is always active (disc spins continuously).
            # update_temperature now handles both Q_in and Q_rad in one call.
            Q = self.thermal.heat_input(voltage, current, scan_duration)
            temp = self.thermal.update_temperature(Q, scan_duration)

            # ---- Inter-scan cooling interval (Q=0, radiation only) ----
            self.thermal.update_temperature(0.0, cooling_duration)

            # ---- Filament degradation + emission ----
            self.filament.evaporation_step(temp, scan_duration)
            filament_current = self.filament.emission_current(temp)

            # ---- Thermal stress accumulation ----
            self.stress.accumulate_cycle(temp)
            current_stress = self.stress.compute_stress(temp)

            # ---- Rotor bearing wear / vibration ----
            rotor_rms = self.mechanical.update(cycle, temp, dt=scan_duration)

            # ---- Vacuum degradation ----
            vacuum_pressure += self.outgassing_rate * (1.0 + temp / 3000.0)

            # ---- Arcing probability (Tang 2026 failure mode) ----
            arc_factor = max(0.0, vacuum_pressure - self.arcing_threshold) / self.arcing_threshold
            arc_prob = self.arcing_base_prob * (1.0 + 50.0 * arc_factor)
            if np.random.random() < arc_prob:
                arcing_events += 1

            # ---- Calendar-age degradation (Innovation 1): accrues with real
            #      calendar time, so idle low-workload tubes still age ----
            self.aging.step(temp, days_per_cycle)

            # ---- Health indicators ----
            filament_health = self.filament.get_health_indicator()
            thermal_health = 1.0 - self.thermal.get_normalized_temperature()
            stress_health = 1.0 - self.stress.get_fatigue_indicator()
            rotor_health = self.mechanical.get_health_indicator()
            vacuum_health = float(np.clip(
                1.0 - (vacuum_pressure - self.vacuum_initial)
                / max(self.vacuum_failure - self.vacuum_initial, 1e-12),
                0.0, 1.0,
            ))
            aging_health = self.aging.get_health_indicator()
            health = min(filament_health, thermal_health, stress_health,
                         rotor_health, vacuum_health, aging_health)

            records.append({
                "tube_id": tube_id,
                "scan_cycle": cycle,
                "anode_temperature": temp,
                "oil_temperature": self.thermal.get_oil_temperature(),
                "tube_voltage_kv": voltage,
                "tube_current_ma": current,
                "filament_current_a": filament_current,
                "filament_diameter_m": self.filament.diameter,
                "vacuum_pressure_pa": vacuum_pressure,
                "rotor_vibration_rms": rotor_rms,
                "thermal_stress_pa": current_stress,
                "cumulative_stress_pa": self.stress.cumulative_stress,
                "ambient_temperature_k": ambient,
                "arcing_events": arcing_events,
                "health_indicator": health,
            })

            # ---- Failure mode checks (Tang 2026 + CHU Ibn Rochd taxonomy) ----
            if self.filament.is_failed():
                failure_mode = "filament_failure"
                break
            if self.mechanical.is_failed():
                failure_mode = "rotor_bearing_failure"
                break
            if vacuum_pressure >= self.vacuum_failure:
                failure_mode = "arcing"
                break
            # Calendar-age failure (Innovation 1): dominates at low workload
            if self.aging.is_failed():
                failure_mode = "age_degradation"
                break
            # Overheating only declared after sustained high-temperature operation
            if self.thermal.is_overheating(threshold_fraction=0.85):
                overheat_consecutive += 1
                if overheat_consecutive >= OVERHEAT_FAILURE_CYCLES:
                    failure_mode = "overheating"
                    break
            else:
                overheat_consecutive = 0

        if len(records) == 0:
            return pd.DataFrame()

        # Compute RUL without building full DataFrame (memory-efficient)
        max_cycle = records[-1]["scan_cycle"]
        failure_str = failure_mode if failure_mode else "censored"
        for r in records:
            r["rul"] = max_cycle - r["scan_cycle"]
            r["failure_mode"] = failure_str

        return records  # return raw list — caller decides how to persist

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------
    def generate_dataset(self, num_tubes: int | None = None, save_path: str | None = None) -> pd.DataFrame:
        """Generate synthetic degradation data for multiple tubes."""
        if num_tubes is None:
            num_tubes = self.sim_config["num_tubes"]

        all_data = []
        for i in range(num_tubes):
            tube_data = self.simulate_single_tube(tube_id=i, seed=i * 42)
            if tube_data:  # simulate_single_tube returns a raw list of dict rows
                all_data.append(pd.DataFrame(tube_data))
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{num_tubes} tube histories")

        dataset = pd.concat(all_data, ignore_index=True)

        if save_path:
            dataset.to_csv(save_path, index=False)
            print(f"  Dataset saved to {save_path}")

        # Quick failure-mode summary (helps calibration against CHU Ibn Rochd Pareto)
        mode_counts = (
            dataset.drop_duplicates("tube_id")["failure_mode"]
            .value_counts()
            .to_dict()
        )
        print(f"  Failure-mode distribution: {mode_counts}")

        return dataset
