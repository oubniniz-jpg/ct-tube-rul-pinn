# CT X-Ray Tube RUL — Physics-Informed Deep Learning

Remaining-useful-life (RUL) estimation for CT-scanner X-ray tubes under severe data
scarcity, using a physics-based degradation simulator (digital twin) and
physics-informed deep learning.

This repository accompanies the paper:
**"CT X-ray Tube RUL Estimation Under Data Scarcity in Low-Resource Hospitals"**,
Z. Oubnini, F. Wang — AFRICAI Workshop, MICCAI 2026 (Springer LNCS, vol. 17264).

## Overview
CT X-ray tubes are expensive, and in low-resource hospitals each unplanned failure
causes weeks of scanner downtime. Real hospitals record only a few failure dates, not
continuous sensor data, so standard deep learning cannot be trained directly. This project:
- builds a **physics-based degradation simulator** of the X-ray tube (anode heating,
  filament evaporation, thermal stress, bearing wear, calendar aging);
- calibrates it as a **digital twin** against real failure records from two hospitals;
- uses the simulator to **generate training data** for RUL models;
- benchmarks nine methods, including a **physics-informed LSTM**, with multiple random
  seeds and significance testing.

## Repository structure
