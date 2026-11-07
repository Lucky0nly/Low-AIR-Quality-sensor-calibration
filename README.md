# Low-Cost Air Quality Sensor Calibration (HyPhyCal Framework)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/Platform-ESP32%20%7C%20STM32%20%7C%20Edge%20IoT-green.svg)]()

> **Research Repository:** Physics-Constrained Machine Learning for Low-Cost Particulate Matter ($\text{PM}_{2.5}$) Sensor Calibration under High Humidity and Severe Smog Inversions.

---

## 📌 Overview

Low-cost optical particle counters (OPCs) such as the **Plantower PMS5003** (used in PurpleAir) and **Honeywell HPMA115C0** provide high spatial density at \$20–\$30 per unit, but suffer from severe physical error modes:
- **Hygroscopic growth:** Optical scattering overestimates mass by 150%–300% when relative humidity exceeds 70%.
- **Hysteresis:** Path-dependent moisture retention between deliquescence (wetting) and efflorescence (drying).
- **Internal thermal heating:** Sensor electronics increase chamber temperature by +2.5°C to +4.0°C.
- **Peak attenuation:** Standard regression models clip hazardous pollution spikes ($>100\,\mu\text{g}/\text{m}^3$).

This repository implements **HyPhyCal** (*Hysteresis-Aware Physics-Constrained Hybrid Calibration*), combining an analytical Hänel fluid-mechanics backbone with an asymmetric peak-aware machine learning residual learner and adaptive conformal uncertainty quantification.

---

## 📊 Key Results

| Metric | Raw Sensor | US EPA Standard | Random Forest | Standalone GBDT | **HyPhyCal (Ours)** |
|---|---|---|---|---|---|
| **India PA-II Test RMSE** | $61.02\,\mu\text{g}/\text{m}^3$ | $21.42\,\mu\text{g}/\text{m}^3$ | $2.36\,\mu\text{g}/\text{m}^3$ | $2.27\,\mu\text{g}/\text{m}^3$ | **$2.67\,\mu\text{g}/\text{m}^3$** *(v1)* |
| **Italy Q-SCALE Test RMSE** | $7.96\,\mu\text{g}/\text{m}^3$ | $4.36\,\mu\text{g}/\text{m}^3$ | $0.82\,\mu\text{g}/\text{m}^3$ | $0.81\,\mu\text{g}/\text{m}^3$ | **$0.75\,\mu\text{g}/\text{m}^3$** 🏆 |
| **Saturated Fog RMSE (>85% RH)** | $78.44\,\mu\text{g}/\text{m}^3$ | $24.45\,\mu\text{g}/\text{m}^3$ | $2.02\,\mu\text{g}/\text{m}^3$ | $1.80\,\mu\text{g}/\text{m}^3$ | **$1.54\,\mu\text{g}/\text{m}^3$** 🏆 |
| **Model Size on Disk** | 0 KB | 0.01 KB | 22,037 KB (22 MB) | 3,086 KB (3.1 MB) | **122.7 KB** *(99.4% smaller)* |
| **Inference Latency** | 0.0 µs | 0.05 µs | 6.74 µs | 1.67 µs | **0.42 µs** *(179,000+ samp/s)* |
| **Peak Retention (>60 µg/m³)** | 0.04% | 43.9% | 99.9% | 100.0% | **100.0%** |
| **Conformal Coverage (90% target)** | — | — | — | — | **89.9% / 90.1%** |

---

## 🗂️ Project Structure & Pipeline Roadmap

See [PIPELINE_ROADMAP.md](PIPELINE_ROADMAP.md) for full architectural documentation across all 7 stages:

1. `01_Data_Ingestion_QAQC_and_Feature_Engineering.ipynb` — Dual-channel filtering, dew-point deficit, rate-of-change dynamics.
2. `02_Physics_Backbone_and_Hysteresis_Modeling.ipynb` — Dual-branch Hänel deliquescence-efflorescence optimization.
3. `03_Joint_CoOptimization_and_Residual_Learning.ipynb` — Alternating Coordinate Optimization (ACO) and PINN backpropagation.
4. `04_Comprehensive_Benchmarking_and_Ablation.ipynb` — Full 12-metric evaluation, RH stratification, hysteresis splits, 5 ablations.
5. `05_Uncertainty_Quantification_Adaptive_Conformal.ipynb` — Distribution-free Adaptive Conformal Inference (ACI).
6. `06_Publication_Visualizations_and_Telemetry_Plots.ipynb` — High-resolution 300 DPI publication graphics.
7. `07_Edge_Deployment_and_C_Firmware_Export.ipynb` — Transpilation to zero-dependency ANSI C for ESP32/STM32 microcontrollers.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Lucky0nly/Low-AIR-Quality-sensor-calibration.git
cd Low-AIR-Quality-sensor-calibration

# Install dependencies
pip install -r requirements.txt

# Run the complete benchmark pipeline
python hyphycal_implementation.py
```

---

## 📄 License
This project is licensed under the MIT License — see the LICENSE file for details.
