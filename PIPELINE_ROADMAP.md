# Low-Cost Air Quality Sensor Calibration: End-to-End Research Pipeline & Roadmap

## Project Objective
Transform low-cost optical particulate matter ($\text{PM}_{2.5}$) sensors (e.g., Plantower PMS5003 in PurpleAir, Honeywell HPMA115C0) costing \$20–\$30 into regulatory-grade instruments comparable to \$30,000 Federal Reference Monitors (e.g., Met One BAM 1020 Beta Attenuation Monitors).

The pipeline resolves four fundamental failure modes of cheap optical sensors:
1. **Hygroscopic Deliquescence:** Water vapor condensation causing 150%–300% artificial mass overestimation above 70% RH.
2. **Efflorescence-Deliquescence Hysteresis:** Path-dependent drying vs. wetting cycles causing dynamic calibration drift.
3. **Internal Sensor Heating:** Electronics generating +2.5°C to +4.0°C chamber temperature offsets.
4. **Extreme Winter Smog Clipping:** Traditional linear and tree-based models clipping hazardous pollution peaks ($>100\,\mu\text{g}/\text{m}^3$).

---

## Architecture: HyPhyCal (Hysteresis-Aware Physics-Constrained Hybrid Calibration)

The framework pairs an analytical physical fluid-mechanics backbone with an asymmetric peak-aware machine learning residual learner and adaptive conformal prediction intervals:

$$\hat{y}(t) = \hat{y}_{\text{phys}}(t) + \Delta\hat{y}_{\text{residual}}(t)$$

$$\hat{y}_{\text{phys}}(t) = \frac{\alpha \cdot \text{PM}_{\text{raw}}(t)}{1 + \left[s(t) c_d + (1 - s(t)) c_e\right] \cdot \left(\frac{a_w(t)}{1 - a_w(t)}\right)^{\left[s(t) \gamma_d + (1 - s(t)) \gamma_e\right]}} + \beta_T \cdot (T(t) - 25^\circ\text{C}) + \beta_0$$

Where:
- $a_w(t) = \text{RH}(t) / 100$: Ambient water activity.
- $s(t) = \sigma\left(\kappa \cdot \frac{d\text{RH}}{dt}\right)$: Smooth sigmoid state index separating wetting (deliquescence: $s \to 1$) from drying (efflorescence: $s \to 0$).
- $\Delta\hat{y}_{\text{residual}}(t)$: Machine-learning compensator trained strictly on the physical residual $\epsilon(t) = y_{\text{ref}}(t) - \hat{y}_{\text{phys}}(t)$ using an Asymmetric Peak-Aware Huber Loss ($\tau_q = 0.65$).

---

## Modular Notebook Pipeline Structure

```
📁 notebooks/
│
├── 📓 01_Data_Ingestion_QAQC_and_Feature_Engineering.ipynb
├── 📓 02_Physics_Backbone_and_Hysteresis_Modeling.ipynb
├── 📓 03_Joint_CoOptimization_and_Residual_Learning.ipynb   <-- (Core Algorithmic Engine)
├── 📓 04_Comprehensive_Benchmarking_and_Ablation.ipynb
├── 📓 05_Uncertainty_Quantification_Adaptive_Conformal.ipynb
├── 📓 06_Publication_Visualizations_and_Telemetry_Plots.ipynb
└── 📓 07_Edge_Deployment_and_C_Firmware_Export.ipynb
```

---

### Detailed Stage Specifications

### Stage 1: `01_Data_Ingestion_QAQC_and_Feature_Engineering.ipynb`
* **Input:** `dataset1_purpleair_india.csv` (26,280 raw records), `dataset2_qscale_italy.csv` (5,500 records).
* **QA/QC Protocol:** 
  * Dual-channel PMS5003 channel A vs. B divergence check: purge records where $|\text{PM}_A - \text{PM}_B| > 5.0\,\mu\text{g}/\text{m}^3$ and relative diff $> 20\%$.
  * Physical boundary filters: $\text{PM} \in [0, 1500]\,\mu\text{g}/\text{m}^3$, $\text{RH} \in [0, 100]\%$, $T \in [-40, 60]^\circ\text{C}$.
* **Atmospheric Feature Engineering:**
  * Dew point temperature: $T_{\text{dew}} = T - \frac{100 - \text{RH}}{5}$.
  * Dew point deficit: $\Delta T_{\text{dew}} = T - T_{\text{dew}}$ (condensation proximity).
  * Temporal rates: $\frac{d\text{RH}}{dt} = \text{RH}_t - \text{RH}_{t-1}$, $\frac{dT}{dt} = T_t - T_{t-1}$.
  * Smooth wetting state: $s(t) = \frac{1}{1 + \exp(-\kappa \cdot d\text{RH}/dt)}$.
  * Hygroscopic interaction: $\text{PM}_{\text{raw}} \times (\text{RH}/100)$.
  * Cyclic diurnal and seasonal embeddings ($\sin/\cos$ of hour and month).
* **Splitting:** Strict forward chronological 60% Train / 20% Validation / 20% Test split.
* **Output:** `data/processed_india.parquet`, `data/processed_italy.parquet`.

---

### Stage 2: `02_Physics_Backbone_and_Hysteresis_Modeling.ipynb`
* **Mathematical Formulation:** 8-parameter bounded non-linear least squares optimization via Differential Evolution.
* **Physical Constraints:** Enforce $c_e < c_d$ (particles retain water lower during drying than wetting).
* **Fitted Physical Coefficients:**
  * $\alpha$: Dry aerosol optical-to-mass factor (soot/dust density correction).
  * $c_d, \gamma_d$: Deliquescence swelling amplitude and power exponent.
  * $c_e, \gamma_e$: Efflorescence retention amplitude and power exponent.
  * $\beta_T$: Internal chamber electronic self-heating offset coefficient.
  * $\beta_0$: Photodiode dark-current / chamber zero-scatter baseline.
  * $\kappa$: Sigmoid path-transition sensitivity.
* **Output:** `models/physics_parameters.json`, `data/physics_residuals_*.parquet`.

---

### Stage 3: `03_Joint_CoOptimization_and_Residual_Learning.ipynb`
* **Objective:** Co-train the physical backbone and the residual learner so the model definitively beats Random Forest and GBDT baselines across all metrics.
* **Architectures Evaluated:**
  1. **Alternating Coordinate Optimization (ACO) Hybrid:**
     * *Step A:* Fit Physics on $(y - \Delta\hat{y}_{\text{tree}})$.
     * *Step B:* Train High-Capacity Gradient Booster (80–120 trees, depth 6–8) on $(y - \hat{y}_{\text{phys}})$.
     * *Step C:* Iterate 3 cycles until convergence.
     * *Loss:* Asymmetric Peak-Aware Huber Loss ($\tau_q = 0.65$, weighting $\lambda_1 = 1.5$ for $y > 35.5$, $\lambda_2 = 3.0$ for $y > 150\,\mu\text{g}/\text{m}^3$).
  2. **End-to-End PyTorch PINN:** Differentiable Hänel layer followed by residual Skip-Connection Tabular Blocks trained via joint backpropagation.
* **Output:** `models/hyphycal_best_model.pkl`, `models/pinn_best_model.pt`.

---

### Stage 4: `04_Comprehensive_Benchmarking_and_Ablation.ipynb`
* **Comparative Baselines:** Raw Sensor, US EPA Standard Linear Correction, MLR, Random Forest (80 trees, depth 12), GBDT Standalone (100 trees, depth 8), SVR-RBF.
* **Problem-Specific Metrics:**
  * Global: RMSE, MAE, MBE, $R^2$, NRMSE (%), sMAPE (%).
  * Peak Retention: % of episodes ($>60\,\mu\text{g}/\text{m}^3$) within $\pm 20\%$ regulatory envelope.
  * Stratified Humidity: Error partitioned into $<50\%$, $50\text{--}70\%$, $70\text{--}85\%$ (Deliquescence), and $>85\%$ (Saturated fog).
  * Hysteresis Partition: Error during Wetting ($\frac{d\text{RH}}{dt} > +2\%/\text{hr}$) vs. Drying ($\frac{d\text{RH}}{dt} < -2\%/\text{hr}$).
* **5 Ablation Experiments:**
  1. A1: No Physics Backbone (pure black-box GBDT).
  2. A2: No Hysteresis (single growth curve $c_d = c_e$).
  3. A3: No Asymmetric Loss (symmetric MSE).
  4. A4: No Temporal Rate Features (no $d\text{RH}/dt$, $dT/dt$).
  5. A5: Iterative Co-optimization Impact (1 cycle vs. 3 cycles).
* **Cross-Site Transfer:** 14-day co-location calibration transfer from India to Italy without machine learning retraining.
* **Output:** `results/benchmark_metrics.csv`, `results/stratified_eval.csv`, `results/ablation_results.csv`.

---

### Stage 5: `05_Uncertainty_Quantification_Adaptive_Conformal.ipynb`
* **Method:** Adaptive Conformal Inference (ACI - Gibbs & Candela, 2021).
* **Guarantees:** Distribution-free validity without Gaussian residual assumptions.
* **Adaptation:** Prediction bands dynamically widen during high-volatility fog/inversion events and narrow during stable ambient conditions.
* **Empirical Validation:** Verify 90% and 95% target coverage convergence on chronological test splits.
* **Out-of-Distribution (OOD) Detection:** Automated flagging when interval width exceeds sensor operational limits ($W_t > \tau_{\text{OOD}}$).
* **Output:** `results/conformal_intervals.parquet`, `results/coverage_summary.json`.

---

### Stage 6: `06_Publication_Visualizations_and_Telemetry_Plots.ipynb`
* **Outputs 8 High-Resolution 300 DPI Publication Graphics:**
  * `fig1_benchmark_scoreboard.png`: Model comparison across metrics.
  * `fig2_time_series_overlay.png`: 14-day winter time series tracking vs. BAM ground truth with $\pm 20\%$ regulatory envelopes.
  * `fig3_parity_scatter_plots.png`: 1:1 parity scatter plots with empirical regression slopes.
  * `fig4_rh_stratified_bias.png`: Mean Bias Error across humidity brackets proving deliquescence neutralization.
  * `fig5_hysteresis_loops.png`: Wetting vs. drying path error decomposition.
  * `fig6_ablation_waterfall.png`: Contribution ranking of individual architectural components.
  * `fig7_conformal_prediction_bands.png`: Continuous prediction intervals with coverage hit/miss markers.
  * `fig8_edge_pareto_frontier.png`: Inference latency vs. accuracy vs. memory footprint.

---

### Stage 7: `07_Edge_Deployment_and_C_Firmware_Export.ipynb`
* **Target Hardware:** Espressif ESP32 (520 KB SRAM), STMicroelectronics STM32 (ARM Cortex-M4/M7).
* **Transpilation:** Export the complete two-stage hybrid model into zero-dependency ANSI C:
  * `firmware/hyphycal_model.h`: Parameter definitions and function headers.
  * `firmware/hyphycal_model.c`: Direct single-precision `float32` inference code.
  * `firmware/esp32_example.ino`: Arduino/ESP-IDF sample sketch reading sensor UART and streaming calibrated concentrations.
* **Parity Verification:** Numerical verification ensuring $|\hat{y}_{\text{Python}} - \hat{y}_{\text{C}}| < 10^{-5}\,\mu\text{g}/\text{m}^3$.
* **Footprint Target:** Model flash footprint $< 130\,\text{KB}$, execution time $< 10\,\mu\text{s}$ per sample.

---

## Benchmark Datasets Reference
1. **Dataset 1 (India Multi-Site Network):**
   * Co-located PurpleAir PA-II (dual Plantower PMS5003 channels A & B) adjacent to Met One BAM 1020 / BAM 1022 regulatory monitors.
   * Sites: Delhi (megacity winter inversions), Hamirpur (sub-Himalayan background), Bengaluru (tropical urban plateau).
   * Range: 4.0 to 480.0 $\mu\text{g}/\text{m}^3$. Cleaned: 25,829 valid hourly records.
2. **Dataset 2 (Q-SCALE Turin Benchmark, Italy):**
   * 4$\times$ Honeywell HPMA115C0 sensors co-located at ARPA Piemonte official station (Torino Rubino).
   * Range: 2.0 to 115.0 $\mu\text{g}/\text{m}^3$. Cleaned: 5,500 hourly records.
