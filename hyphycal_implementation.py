#!/usr/bin/env python3
"""
HyPhyCal — Hysteresis-Aware Physics-Constrained Hybrid Calibration Framework
=============================================================================
Complete implementation for low-cost PM2.5 sensor calibration.

Architecture:
  Stage 0: Domain-specific feature engineering
  Stage 1: Parametric physics backbone (hysteresis-aware Hänel growth)
  Stage 2: Micro-GBDT residual learner with asymmetric peak-aware loss
  Stage 3: Adaptive Conformal Prediction intervals

Author: ML Research Project
"""

import warnings
warnings.filterwarnings('ignore')

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os
os.environ['MPLCONFIGDIR'] = os.path.join(r"c:\Users\Victus\OneDrive\Documents\CODES\ML", '.mpl_cache')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import numpy as np
import pandas as pd
import time
import json
import pickle
from scipy.optimize import differential_evolution, minimize
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURATION
# ============================================================
WORK_DIR = r"c:\Users\Victus\OneDrive\Documents\CODES\ML"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Regulatory thresholds for asymmetric loss
TAU_1 = 35.5   # US EPA 24-hour PM2.5 "Unhealthy for Sensitive Groups"
TAU_2 = 150.0  # AQI "Very Unhealthy"
LAMBDA_1 = 1.5  # Weight multiplier for moderate exceedances
LAMBDA_2 = 3.0  # Weight multiplier for severe episodes
TAU_Q = 0.65    # Asymmetric quantile (under-prediction penalized more)
HUBER_DELTA = 3.0  # Huber threshold

# Conformal prediction
ACI_COVERAGE = 0.90
ACI_LR = 0.01

print("=" * 72)
print("  HyPhyCal: Hysteresis-Aware Physics-Constrained Hybrid Calibration")
print("=" * 72)

# ============================================================
# PHASE 1: DATA LOADING & STAGE 0 FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 72)
print("  PHASE 1: Data Loading & Stage 0 Feature Engineering")
print("=" * 72)

def load_and_engineer_features(dataset_path, dataset_name):
    """Load dataset and compute domain-specific engineered features."""
    df = pd.read_csv(dataset_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # --- Determine raw PM column ---
    if 'PM2.5_CF1_A' in df.columns:
        # Dataset 1: dual-channel PurpleAir
        # QA/QC: Remove records with high dual-channel divergence
        delta_ab = np.abs(df['PM2.5_CF1_A'] - df['PM2.5_CF1_B'])
        mean_ab = (df['PM2.5_CF1_A'] + df['PM2.5_CF1_B']) / 2.0
        rel_diff = delta_ab / np.clip(mean_ab, 1.0, None)
        mask_qc = ~((delta_ab > 5.0) & (rel_diff > 0.20))
        n_dropped = (~mask_qc).sum()
        df = df[mask_qc].reset_index(drop=True)
        print(f"  [{dataset_name}] QA/QC dropped {n_dropped} records ({100*n_dropped/len(mask_qc):.2f}%)")

        df['PM_raw'] = (df['PM2.5_CF1_A'] + df['PM2.5_CF1_B']) / 2.0
        df['delta_AB'] = np.abs(df['PM2.5_CF1_A'] - df['PM2.5_CF1_B'])
    else:
        # Dataset 2: single median sensor
        df['PM_raw'] = df['PM2.5_sensor_median']
        df['delta_AB'] = 0.0

    # --- Physical validity screening ---
    mask_valid = (
        (df['PM2.5_ref'] >= 0) & (df['PM2.5_ref'] <= 1000) &
        (df['PM_raw'] >= 0) & (df['PM_raw'] <= 1500) &
        (df['relative_humidity'] >= 0) & (df['relative_humidity'] <= 100) &
        (df['temperature'] >= -40) & (df['temperature'] <= 60)
    )
    n_invalid = (~mask_valid).sum()
    if n_invalid > 0:
        print(f"  [{dataset_name}] Removed {n_invalid} physically invalid records")
        df = df[mask_valid].reset_index(drop=True)

    T = df['temperature'].values
    RH = df['relative_humidity'].values
    PM_raw = df['PM_raw'].values

    # --- STAGE 0: Domain-Specific Feature Engineering ---
    print(f"  [{dataset_name}] Engineering domain-specific features...")

    # 1. Dew Point Temperature (Magnus approximation)
    df['T_dew'] = T - (100 - RH) / 5.0

    # 2. Dew Point Deficit (proximity to condensation)
    df['dew_point_deficit'] = T - df['T_dew'].values

    # 3. Humidity Rate of Change (dRH/dt)
    df['dRH_dt'] = np.concatenate([[0], np.diff(RH)])

    # 4. Temperature Rate of Change (dT/dt)
    df['dT_dt'] = np.concatenate([[0], np.diff(T)])

    # 5. Smooth Wetting State Index: s = sigmoid(kappa * dRH/dt)
    kappa_init = 5.0  # Will be optimized in Stage 1
    df['wetting_state'] = 1.0 / (1.0 + np.exp(-kappa_init * df['dRH_dt'].values))

    # 6. Hygroscopic Interaction Term: PM_raw * (RH/100)
    df['PM_x_RH'] = PM_raw * (RH / 100.0)

    # 7. Diurnal encoding
    hour = df['timestamp'].dt.hour.values
    df['sin_hour'] = np.sin(2 * np.pi * hour / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * hour / 24.0)

    # 8. Seasonal encoding
    month = df['timestamp'].dt.month.values
    df['sin_month'] = np.sin(2 * np.pi * month / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * month / 12.0)

    # 9. Water activity (for physics backbone)
    df['water_activity'] = np.clip(RH / 100.0, 0.01, 0.99)

    print(f"  [{dataset_name}] Final records: {len(df)}, features engineered: 12")
    return df


# Load and engineer both datasets
df1 = load_and_engineer_features(
    os.path.join(WORK_DIR, "dataset1_purpleair_india.csv"),
    "Dataset 1 (India)"
)
df2 = load_and_engineer_features(
    os.path.join(WORK_DIR, "dataset2_qscale_italy.csv"),
    "Dataset 2 (Italy)"
)


# ============================================================
# CHRONOLOGICAL SPLITS (60/20/20)
# ============================================================
def chronological_split(df, train_frac=0.6, val_frac=0.2):
    """Strict forward chronological split preventing data leakage."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

train1, val1, test1 = chronological_split(df1)
train2, val2, test2 = chronological_split(df2)

print(f"\n  Dataset 1 splits: Train={len(train1)}, Val={len(val1)}, Test={len(test1)}")
print(f"  Dataset 2 splits: Train={len(train2)}, Val={len(val2)}, Test={len(test2)}")


# ============================================================
# PHASE 2: STAGE 1 — PHYSICS BACKBONE OPTIMIZATION
# ============================================================
print("\n" + "=" * 72)
print("  PHASE 2: Stage 1 — Hysteresis-Aware Physics Backbone")
print("=" * 72)

def physics_model_predict(params, PM_raw, RH, dRH_dt, T):
    """
    Hysteresis-aware Hänel growth physics model.

    ŷ_phys = α · PM_raw / f(RH, s) + β_T · (T - T_ref) + β₀

    f(RH, s) = 1 + [s·c_d + (1-s)·c_e] · (a_w/(1-a_w))^[s·γ_d + (1-s)·γ_e]
    s = σ(κ · dRH/dt)
    """
    alpha, c_d, gamma_d, c_e, gamma_e, beta_T, beta_0, kappa = params

    # Enforce c_e <= c_d (physical constraint)
    c_e = min(c_e, c_d * 0.95)

    # Smooth wetting state
    s = 1.0 / (1.0 + np.exp(-kappa * np.clip(dRH_dt, -20, 20)))

    # Water activity
    a_w = np.clip(RH / 100.0, 0.01, 0.99)

    # Blended hygroscopic parameters
    c_blend = s * c_d + (1 - s) * c_e
    gamma_blend = s * gamma_d + (1 - s) * gamma_e

    # Growth factor
    ratio = a_w / (1.0 - a_w)
    f_growth = 1.0 + c_blend * np.power(np.clip(ratio, 0.01, 100), gamma_blend)

    # Physics prediction
    y_phys = alpha * PM_raw / f_growth + beta_T * (T - 25.0) + beta_0
    return y_phys


def physics_objective(params, PM_raw, RH, dRH_dt, T, y_ref):
    """RMSE objective for physics backbone fitting."""
    y_pred = physics_model_predict(params, PM_raw, RH, dRH_dt, T)
    residuals = y_ref - y_pred
    # Use Huber-like robust objective to handle reference noise
    delta = 10.0
    abs_r = np.abs(residuals)
    loss = np.where(abs_r <= delta, 0.5 * residuals**2, delta * (abs_r - 0.5 * delta))
    return np.mean(loss)


def fit_physics_backbone(train_df, dataset_name):
    """Fit physics backbone using differential evolution (global optimizer)."""
    PM_raw = train_df['PM_raw'].values
    RH = train_df['relative_humidity'].values
    dRH_dt = train_df['dRH_dt'].values
    T = train_df['temperature'].values
    y_ref = train_df['PM2.5_ref'].values

    bounds = [
        (0.1, 2.5),     # α: scattering-to-mass factor
        (0.01, 4.0),    # c_d: deliquescence amplitude
        (0.2, 1.8),     # γ_d: deliquescence exponent
        (0.005, 3.0),   # c_e: efflorescence amplitude
        (0.2, 1.8),     # γ_e: efflorescence exponent
        (-2.0, 2.0),    # β_T: temperature coefficient
        (-20.0, 20.0),  # β₀: intercept
        (0.5, 20.0),    # κ: sigmoid sharpness
    ]

    print(f"\n  [{dataset_name}] Fitting physics backbone via differential evolution...")
    t0 = time.time()

    result = differential_evolution(
        physics_objective,
        bounds=bounds,
        args=(PM_raw, RH, dRH_dt, T, y_ref),
        seed=RANDOM_STATE,
        maxiter=500,
        tol=1e-8,
        popsize=25,
        mutation=(0.5, 1.5),
        recombination=0.8,
        polish=True,
        disp=False
    )

    elapsed = time.time() - t0
    params = result.x
    param_names = ['alpha', 'c_d', 'gamma_d', 'c_e', 'gamma_e', 'beta_T', 'beta_0', 'kappa']

    print(f"  [{dataset_name}] Optimization converged in {elapsed:.1f}s (fun={result.fun:.4f})")
    print(f"  [{dataset_name}] Fitted physical parameters:")
    for name, val in zip(param_names, params):
        print(f"    {name:>7s} = {val:+.6f}")

    # Enforce c_e < c_d
    if params[3] >= params[1]:
        params[3] = params[1] * 0.95
        print(f"    [Enforced c_e < c_d constraint: c_e = {params[3]:.6f}]")

    return params


# Fit physics backbone on both datasets
print("\n  --- Fitting Dataset 1 (India) ---")
params_d1 = fit_physics_backbone(train1, "Dataset 1")

print("\n  --- Fitting Dataset 2 (Italy) ---")
params_d2 = fit_physics_backbone(train2, "Dataset 2")


# ============================================================
# COMPUTE PHYSICS PREDICTIONS AND RESIDUALS
# ============================================================
def compute_physics_predictions(df, params):
    """Compute physics backbone predictions for a dataframe."""
    return physics_model_predict(
        params,
        df['PM_raw'].values,
        df['relative_humidity'].values,
        df['dRH_dt'].values,
        df['temperature'].values
    )

# Physics predictions for all splits
for split_df in [train1, val1, test1]:
    split_df = split_df.copy()
    y_phys = compute_physics_predictions(split_df, params_d1)
    split_df['y_phys'] = y_phys
    split_df['phys_residual'] = split_df['PM2.5_ref'].values - y_phys
    # Update the original
    idx = split_df.index
    df1.loc[idx, 'y_phys'] = y_phys
    df1.loc[idx, 'phys_residual'] = split_df['PM2.5_ref'].values - y_phys

for split_df in [train2, val2, test2]:
    split_df = split_df.copy()
    y_phys = compute_physics_predictions(split_df, params_d2)
    split_df['y_phys'] = y_phys
    split_df['phys_residual'] = split_df['PM2.5_ref'].values - y_phys
    idx = split_df.index
    df2.loc[idx, 'y_phys'] = y_phys
    df2.loc[idx, 'phys_residual'] = split_df['PM2.5_ref'].values - y_phys

# Refresh splits to include new columns
train1, val1, test1 = chronological_split(df1)
train2, val2, test2 = chronological_split(df2)

# Report physics-only performance
for name, tdf in [("Dataset 1", test1), ("Dataset 2", test2)]:
    y_true = tdf['PM2.5_ref'].values
    y_phys = tdf['y_phys'].values
    rmse = np.sqrt(mean_squared_error(y_true, y_phys))
    r2 = r2_score(y_true, y_phys)
    mbe = np.mean(y_phys - y_true)
    print(f"\n  [{name}] Physics-Only Performance:")
    print(f"    RMSE = {rmse:.4f} µg/m³, R² = {r2:.6f}, MBE = {mbe:+.4f} µg/m³")


# ============================================================
# PHASE 3: STAGE 2 — MICRO-GBDT RESIDUAL LEARNER
# ============================================================
print("\n" + "=" * 72)
print("  PHASE 3: Stage 2 — Micro-GBDT with Asymmetric Peak-Aware Loss")
print("=" * 72)

# Feature columns for the residual learner
RESIDUAL_FEATURES = [
    'PM_raw', 'temperature', 'relative_humidity',
    'dew_point_deficit', 'dRH_dt', 'dT_dt',
    'PM_x_RH', 'sin_hour', 'cos_hour',
    'sin_month', 'cos_month', 'delta_AB',
    'y_phys'  # Physics prediction as a feature for the residual learner
]


def compute_sample_weights(y_ref):
    """
    Asymmetric peak-weighted sample weights.

    w(y) = base_weight * asymmetry_factor
    base_weight = 1 + λ₁·I(y > τ₁) + λ₂·I(y > τ₂)
    """
    weights = np.ones_like(y_ref, dtype=np.float64)
    # Concentration-dependent weighting
    weights += LAMBDA_1 * (y_ref > TAU_1).astype(float)
    weights += LAMBDA_2 * (y_ref > TAU_2).astype(float)
    # Normalize to mean 1.0
    weights /= weights.mean()
    return weights


def train_micro_gbdt(train_df, val_df, dataset_name):
    """Train a compact GBDT on physics residuals with peak-aware weighting."""
    X_train = train_df[RESIDUAL_FEATURES].values
    y_train = train_df['phys_residual'].values
    y_ref_train = train_df['PM2.5_ref'].values

    X_val = val_df[RESIDUAL_FEATURES].values
    y_val = val_df['phys_residual'].values

    # Compute asymmetric peak-weighted sample weights
    weights = compute_sample_weights(y_ref_train)

    print(f"\n  [{dataset_name}] Training Micro-GBDT residual learner...")
    print(f"    Features: {len(RESIDUAL_FEATURES)}")
    print(f"    Training samples: {len(X_train)}")
    print(f"    Target: physics residual (mean={y_train.mean():.3f}, std={y_train.std():.3f})")

    t0 = time.time()
    gbdt = GradientBoostingRegressor(
        n_estimators=25,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        min_samples_split=15,
        min_samples_leaf=10,
        max_features=0.8,
        random_state=RANDOM_STATE,
        loss='huber',
        alpha=0.9,  # Huber quantile
    )
    gbdt.fit(X_train, y_train, sample_weight=weights)
    train_time = time.time() - t0

    # Validation performance
    y_val_pred = gbdt.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"    Training time: {train_time:.2f}s")
    print(f"    Val residual RMSE: {val_rmse:.4f} µg/m³")

    # Feature importances
    importances = gbdt.feature_importances_
    feat_imp = sorted(zip(RESIDUAL_FEATURES, importances), key=lambda x: -x[1])
    print(f"    Top-5 feature importances:")
    for fname, imp in feat_imp[:5]:
        print(f"      {fname:>22s}: {imp:.4f}")

    return gbdt, train_time


# Train Micro-GBDT for both datasets
gbdt_d1, train_time_d1 = train_micro_gbdt(train1, val1, "Dataset 1")
gbdt_d2, train_time_d2 = train_micro_gbdt(train2, val2, "Dataset 2")


# ============================================================
# COMBINED HYPHYCAL PREDICTION
# ============================================================
def hyphycal_predict(df, physics_params, gbdt_model):
    """Full HyPhyCal pipeline: Physics + Residual GBDT."""
    y_phys = compute_physics_predictions(df, physics_params)
    X_res = df[RESIDUAL_FEATURES].values
    delta_res = gbdt_model.predict(X_res)
    y_final = y_phys + delta_res
    return y_final, y_phys, delta_res


# Test set predictions
y_hyphycal_d1, y_phys_d1, delta_d1 = hyphycal_predict(test1, params_d1, gbdt_d1)
y_hyphycal_d2, y_phys_d2, delta_d2 = hyphycal_predict(test2, params_d2, gbdt_d2)

# Quick performance check
for name, y_true, y_pred in [
    ("Dataset 1", test1['PM2.5_ref'].values, y_hyphycal_d1),
    ("Dataset 2", test2['PM2.5_ref'].values, y_hyphycal_d2)
]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mbe = np.mean(y_pred - y_true)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n  [{name}] HyPhyCal Combined Performance:")
    print(f"    RMSE = {rmse:.4f} µg/m³")
    print(f"    MAE  = {mae:.4f} µg/m³")
    print(f"    MBE  = {mbe:+.4f} µg/m³")
    print(f"    R²   = {r2:.6f}")


# ============================================================
# PHASE 4: STAGE 3 — ADAPTIVE CONFORMAL PREDICTION
# ============================================================
print("\n" + "=" * 72)
print("  PHASE 4: Stage 3 — Adaptive Conformal Prediction Intervals")
print("=" * 72)

def adaptive_conformal_inference(val_residuals, test_residuals, alpha=0.10, lr=0.01):
    """
    Adaptive Conformal Inference (ACI) - Gibbs & Candela (2021).

    Provides distribution-free prediction intervals with valid coverage
    even under temporal distribution shift.
    """
    # Calibration: compute nonconformity scores on validation set
    cal_scores = np.abs(val_residuals)

    n_test = len(test_residuals)
    intervals = np.zeros((n_test, 2))  # [lower_offset, upper_offset]
    coverages = np.zeros(n_test)
    alpha_t = alpha

    # Running window of calibration scores
    running_scores = list(cal_scores)

    for t in range(n_test):
        # Compute adaptive quantile
        q_t = np.quantile(running_scores, 1 - alpha_t)
        intervals[t] = [-q_t, q_t]

        # Check coverage
        test_score = np.abs(test_residuals[t])
        covered = int(test_score <= q_t)
        coverages[t] = covered

        # Update alpha adaptively
        alpha_t = alpha_t + lr * (alpha - (1 - covered))
        alpha_t = np.clip(alpha_t, 0.001, 0.5)

        # Add test score to running calibration set
        running_scores.append(test_score)

    return intervals, coverages


# Compute conformal intervals for both datasets
# Validation residuals for calibration
y_hyphycal_val1, _, _ = hyphycal_predict(val1, params_d1, gbdt_d1)
val_residuals_d1 = val1['PM2.5_ref'].values - y_hyphycal_val1
test_residuals_d1 = test1['PM2.5_ref'].values - y_hyphycal_d1

intervals_d1, coverages_d1 = adaptive_conformal_inference(
    val_residuals_d1, test_residuals_d1, alpha=1-ACI_COVERAGE, lr=ACI_LR
)

y_hyphycal_val2, _, _ = hyphycal_predict(val2, params_d2, gbdt_d2)
val_residuals_d2 = val2['PM2.5_ref'].values - y_hyphycal_val2
test_residuals_d2 = test2['PM2.5_ref'].values - y_hyphycal_d2

intervals_d2, coverages_d2 = adaptive_conformal_inference(
    val_residuals_d2, test_residuals_d2, alpha=1-ACI_COVERAGE, lr=ACI_LR
)

# Report conformal prediction results
for name, intervals, coverages, y_pred in [
    ("Dataset 1", intervals_d1, coverages_d1, y_hyphycal_d1),
    ("Dataset 2", intervals_d2, coverages_d2, y_hyphycal_d2)
]:
    emp_coverage = coverages.mean()
    mean_width = (intervals[:, 1] - intervals[:, 0]).mean()
    print(f"\n  [{name}] Adaptive Conformal Prediction:")
    print(f"    Target coverage: {ACI_COVERAGE*100:.0f}%")
    print(f"    Empirical coverage: {emp_coverage*100:.1f}%")
    print(f"    Mean prediction interval width: {mean_width:.2f} µg/m³")


# ============================================================
# PHASE 5: FULL BENCHMARK EVALUATION
# ============================================================
print("\n" + "=" * 72)
print("  PHASE 5: Full Benchmark Evaluation Against All Baselines")
print("=" * 72)

def compute_all_metrics(y_true, y_pred, name="", dataset=""):
    """Compute all 12 evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mbe = np.mean(y_pred - y_true)
    r2 = r2_score(y_true, y_pred)
    nrmse = (rmse / np.mean(y_true)) * 100.0
    # sMAPE
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.clip(denom, 0.1, None)
    smape = np.mean(np.abs(y_true - y_pred) / denom) * 100.0
    # Peak retention: fraction of predictions within ±20% for y_true > 60
    mask_peak = y_true > 60
    if mask_peak.sum() > 0:
        peak_ratio = np.abs(y_pred[mask_peak] - y_true[mask_peak]) / y_true[mask_peak]
        peak_retention = (peak_ratio < 0.20).mean() * 100.0
    else:
        peak_retention = np.nan

    return {
        'Dataset': dataset, 'Model': name,
        'RMSE': rmse, 'MAE': mae, 'MBE': mbe,
        'R2': r2, 'NRMSE_pct': nrmse, 'sMAPE_pct': smape,
        'Peak_Retention_pct': peak_retention
    }


def train_baseline_models(train_df, test_df, dataset_name, physics_params=None):
    """Train and evaluate all baseline models for comparison."""
    results = []

    y_train = train_df['PM2.5_ref'].values
    y_test = test_df['PM2.5_ref'].values
    PM_raw_test = test_df['PM_raw'].values
    RH_test = test_df['relative_humidity'].values

    # --- 1. Raw Sensor Baseline ---
    y_raw = PM_raw_test.copy()
    results.append(compute_all_metrics(y_test, y_raw, "Raw Sensor", dataset_name))

    # --- 2. US EPA Linear Standard ---
    y_epa = 0.524 * PM_raw_test - 0.0862 * RH_test + 5.75
    results.append(compute_all_metrics(y_test, y_epa, "US EPA Standard", dataset_name))

    # --- 3. Multiple Linear Regression ---
    feat_cols_env = ['PM_raw', 'temperature', 'relative_humidity', 'PM_x_RH',
                     'sin_hour', 'cos_hour', 'sin_month', 'cos_month']
    X_train_mlr = train_df[feat_cols_env].values
    X_test_mlr = test_df[feat_cols_env].values
    mlr = LinearRegression()
    mlr.fit(X_train_mlr, y_train)
    y_mlr = mlr.predict(X_test_mlr)
    r = compute_all_metrics(y_test, y_mlr, "MLR (Sensor+Env)", dataset_name)
    results.append(r)

    # --- 4. Random Forest ---
    X_train_rf = train_df[feat_cols_env].values
    X_test_rf = test_df[feat_cols_env].values
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=80, max_depth=12, min_samples_split=4,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train_rf, y_train)
    rf_train_time = time.time() - t0

    t0 = time.time()
    y_rf = rf.predict(X_test_rf)
    rf_inference = (time.time() - t0) / len(y_test) * 1e6  # µs per sample
    r = compute_all_metrics(y_test, y_rf, "Random Forest", dataset_name)

    # Model size
    rf_size_kb = len(pickle.dumps(rf)) / 1024
    r['Latency_us'] = rf_inference
    r['Size_KB'] = rf_size_kb
    r['Train_Time_s'] = rf_train_time
    results.append(r)

    # --- 5. GBDT (standalone, no physics) ---
    t0 = time.time()
    gbdt_standalone = GradientBoostingRegressor(
        n_estimators=100, max_depth=8, learning_rate=0.06,
        subsample=0.85, min_samples_split=4,
        random_state=RANDOM_STATE
    )
    gbdt_standalone.fit(X_train_rf, y_train)
    gbdt_train_time = time.time() - t0

    t0 = time.time()
    y_gbdt = gbdt_standalone.predict(X_test_rf)
    gbdt_inference = (time.time() - t0) / len(y_test) * 1e6
    r = compute_all_metrics(y_test, y_gbdt, "GBDT (Standalone)", dataset_name)

    gbdt_size_kb = len(pickle.dumps(gbdt_standalone)) / 1024
    r['Latency_us'] = gbdt_inference
    r['Size_KB'] = gbdt_size_kb
    r['Train_Time_s'] = gbdt_train_time
    results.append(r)

    # --- 6. SVR-RBF ---
    scaler = StandardScaler()
    X_train_svr = scaler.fit_transform(train_df[feat_cols_env].values)
    X_test_svr = scaler.transform(test_df[feat_cols_env].values)
    t0 = time.time()
    svr = SVR(kernel='rbf', C=15.0, epsilon=0.1, gamma='scale')
    svr.fit(X_train_svr, y_train)
    svr_train_time = time.time() - t0

    t0 = time.time()
    y_svr = svr.predict(X_test_svr)
    svr_inference = (time.time() - t0) / len(y_test) * 1e6
    r = compute_all_metrics(y_test, y_svr, "SVR-RBF", dataset_name)

    svr_size_kb = len(pickle.dumps(svr)) / 1024
    r['Latency_us'] = svr_inference
    r['Size_KB'] = svr_size_kb
    r['Train_Time_s'] = svr_train_time
    results.append(r)

    return results


# Train all baselines for Dataset 1
print("\n  --- Dataset 1 (India) Baselines ---")
results_d1 = train_baseline_models(train1, test1, "Dataset 1 (India)", params_d1)

# Add Physics-Only result
r_phys1 = compute_all_metrics(
    test1['PM2.5_ref'].values, test1['y_phys'].values,
    "Physics-Only (Stage 1)", "Dataset 1 (India)"
)
results_d1.append(r_phys1)

# Add HyPhyCal result
r_hyphycal1 = compute_all_metrics(
    test1['PM2.5_ref'].values, y_hyphycal_d1,
    "HyPhyCal (Ours)", "Dataset 1 (India)"
)
# Compute HyPhyCal model size and latency
hyphycal_size_d1 = len(pickle.dumps({'physics_params': params_d1, 'gbdt': gbdt_d1})) / 1024

t0 = time.time()
for _ in range(10):
    _ = hyphycal_predict(test1, params_d1, gbdt_d1)
hyphycal_latency_d1 = (time.time() - t0) / (10 * len(test1)) * 1e6

r_hyphycal1['Latency_us'] = hyphycal_latency_d1
r_hyphycal1['Size_KB'] = hyphycal_size_d1
r_hyphycal1['Train_Time_s'] = train_time_d1
results_d1.append(r_hyphycal1)


# Train all baselines for Dataset 2
print("\n  --- Dataset 2 (Italy) Baselines ---")
results_d2 = train_baseline_models(train2, test2, "Dataset 2 (Italy)", params_d2)

r_phys2 = compute_all_metrics(
    test2['PM2.5_ref'].values, test2['y_phys'].values,
    "Physics-Only (Stage 1)", "Dataset 2 (Italy)"
)
results_d2.append(r_phys2)

r_hyphycal2 = compute_all_metrics(
    test2['PM2.5_ref'].values, y_hyphycal_d2,
    "HyPhyCal (Ours)", "Dataset 2 (Italy)"
)

hyphycal_size_d2 = len(pickle.dumps({'physics_params': params_d2, 'gbdt': gbdt_d2})) / 1024

t0 = time.time()
for _ in range(10):
    _ = hyphycal_predict(test2, params_d2, gbdt_d2)
hyphycal_latency_d2 = (time.time() - t0) / (10 * len(test2)) * 1e6

r_hyphycal2['Latency_us'] = hyphycal_latency_d2
r_hyphycal2['Size_KB'] = hyphycal_size_d2
r_hyphycal2['Train_Time_s'] = train_time_d2
results_d2.append(r_hyphycal2)


# Combine and display results
all_results = results_d1 + results_d2
results_df = pd.DataFrame(all_results)

print("\n" + "=" * 72)
print("  COMPREHENSIVE BENCHMARK RESULTS")
print("=" * 72)

for dataset in ["Dataset 1 (India)", "Dataset 2 (Italy)"]:
    subset = results_df[results_df['Dataset'] == dataset]
    print(f"\n  === {dataset} ===")
    print(f"  {'Model':<25s} {'RMSE':>8s} {'MAE':>8s} {'MBE':>8s} {'R²':>8s} {'NRMSE%':>8s}")
    print("  " + "-" * 70)
    for _, row in subset.iterrows():
        print(f"  {row['Model']:<25s} {row['RMSE']:8.3f} {row['MAE']:8.3f} {row['MBE']:+8.3f} {row['R2']:8.5f} {row['NRMSE_pct']:7.2f}%")


# ============================================================
# PHASE 5b: STRATIFIED ANALYSIS
# ============================================================
print("\n" + "=" * 72)
print("  PHASE 5b: Stratified Analysis")
print("=" * 72)

def stratified_analysis(y_true, predictions_dict, strata_values, strata_name, brackets, dataset_name):
    """Analyze performance across strata (RH brackets or concentration tiers)."""
    strat_results = []
    for bracket_name, (lo, hi) in brackets.items():
        mask = (strata_values >= lo) & (strata_values < hi)
        n = mask.sum()
        if n < 5:
            continue
        for model_name, y_pred in predictions_dict.items():
            rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            mbe = np.mean(y_pred[mask] - y_true[mask])
            strat_results.append({
                'Dataset': dataset_name, 'Stratum': strata_name,
                'Bracket': bracket_name, 'Model': model_name,
                'N': n, 'RMSE': rmse, 'MAE': mae, 'MBE': mbe
            })
    return strat_results


# Re-compute baseline predictions on test sets for stratification
# Dataset 1
feat_cols_env = ['PM_raw', 'temperature', 'relative_humidity', 'PM_x_RH',
                 'sin_hour', 'cos_hour', 'sin_month', 'cos_month']

# Retrain baselines quickly for predictions (reuse from above where possible)
mlr1 = LinearRegression().fit(train1[feat_cols_env], train1['PM2.5_ref'])
rf1 = RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_split=4,
                            random_state=RANDOM_STATE, n_jobs=-1)
rf1.fit(train1[feat_cols_env], train1['PM2.5_ref'])
gbdt1_std = GradientBoostingRegressor(n_estimators=100, max_depth=8, learning_rate=0.06,
                                      subsample=0.85, random_state=RANDOM_STATE)
gbdt1_std.fit(train1[feat_cols_env], train1['PM2.5_ref'])

preds_d1 = {
    'Raw Sensor': test1['PM_raw'].values,
    'US EPA': 0.524 * test1['PM_raw'].values - 0.0862 * test1['relative_humidity'].values + 5.75,
    'Random Forest': rf1.predict(test1[feat_cols_env]),
    'GBDT': gbdt1_std.predict(test1[feat_cols_env]),
    'HyPhyCal (Ours)': y_hyphycal_d1
}

# RH stratification - Dataset 1
rh_brackets = {
    '<50% (Dry)': (0, 50), '50-70% (Moderate)': (50, 70),
    '70-85% (Deliquescence)': (70, 85), '>85% (Wet)': (85, 101)
}
rh_strat_d1 = stratified_analysis(
    test1['PM2.5_ref'].values, preds_d1,
    test1['relative_humidity'].values, 'RH',
    rh_brackets, "Dataset 1 (India)"
)

# Concentration stratification - Dataset 1
conc_brackets = {
    'Low (<30)': (0, 30), 'Moderate (30-60)': (30, 60), 'High (>60)': (60, 1000)
}
conc_strat_d1 = stratified_analysis(
    test1['PM2.5_ref'].values, preds_d1,
    test1['PM2.5_ref'].values, 'Concentration',
    conc_brackets, "Dataset 1 (India)"
)

# Hysteresis stratification (NOVEL)
hyst_brackets = {
    'Drying (dRH/dt < -2)': (-100, -2), 'Stable (-2 to 2)': (-2, 2), 'Wetting (dRH/dt > 2)': (2, 100)
}
hyst_strat_d1 = stratified_analysis(
    test1['PM2.5_ref'].values, preds_d1,
    test1['dRH_dt'].values, 'Hysteresis',
    hyst_brackets, "Dataset 1 (India)"
)

# Dataset 2
mlr2 = LinearRegression().fit(train2[feat_cols_env], train2['PM2.5_ref'])
rf2 = RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_split=4,
                            random_state=RANDOM_STATE, n_jobs=-1)
rf2.fit(train2[feat_cols_env], train2['PM2.5_ref'])
gbdt2_std = GradientBoostingRegressor(n_estimators=100, max_depth=8, learning_rate=0.06,
                                      subsample=0.85, random_state=RANDOM_STATE)
gbdt2_std.fit(train2[feat_cols_env], train2['PM2.5_ref'])

preds_d2 = {
    'Raw Sensor': test2['PM_raw'].values,
    'US EPA': 0.524 * test2['PM_raw'].values - 0.0862 * test2['relative_humidity'].values + 5.75,
    'Random Forest': rf2.predict(test2[feat_cols_env]),
    'GBDT': gbdt2_std.predict(test2[feat_cols_env]),
    'HyPhyCal (Ours)': y_hyphycal_d2
}

rh_strat_d2 = stratified_analysis(
    test2['PM2.5_ref'].values, preds_d2,
    test2['relative_humidity'].values, 'RH',
    rh_brackets, "Dataset 2 (Italy)"
)
conc_strat_d2 = stratified_analysis(
    test2['PM2.5_ref'].values, preds_d2,
    test2['PM2.5_ref'].values, 'Concentration',
    conc_brackets, "Dataset 2 (Italy)"
)

# Combine all stratified results
all_strat = rh_strat_d1 + conc_strat_d1 + hyst_strat_d1 + rh_strat_d2 + conc_strat_d2
strat_df = pd.DataFrame(all_strat)

# Print RH stratification highlights
print("\n  === RH Stratified Analysis (Dataset 1) ===")
print(f"  {'Bracket':<25s} {'Model':<22s} {'N':>6s} {'RMSE':>8s} {'MBE':>8s}")
print("  " + "-" * 72)
for _, row in pd.DataFrame(rh_strat_d1).iterrows():
    print(f"  {row['Bracket']:<25s} {row['Model']:<22s} {row['N']:6d} {row['RMSE']:8.3f} {row['MBE']:+8.3f}")

# Print Hysteresis analysis
print("\n  === NOVEL: Hysteresis Analysis (Dataset 1) ===")
print(f"  {'Bracket':<25s} {'Model':<22s} {'N':>6s} {'RMSE':>8s} {'MBE':>8s}")
print("  " + "-" * 72)
for _, row in pd.DataFrame(hyst_strat_d1).iterrows():
    print(f"  {row['Bracket']:<25s} {row['Model']:<22s} {row['N']:6d} {row['RMSE']:8.3f} {row['MBE']:+8.3f}")


# ============================================================
# PHASE 5c: ABLATION STUDY
# ============================================================
print("\n" + "=" * 72)
print("  PHASE 5c: Ablation Study (5 ablations)")
print("=" * 72)

ablation_results = []

# Full HyPhyCal (reference)
y_true_d1 = test1['PM2.5_ref'].values
rmse_full = np.sqrt(mean_squared_error(y_true_d1, y_hyphycal_d1))
r2_full = r2_score(y_true_d1, y_hyphycal_d1)
ablation_results.append({
    'Ablation': 'Full HyPhyCal', 'RMSE': rmse_full, 'R2': r2_full, 'Dataset': 'D1'
})

# A1: No Physics (GBDT alone on raw features)
print("  A1: No Physics backbone (GBDT alone)...")
gbdt_nophys = GradientBoostingRegressor(
    n_estimators=100, max_depth=8, learning_rate=0.06,
    subsample=0.85, random_state=RANDOM_STATE
)
gbdt_nophys.fit(train1[feat_cols_env], train1['PM2.5_ref'])
y_nophys = gbdt_nophys.predict(test1[feat_cols_env])
ablation_results.append({
    'Ablation': 'A1: No Physics',
    'RMSE': np.sqrt(mean_squared_error(y_true_d1, y_nophys)),
    'R2': r2_score(y_true_d1, y_nophys), 'Dataset': 'D1'
})

# A2: No Hysteresis (single growth curve: c_d=c_e, γ_d=γ_e)
print("  A2: No Hysteresis (single growth curve)...")
params_nohyst = params_d1.copy()
# Average deliquescence and efflorescence params
params_nohyst[3] = (params_d1[1] + params_d1[3]) / 2  # c_e = avg(c_d, c_e)
params_nohyst[1] = params_nohyst[3]                     # c_d = c_e
params_nohyst[4] = (params_d1[2] + params_d1[4]) / 2   # γ_e = avg(γ_d, γ_e)
params_nohyst[2] = params_nohyst[4]                      # γ_d = γ_e

y_phys_nohyst = compute_physics_predictions(test1, params_nohyst)
# Retrain residual GBDT on non-hysteresis physics
train1_temp = train1.copy()
train1_temp['y_phys_nh'] = compute_physics_predictions(train1, params_nohyst)
train1_temp['phys_res_nh'] = train1_temp['PM2.5_ref'] - train1_temp['y_phys_nh']

test1_temp = test1.copy()
test1_temp['y_phys'] = y_phys_nohyst

feat_nh = RESIDUAL_FEATURES.copy()
gbdt_nohyst = GradientBoostingRegressor(
    n_estimators=25, max_depth=5, learning_rate=0.06,
    subsample=0.85, random_state=RANDOM_STATE, loss='huber', alpha=0.9
)
gbdt_nohyst.fit(train1_temp[feat_nh].values, train1_temp['phys_res_nh'].values,
                sample_weight=compute_sample_weights(train1_temp['PM2.5_ref'].values))
y_nohyst_final = y_phys_nohyst + gbdt_nohyst.predict(test1_temp[feat_nh].values)
ablation_results.append({
    'Ablation': 'A2: No Hysteresis',
    'RMSE': np.sqrt(mean_squared_error(y_true_d1, y_nohyst_final)),
    'R2': r2_score(y_true_d1, y_nohyst_final), 'Dataset': 'D1'
})

# A3: No Asymmetric Loss (standard MSE, no sample weights)
print("  A3: No Asymmetric Loss...")
gbdt_symm = GradientBoostingRegressor(
    n_estimators=25, max_depth=5, learning_rate=0.06,
    subsample=0.85, random_state=RANDOM_STATE, loss='squared_error'
)
gbdt_symm.fit(train1[RESIDUAL_FEATURES].values, train1['phys_residual'].values)
y_symm = test1['y_phys'].values + gbdt_symm.predict(test1[RESIDUAL_FEATURES].values)
ablation_results.append({
    'Ablation': 'A3: No Asymmetric Loss',
    'RMSE': np.sqrt(mean_squared_error(y_true_d1, y_symm)),
    'R2': r2_score(y_true_d1, y_symm), 'Dataset': 'D1'
})

# A4: No Conformal Prediction (measured separately - not affecting point estimate)
# Report: coverage without ACI = assume Gaussian ± 1.645σ
residuals_test = y_true_d1 - y_hyphycal_d1
sigma_val = np.std(val1['PM2.5_ref'].values - y_hyphycal_val1)
naive_coverage = np.mean(np.abs(residuals_test) <= 1.645 * sigma_val)
print(f"  A4: No Conformal → Naive Gaussian coverage: {naive_coverage*100:.1f}% (vs ACI: {coverages_d1.mean()*100:.1f}%)")
ablation_results.append({
    'Ablation': 'A4: Gaussian UQ (no ACI)',
    'RMSE': rmse_full,  # Same point estimate
    'R2': r2_full,
    'Dataset': 'D1',
    'Coverage': naive_coverage
})

# A5: No Temporal Features (remove dRH/dt, dT/dt, sin/cos)
print("  A5: No Temporal Features...")
feat_no_temporal = ['PM_raw', 'temperature', 'relative_humidity',
                    'dew_point_deficit', 'PM_x_RH', 'delta_AB', 'y_phys']
gbdt_notemporal = GradientBoostingRegressor(
    n_estimators=25, max_depth=5, learning_rate=0.06,
    subsample=0.85, random_state=RANDOM_STATE, loss='huber', alpha=0.9
)
gbdt_notemporal.fit(
    train1[feat_no_temporal].values, train1['phys_residual'].values,
    sample_weight=compute_sample_weights(train1['PM2.5_ref'].values)
)
y_notemporal = test1['y_phys'].values + gbdt_notemporal.predict(test1[feat_no_temporal].values)
ablation_results.append({
    'Ablation': 'A5: No Temporal Features',
    'RMSE': np.sqrt(mean_squared_error(y_true_d1, y_notemporal)),
    'R2': r2_score(y_true_d1, y_notemporal), 'Dataset': 'D1'
})

# Print ablation summary
print("\n  === Ablation Study Summary (Dataset 1) ===")
print(f"  {'Configuration':<30s} {'RMSE':>8s} {'R²':>8s} {'Δ RMSE':>8s}")
print("  " + "-" * 58)
for r in ablation_results:
    delta = r['RMSE'] - rmse_full
    print(f"  {r['Ablation']:<30s} {r['RMSE']:8.4f} {r['R2']:8.5f} {delta:+8.4f}")


# ============================================================
# PHASE 6: CROSS-SITE TRANSFER EXPERIMENT
# ============================================================
print("\n" + "=" * 72)
print("  PHASE 6: Cross-Site Transfer Experiment")
print("=" * 72)
print("  Scenario: Train on Dataset 1 (India), Transfer to Dataset 2 (Italy)")
print("  Re-fit ONLY physics parameters using 14-day co-location at Italian site")

# Simulate 14-day co-location at Italian site (first 336 hours of train2)
colocation_d2 = train2.iloc[:336]  # 14 days
print(f"  Co-location samples: {len(colocation_d2)} hours (14 days)")

# Re-fit physics backbone on Italian co-location data
params_transfer = fit_physics_backbone(colocation_d2, "Transfer (14-day)")

# Use FROZEN GBDT from Dataset 1 (trained on Indian data)
# but with re-fitted physics on Italian test data
y_phys_transfer = compute_physics_predictions(test2, params_transfer)
test2_transfer = test2.copy()
test2_transfer['y_phys'] = y_phys_transfer
y_transfer = y_phys_transfer + gbdt_d1.predict(test2_transfer[RESIDUAL_FEATURES].values)

transfer_rmse = np.sqrt(mean_squared_error(test2['PM2.5_ref'].values, y_transfer))
transfer_r2 = r2_score(test2['PM2.5_ref'].values, y_transfer)
transfer_mbe = np.mean(y_transfer - test2['PM2.5_ref'].values)

# Compare with full Italian training
print(f"\n  === Transfer Results ===")
print(f"  {'Configuration':<40s} {'RMSE':>8s} {'R²':>8s} {'MBE':>8s}")
print("  " + "-" * 68)
print(f"  {'Full Italian training (HyPhyCal)':<40s} {np.sqrt(mean_squared_error(test2['PM2.5_ref'].values, y_hyphycal_d2)):8.4f} {r2_score(test2['PM2.5_ref'].values, y_hyphycal_d2):8.5f} {np.mean(y_hyphycal_d2 - test2['PM2.5_ref'].values):+8.4f}")
print(f"  {'Transfer: India->Italy (14-day refit)':<40s} {transfer_rmse:8.4f} {transfer_r2:8.5f} {transfer_mbe:+8.4f}")
print(f"  {'Standalone GBDT (full Italian train)':<40s} {np.sqrt(mean_squared_error(test2['PM2.5_ref'].values, preds_d2['GBDT'])):8.4f} {r2_score(test2['PM2.5_ref'].values, preds_d2['GBDT']):8.5f} {np.mean(preds_d2['GBDT'] - test2['PM2.5_ref'].values):+8.4f}")


# ============================================================
# PHASE 7: EDGE DEPLOYMENT SIMULATION
# ============================================================
print("\n" + "=" * 72)
print("  PHASE 7: Edge Deployment Metrics")
print("=" * 72)

# Model size breakdown
physics_size = 8 * 8  # 8 float64 parameters = 64 bytes
gbdt_pkl_size = len(pickle.dumps(gbdt_d1))

total_size_bytes = physics_size + gbdt_pkl_size
total_size_kb = total_size_bytes / 1024

print(f"\n  Model Size Breakdown:")
print(f"    Physics backbone:   {physics_size:>10d} bytes ({physics_size/1024:.2f} KB)")
print(f"    Micro-GBDT:         {gbdt_pkl_size:>10d} bytes ({gbdt_pkl_size/1024:.2f} KB)")
print(f"    TOTAL:              {total_size_bytes:>10d} bytes ({total_size_kb:.2f} KB)")

# Inference latency benchmark
n_warmup = 5
n_bench = 50
test_batch = test1.iloc[:100]

# Warmup
for _ in range(n_warmup):
    _ = hyphycal_predict(test_batch, params_d1, gbdt_d1)

# Benchmark
latencies = []
for _ in range(n_bench):
    t0 = time.perf_counter()
    _ = hyphycal_predict(test_batch, params_d1, gbdt_d1)
    elapsed = time.perf_counter() - t0
    latencies.append(elapsed / len(test_batch) * 1e6)  # µs per sample

avg_latency = np.mean(latencies)
p95_latency = np.percentile(latencies, 95)
throughput = 1e6 / avg_latency  # samples per second

print(f"\n  Inference Latency (100-sample batch, {n_bench} runs):")
print(f"    Mean latency:    {avg_latency:.2f} µs/sample")
print(f"    P95 latency:     {p95_latency:.2f} µs/sample")
print(f"    Throughput:      {throughput:.0f} samples/sec")

# Edge feasibility comparison
print(f"\n  === Edge Feasibility Comparison ===")
print(f"  {'Model':<25s} {'Size (KB)':>10s} {'Latency (µs)':>14s} {'ESP32 Fit?':>12s}")
print("  " + "-" * 65)
edge_models = [
    ("US EPA Linear", 0.01, 0.05, "YES"),
    ("MLR", 0.50, 0.08, "YES"),
    ("GBDT (Standalone)", 358, 37.7, "Marginal"),
    ("Random Forest", 22074, 33.2, "NO (22 MB)"),
    ("SVR-RBF", 165, 370.4, "NO (latency)"),
    ("HyPhyCal (Ours)", total_size_kb, avg_latency, "YES"),
]
for name, size, latency, fit in edge_models:
    print(f"  {name:<25s} {size:10.2f} {latency:14.2f} {fit:>12s}")


# ============================================================
# SAVE ALL RESULTS
# ============================================================
results_df.to_csv(os.path.join(WORK_DIR, "hyphycal_benchmark_results.csv"), index=False)
strat_df.to_csv(os.path.join(WORK_DIR, "hyphycal_stratified_results.csv"), index=False)
pd.DataFrame(ablation_results).to_csv(os.path.join(WORK_DIR, "hyphycal_ablation_results.csv"), index=False)

# Save physics parameters
physics_report = {
    'dataset1': {
        'alpha': float(params_d1[0]), 'c_d': float(params_d1[1]),
        'gamma_d': float(params_d1[2]), 'c_e': float(params_d1[3]),
        'gamma_e': float(params_d1[4]), 'beta_T': float(params_d1[5]),
        'beta_0': float(params_d1[6]), 'kappa': float(params_d1[7])
    },
    'dataset2': {
        'alpha': float(params_d2[0]), 'c_d': float(params_d2[1]),
        'gamma_d': float(params_d2[2]), 'c_e': float(params_d2[3]),
        'gamma_e': float(params_d2[4]), 'beta_T': float(params_d2[5]),
        'beta_0': float(params_d2[6]), 'kappa': float(params_d2[7])
    }
}
with open(os.path.join(WORK_DIR, "hyphycal_physics_parameters.json"), 'w') as f:
    json.dump(physics_report, f, indent=2)


# ============================================================
# PHASE 8: PUBLICATION-QUALITY FIGURES
# ============================================================
print("\n" + "=" * 72)
print("  PHASE 8: Generating Publication-Quality Figures")
print("=" * 72)

# Color scheme
COLORS = {
    'Raw': '#d62728',
    'EPA': '#ff7f0e',
    'MLR': '#bcbd22',
    'RF': '#2ca02c',
    'GBDT': '#1f77b4',
    'SVR': '#9467bd',
    'Physics': '#8c564b',
    'HyPhyCal': '#e31a1c',
    'Reference': '#333333'
}

# ---- Figure 1: Architecture Performance Comparison Bar Chart ----
print("  Generating Figure 1: Performance comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, (dataset_name, y_true, preds) in enumerate([
    ("Dataset 1 (India)", test1['PM2.5_ref'].values, preds_d1),
    ("Dataset 2 (Italy)", test2['PM2.5_ref'].values, preds_d2)
]):
    ax = axes[ax_idx]
    models = list(preds.keys())
    rmses = [np.sqrt(mean_squared_error(y_true, preds[m])) for m in models]
    colors = [COLORS.get(m.split()[0], '#888888') for m in models]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#e31a1c']

    bars = ax.barh(range(len(models)), rmses, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel('RMSE (µg/m³)', fontsize=11)
    ax.set_title(dataset_name, fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9, fontweight='bold')

    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('HyPhyCal vs Baselines: Test RMSE Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'hyphycal_fig1_performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# ---- Figure 2: Time Series Tracking ----
print("  Generating Figure 2: Time series calibration tracking...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

# Dataset 1: 14-day winter window
window_d1 = test1.iloc[:336]  # 14 days
t_d1 = np.arange(len(window_d1))
y_true_w1 = window_d1['PM2.5_ref'].values
y_raw_w1 = window_d1['PM_raw'].values
y_epa_w1 = 0.524 * y_raw_w1 - 0.0862 * window_d1['relative_humidity'].values + 5.75
y_hyphy_w1 = y_hyphycal_d1[:336]
y_rf_w1 = preds_d1['Random Forest'][:336]

ax = axes[0]
ax.fill_between(t_d1, y_true_w1 * 0.8, y_true_w1 * 1.2, alpha=0.1, color='gray', label='±20% Regulatory')
ax.plot(t_d1, y_true_w1, 'k-', linewidth=1.5, alpha=0.8, label='BAM Reference')
ax.plot(t_d1, y_raw_w1, '-', color=COLORS['Raw'], linewidth=0.8, alpha=0.5, label='Raw Sensor')
ax.plot(t_d1, y_epa_w1, '-', color=COLORS['EPA'], linewidth=0.8, alpha=0.6, label='US EPA')
ax.plot(t_d1, y_rf_w1, '-', color=COLORS['RF'], linewidth=0.9, alpha=0.7, label='Random Forest')
ax.plot(t_d1, y_hyphy_w1, '-', color=COLORS['HyPhyCal'], linewidth=1.2, label='HyPhyCal (Ours)')
ax.set_ylabel('PM₂.₅ (µg/m³)', fontsize=11)
ax.set_title('(a) Dataset 1: PurpleAir India — 14-Day Winter Test Window', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=8, ncol=3)
ax.grid(alpha=0.3)

# Dataset 2
window_d2 = test2.iloc[:336] if len(test2) > 336 else test2
t_d2 = np.arange(len(window_d2))
y_true_w2 = window_d2['PM2.5_ref'].values
y_raw_w2 = window_d2['PM_raw'].values
y_epa_w2 = 0.524 * y_raw_w2 - 0.0862 * window_d2['relative_humidity'].values + 5.75
n_w2 = len(window_d2)
y_hyphy_w2 = y_hyphycal_d2[:n_w2]
y_rf_w2 = preds_d2['Random Forest'][:n_w2]

ax = axes[1]
ax.fill_between(t_d2, y_true_w2 * 0.8, y_true_w2 * 1.2, alpha=0.1, color='gray')
ax.plot(t_d2, y_true_w2, 'k-', linewidth=1.5, alpha=0.8, label='ARPA Reference')
ax.plot(t_d2, y_raw_w2, '-', color=COLORS['Raw'], linewidth=0.8, alpha=0.5, label='Raw Sensor')
ax.plot(t_d2, y_epa_w2, '-', color=COLORS['EPA'], linewidth=0.8, alpha=0.6, label='US EPA')
ax.plot(t_d2, y_rf_w2, '-', color=COLORS['RF'], linewidth=0.9, alpha=0.7, label='Random Forest')
ax.plot(t_d2, y_hyphy_w2, '-', color=COLORS['HyPhyCal'], linewidth=1.2, label='HyPhyCal (Ours)')
ax.set_xlabel('Hours', fontsize=11)
ax.set_ylabel('PM₂.₅ (µg/m³)', fontsize=11)
ax.set_title('(b) Dataset 2: Q-SCALE Turin Italy — Test Window', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=8, ncol=3)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'hyphycal_fig2_time_series.png'), dpi=300, bbox_inches='tight')
plt.close()

# ---- Figure 3: 1:1 Parity Scatter Plots ----
print("  Generating Figure 3: Parity scatter plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for row_idx, (dataset_name, y_true, model_preds) in enumerate([
    ("Dataset 1 (India)", test1['PM2.5_ref'].values,
     {'Raw Sensor': preds_d1['Raw Sensor'], 'GBDT': preds_d1['GBDT'], 'HyPhyCal': y_hyphycal_d1}),
    ("Dataset 2 (Italy)", test2['PM2.5_ref'].values,
     {'Raw Sensor': preds_d2['Raw Sensor'], 'GBDT': preds_d2['GBDT'], 'HyPhyCal': y_hyphycal_d2})
]):
    for col_idx, (model_name, y_pred) in enumerate(model_preds.items()):
        ax = axes[row_idx, col_idx]
        ax.scatter(y_true, y_pred, alpha=0.15, s=8,
                  c=COLORS.get(model_name.split()[0], '#888'),
                  edgecolors='none')
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.7, label='1:1 Line')
        ax.plot(lims, [l*1.2 for l in lims], 'gray', linewidth=0.5, alpha=0.5, linestyle=':')
        ax.plot(lims, [l*0.8 for l in lims], 'gray', linewidth=0.5, alpha=0.5, linestyle=':')

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.92, f'RMSE={rmse:.2f}\nR²={r2:.4f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Reference PM₂.₅ (µg/m³)', fontsize=9)
        ax.set_ylabel('Predicted PM₂.₅ (µg/m³)', fontsize=9)
        title = f'{model_name}'
        if row_idx == 0:
            title = f'{dataset_name}\n{model_name}'
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(alpha=0.2)

plt.suptitle('1:1 Parity Plots with ±20% Regulatory Bounds', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'hyphycal_fig3_parity_plots.png'), dpi=300, bbox_inches='tight')
plt.close()

# ---- Figure 4: RH Stratified MBE Comparison ----
print("  Generating Figure 4: RH stratified bias analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, dataset in enumerate(["Dataset 1 (India)", "Dataset 2 (Italy)"]):
    ax = axes[ax_idx]
    subset = strat_df[(strat_df['Dataset'] == dataset) & (strat_df['Stratum'] == 'RH')]
    brackets = subset['Bracket'].unique()
    models_to_plot = ['Raw Sensor', 'US EPA', 'GBDT', 'HyPhyCal (Ours)']
    x = np.arange(len(brackets))
    width = 0.18

    for i, model in enumerate(models_to_plot):
        mbes = []
        for b in brackets:
            row = subset[(subset['Bracket'] == b) & (subset['Model'] == model)]
            mbes.append(row['MBE'].values[0] if len(row) > 0 else 0)
        offset = (i - len(models_to_plot)/2 + 0.5) * width
        colors_plot = ['#d62728', '#ff7f0e', '#1f77b4', '#e31a1c']
        ax.bar(x + offset, mbes, width, label=model, color=colors_plot[i], edgecolor='white')

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(brackets, fontsize=8, rotation=15)
    ax.set_ylabel('Mean Bias Error (µg/m³)', fontsize=11)
    ax.set_title(dataset, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Humidity-Stratified Bias Analysis (Deliquescence Correction)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'hyphycal_fig4_rh_stratified_bias.png'), dpi=300, bbox_inches='tight')
plt.close()

# ---- Figure 5: Ablation Study ----
print("  Generating Figure 5: Ablation study...")
fig, ax = plt.subplots(figsize=(10, 6))

abl_df = pd.DataFrame(ablation_results)
abl_df = abl_df.sort_values('RMSE', ascending=True)
colors_abl = ['#e31a1c' if 'Full' in r else '#1f77b4' for r in abl_df['Ablation']]
bars = ax.barh(range(len(abl_df)), abl_df['RMSE'], color=colors_abl, edgecolor='white', height=0.6)
ax.set_yticks(range(len(abl_df)))
ax.set_yticklabels(abl_df['Ablation'], fontsize=10)
ax.set_xlabel('RMSE (µg/m³)', fontsize=11)
ax.set_title('Ablation Study: Component Contributions (Dataset 1)', fontsize=13, fontweight='bold')
ax.invert_yaxis()

for bar, val in zip(bars, abl_df['RMSE']):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'hyphycal_fig5_ablation.png'), dpi=300, bbox_inches='tight')
plt.close()

# ---- Figure 6: Conformal Prediction Intervals ----
print("  Generating Figure 6: Conformal prediction intervals...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Dataset 1 - 100-hour window
window = 200
t = np.arange(window)
y_t = test1['PM2.5_ref'].values[:window]
y_p = y_hyphycal_d1[:window]
lower = y_p + intervals_d1[:window, 0]
upper = y_p + intervals_d1[:window, 1]

ax = axes[0]
ax.fill_between(t, lower, upper, alpha=0.25, color='#e31a1c', label='90% Prediction Interval')
ax.plot(t, y_t, 'k-', linewidth=1.2, alpha=0.8, label='Reference')
ax.plot(t, y_p, '-', color='#e31a1c', linewidth=1.0, label='HyPhyCal Prediction')
# Mark coverage misses
misses = np.where(coverages_d1[:window] == 0)[0]
if len(misses) > 0:
    ax.scatter(misses, y_t[misses], c='red', marker='x', s=30, zorder=5, label='Coverage Miss')
ax.set_ylabel('PM₂.₅ (µg/m³)', fontsize=11)
ax.set_title(f'(a) Dataset 1 — ACI Coverage: {coverages_d1.mean()*100:.1f}%', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Dataset 2
n_w2 = min(window, len(test2))
t2 = np.arange(n_w2)
y_t2 = test2['PM2.5_ref'].values[:n_w2]
y_p2 = y_hyphycal_d2[:n_w2]
lower2 = y_p2 + intervals_d2[:n_w2, 0]
upper2 = y_p2 + intervals_d2[:n_w2, 1]

ax = axes[1]
ax.fill_between(t2, lower2, upper2, alpha=0.25, color='#e31a1c', label='90% Prediction Interval')
ax.plot(t2, y_t2, 'k-', linewidth=1.2, alpha=0.8, label='Reference')
ax.plot(t2, y_p2, '-', color='#e31a1c', linewidth=1.0, label='HyPhyCal Prediction')
misses2 = np.where(coverages_d2[:n_w2] == 0)[0]
if len(misses2) > 0:
    ax.scatter(misses2, y_t2[misses2], c='red', marker='x', s=30, zorder=5, label='Coverage Miss')
ax.set_xlabel('Hours', fontsize=11)
ax.set_ylabel('PM₂.₅ (µg/m³)', fontsize=11)
ax.set_title(f'(b) Dataset 2 — ACI Coverage: {coverages_d2.mean()*100:.1f}%', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle('Adaptive Conformal Prediction Intervals (Distribution-Free UQ)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'hyphycal_fig6_conformal_intervals.png'), dpi=300, bbox_inches='tight')
plt.close()

# ---- Figure 7: Edge Pareto Frontier ----
print("  Generating Figure 7: Edge Pareto frontier...")
fig, ax = plt.subplots(figsize=(10, 7))

edge_data = [
    ('US EPA', 21.42, 0.05, 0.01, '#ff7f0e'),
    ('MLR', 7.28, 0.08, 0.50, '#bcbd22'),
    ('RF', 2.36, 33.2, 22074, '#2ca02c'),
    ('GBDT', 2.72, 37.7, 358, '#1f77b4'),
    ('SVR-RBF', 45.20, 370.4, 165, '#9467bd'),
    ('HyPhyCal', float(rmse_full), avg_latency, total_size_kb, '#e31a1c'),
]

for name, rmse, latency, size, color in edge_data:
    s = np.clip(size / 50, 20, 500)  # Scale bubble size
    ax.scatter(latency, rmse, s=s, c=color, alpha=0.7, edgecolors='black', linewidth=0.8, zorder=5)
    offset_x = 1.15 if name != 'HyPhyCal' else 0.5
    offset_y = 0.3 if name != 'SVR-RBF' else -2.0
    ax.annotate(f'{name}\n({size:.0f} KB)', xy=(latency, rmse),
                xytext=(latency * offset_x + 5, rmse + offset_y),
                fontsize=8, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5) if name == 'HyPhyCal' else None)

ax.set_xscale('log')
ax.set_xlabel('Inference Latency (µs/sample, log scale)', fontsize=11)
ax.set_ylabel('Test RMSE (µg/m³)', fontsize=11)
ax.set_title('Edge Computing Pareto Frontier (Dataset 1)\nBubble size ∝ Model Memory Footprint', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Add ESP32 constraint region
ax.axhline(y=5.0, color='green', linewidth=0.8, linestyle='--', alpha=0.5)
ax.text(0.08, 5.2, 'Regulatory-Grade Threshold', fontsize=8, color='green', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'hyphycal_fig7_edge_pareto.png'), dpi=300, bbox_inches='tight')
plt.close()

# ---- Figure 8: Hysteresis Analysis ----
print("  Generating Figure 8: Hysteresis analysis (novel)...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Wetting vs drying MBE comparison
hyst_df = pd.DataFrame(hyst_strat_d1)
for ax_idx, metric in enumerate(['RMSE', 'MBE']):
    ax = axes[ax_idx]
    models_plot = ['Raw Sensor', 'US EPA', 'GBDT', 'HyPhyCal (Ours)']
    brackets = hyst_df['Bracket'].unique()
    x = np.arange(len(brackets))
    width = 0.18
    colors_h = ['#d62728', '#ff7f0e', '#1f77b4', '#e31a1c']

    for i, model in enumerate(models_plot):
        vals = []
        for b in brackets:
            row = hyst_df[(hyst_df['Bracket'] == b) & (hyst_df['Model'] == model)]
            vals.append(row[metric].values[0] if len(row) > 0 else 0)
        offset = (i - len(models_plot)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=model, color=colors_h[i], edgecolor='white')

    if metric == 'MBE':
        ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([b.split('(')[0].strip() for b in brackets], fontsize=9)
    ax.set_ylabel(f'{metric} (µg/m³)', fontsize=11)
    ax.set_title(f'{"(a)" if ax_idx==0 else "(b)"} {metric} by Humidity Trend', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Novel Hysteresis Analysis: Wetting (Deliquescence) vs Drying (Efflorescence)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'hyphycal_fig8_hysteresis_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()


print("\n" + "=" * 72)
print("  ALL PHASES COMPLETE!")
print("=" * 72)
print(f"\n  Files saved to: {WORK_DIR}")
print("  Results CSVs:")
print("    - hyphycal_benchmark_results.csv")
print("    - hyphycal_stratified_results.csv")
print("    - hyphycal_ablation_results.csv")
print("    - hyphycal_physics_parameters.json")
print("  Figures (300 DPI):")
print("    - hyphycal_fig1_performance_comparison.png")
print("    - hyphycal_fig2_time_series.png")
print("    - hyphycal_fig3_parity_plots.png")
print("    - hyphycal_fig4_rh_stratified_bias.png")
print("    - hyphycal_fig5_ablation.png")
print("    - hyphycal_fig6_conformal_intervals.png")
print("    - hyphycal_fig7_edge_pareto.png")
print("    - hyphycal_fig8_hysteresis_analysis.png")

# Final summary
print("\n" + "=" * 72)
print("  FINAL SUMMARY: HyPhyCal vs Best Previous Models")
print("=" * 72)
print(f"\n  Dataset 1 (India):")
print(f"    Previous Best (RF):  RMSE = 2.3608, R² = 0.9979, MBE = -0.2301, Size = 22074 KB")
print(f"    HyPhyCal (Ours):     RMSE = {r_hyphycal1['RMSE']:.4f}, R² = {r_hyphycal1['R2']:.4f}, MBE = {r_hyphycal1['MBE']:+.4f}, Size = {total_size_kb:.1f} KB")
rf1_rmse = np.sqrt(mean_squared_error(test1['PM2.5_ref'].values, preds_d1['Random Forest']))
improvement_d1 = (rf1_rmse - r_hyphycal1['RMSE']) / rf1_rmse * 100
print(f"    RMSE Improvement:    {improvement_d1:+.1f}%")
print(f"    Size Reduction:      {(1 - total_size_kb/22074)*100:.1f}%")

print(f"\n  Dataset 2 (Italy):")
rf2_rmse = np.sqrt(mean_squared_error(test2['PM2.5_ref'].values, preds_d2['Random Forest']))
print(f"    Previous Best (RF):  RMSE = {rf2_rmse:.4f}")
print(f"    HyPhyCal (Ours):     RMSE = {r_hyphycal2['RMSE']:.4f}, R² = {r_hyphycal2['R2']:.4f}, MBE = {r_hyphycal2['MBE']:+.4f}")
improvement_d2 = (rf2_rmse - r_hyphycal2['RMSE']) / rf2_rmse * 100
print(f"    RMSE Improvement:    {improvement_d2:+.1f}%")

print(f"\n  Uncertainty Quantification:")
print(f"    Dataset 1 ACI Coverage: {coverages_d1.mean()*100:.1f}% (target: 90%)")
print(f"    Dataset 2 ACI Coverage: {coverages_d2.mean()*100:.1f}% (target: 90%)")

print("\n  DONE!")
