"""
Comprehensive Machine Learning Calibration Benchmark for Low-Cost Air Quality Sensors
Implements:
1. Raw Sensor Baseline
2. Multiple Linear Regression (MLR / EPA Benchmark)
3. Random Forest Regression (Methodology 1)
4. Gradient Boosted Decision Trees / XGBoost equivalent (Methodology 2)
5. Support Vector Regression with RBF Kernel (Methodology 3)

Features:
- Strict Chronological Splitting (60% Train, 20% Val, 20% Test)
- Systematic Feature Ablation (Sensor-Only vs Sensor+Environmental)
- Problem-Specific Air Quality Metrics (RMSE, MAE, MBE, R2, NRMSE, sMAPE)
- Stratified Error Decomposition (RH brackets & Concentration brackets)
- Edge Computing Feasibility Profiling (Inference Latency, Memory Footprint)
- 5 Publication-Grade Visualization Figures
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

# Prevent Windows loky subprocess hanging on missing wmic
import joblib.externals.loky.backend.context as loky_ctx
loky_ctx._count_physical_cores = lambda: 1
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['MPLCONFIGDIR'] = os.path.abspath('.mplconfig')
os.makedirs('.mplconfig', exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style for publication quality plots
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300
})
sns.set_theme(style="whitegrid")

# Ensure reproducibility
np.random.seed(42)

def calculate_metrics(y_true, y_pred):
    """Calculates problem-specific air quality evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mbe = np.mean(y_pred - y_true)  # Mean Bias Error
    r2 = r2_score(y_true, y_pred)
    mean_true = np.mean(y_true)
    nrmse = (rmse / mean_true * 100.0) if mean_true != 0 else np.nan
    
    # Symmetric Mean Absolute Percentage Error (sMAPE)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    valid_idx = denominator > 1e-3
    smape = np.mean(np.abs(y_pred[valid_idx] - y_true[valid_idx]) / denominator[valid_idx]) * 100.0
    
    # Physical plausibility metrics
    pct_negative = np.mean(y_pred < 0) * 100.0
    peak_retention = (np.max(y_pred) / np.max(y_true)) * 100.0 if np.max(y_true) > 0 else np.nan
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MBE': mbe,
        'R2': r2,
        'NRMSE_%': nrmse,
        'sMAPE_%': smape,
        'Pct_Negative_%': pct_negative,
        'Peak_Retention_%': peak_retention
    }

def prepare_dataset1():
    """Loads and preprocesses Dataset 1: PurpleAir Sensor Network in India."""
    df = pd.read_csv('dataset1_purpleair_india.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', inplace=True)
    
    # QA/QC Channel Divergence Check (Campmier et al., 2023)
    # Flag clogs where absolute diff > 5 ug/m3 and relative diff > 20%
    abs_diff = np.abs(df['PM2.5_CF1_A'] - df['PM2.5_CF1_B'])
    mean_val = (df['PM2.5_CF1_A'] + df['PM2.5_CF1_B']) / 2.0
    rel_diff = abs_diff / (mean_val + 1e-4)
    qc_pass = ~((abs_diff > 5.0) & (rel_diff > 0.20))
    
    dropped_count = len(df) - qc_pass.sum()
    print(f"Dataset 1 QA/QC: Dropped {dropped_count} suspect records ({dropped_count/len(df)*100:.2f}%) due to channel divergence.")
    df_clean = df[qc_pass].copy()
    
    # Compute low-cost sensor mean feature
    df_clean['PM2.5_raw'] = (df_clean['PM2.5_CF1_A'] + df_clean['PM2.5_CF1_B']) / 2.0
    
    # Feature engineering: hour, month, PM * RH interaction
    df_clean['hour'] = df_clean['timestamp'].dt.hour
    df_clean['month'] = df_clean['timestamp'].dt.month
    df_clean['PM_x_RH'] = df_clean['PM2.5_raw'] * (df_clean['relative_humidity'] / 100.0)
    
    return df_clean

def prepare_dataset2():
    """Loads and preprocesses Dataset 2: Q-SCALE Turin, Italy."""
    df = pd.read_csv('dataset2_qscale_italy.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', inplace=True)
    
    df['PM2.5_raw'] = df['PM2.5_sensor_median']
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['PM_x_RH'] = df['PM2.5_raw'] * (df['relative_humidity'] / 100.0)
    
    return df

def run_experiment_on_dataset(df, dataset_name, raw_col='PM2.5_raw', target_col='PM2.5_ref'):
    """
    Executes the complete chronological modeling, ablation, stratification,
    and computational profiling on a given dataset.
    """
    print(f"\n=======================================================")
    print(f"Running Calibration Benchmark on {dataset_name}")
    print(f"Total Observations: {len(df)}")
    print(f"=======================================================")
    
    n = len(df)
    n_train = int(0.60 * n)
    n_val = int(0.20 * n)
    n_test = n - n_train - n_val
    
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    
    print(f"Chronological Splits -> Train: {len(train_df)} ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"                     -> Val:   {len(val_df)} ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
    print(f"                     -> Test:  {len(test_df)} ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values
    
    # Feature configurations for ablation study
    features_sensor_only = [raw_col]
    features_env = [raw_col, 'temperature', 'relative_humidity', 'PM_x_RH', 'hour', 'month']
    
    results_list = []
    test_predictions = {'y_true': y_test, 'timestamp': test_df['timestamp'].values, 'RH': test_df['relative_humidity'].values}
    models_dict = {}
    
    # 1. Raw Sensor Baseline (Identity mapping)
    y_pred_raw = test_df[raw_col].values
    m_raw = calculate_metrics(y_test, y_pred_raw)
    m_raw.update({
        'Dataset': dataset_name,
        'Model': 'Raw Sensor Baseline',
        'Feature_Set': 'Sensor-Only',
        'Train_Time_s': 0.0,
        'Inference_Latency_us': 0.0,
        'Model_Size_KB': 0.0
    })
    results_list.append(m_raw)
    test_predictions['Raw Sensor Baseline'] = y_pred_raw
    
    # 2. US EPA Operational Baseline Formula (Barkjohn et al., 2021)
    # y = 0.524 * raw - 0.0862 * RH + 5.75
    y_pred_epa = np.clip(0.524 * test_df[raw_col].values - 0.0862 * test_df['relative_humidity'].values + 5.75, 0.0, None)
    m_epa = calculate_metrics(y_test, y_pred_epa)
    m_epa.update({
        'Dataset': dataset_name,
        'Model': 'US EPA Linear Standard',
        'Feature_Set': 'Sensor+Env (Fixed EPA)',
        'Train_Time_s': 0.0,
        'Inference_Latency_us': 0.05,
        'Model_Size_KB': 0.01
    })
    results_list.append(m_epa)
    test_predictions['US EPA Linear Standard'] = y_pred_epa
    
    # Candidate ML Models to benchmark under both Feature Spaces
    model_constructors = {
        'Multiple Linear Regression (MLR)': lambda: LinearRegression(),
        'Random Forest Regressor': lambda: RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_split=4, n_jobs=1, random_state=42),
        'Gradient Boosted Decision Trees (GBDT/XGB)': lambda: HistGradientBoostingRegressor(learning_rate=0.06, max_iter=100, max_depth=8, l2_regularization=0.1, random_state=42),
        'Support Vector Regression (SVR - RBF)': lambda: SVR(kernel='rbf', C=15.0, epsilon=0.1, gamma='scale')
    }
    
    for f_name, f_cols in [('Sensor-Only', features_sensor_only), ('Sensor+Environmental', features_env)]:
        X_train = train_df[f_cols].values
        X_val = val_df[f_cols].values
        X_test = test_df[f_cols].values
        
        # Scaling is mandatory for SVR
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        for m_name, constructor in model_constructors.items():
            model = constructor()
            
            # Select scaled data if SVR, else raw feature values
            X_tr = X_train_scaled if 'SVR' in m_name else X_train
            X_te = X_test_scaled if 'SVR' in m_name else X_test
            
            # Subsample for SVR if training set is large to avoid O(N^3) quadratic freeze
            if 'SVR' in m_name and len(X_tr) > 3000:
                sub_idx = np.linspace(0, len(X_tr) - 1, 3000, dtype=int)
                X_tr_fit = X_tr[sub_idx]
                y_tr_fit = y_train[sub_idx]
            else:
                X_tr_fit = X_tr
                y_tr_fit = y_train
            
            # Measure Training Wall-Clock Time
            t0 = time.perf_counter()
            model.fit(X_tr_fit, y_tr_fit)
            t_train = time.perf_counter() - t0
            
            # Measure Inference Latency per sample
            t0 = time.perf_counter()
            y_pred = model.predict(X_te)
            t_inf = (time.perf_counter() - t0) / len(X_te) * 1e6  # in microseconds (us)
            
            # Ensure physical non-negativity constraint for final calibrated PM
            # (zero-clipping is standard operational practice in environmental monitoring)
            y_pred_clipped = np.clip(y_pred, 0.0, None)
            
            # Estimate Model Footprint on Disk
            model_bytes = len(pickle.dumps(model))
            model_kb = model_bytes / 1024.0
            
            m_metrics = calculate_metrics(y_test, y_pred_clipped)
            m_metrics.update({
                'Dataset': dataset_name,
                'Model': m_name,
                'Feature_Set': f_name,
                'Train_Time_s': t_train,
                'Inference_Latency_us': t_inf,
                'Model_Size_KB': model_kb
            })
            results_list.append(m_metrics)
            
            if f_name == 'Sensor+Environmental':
                test_predictions[m_name] = y_pred_clipped
                models_dict[m_name] = model
                
            print(f"[{f_name}] {m_name:<42} | RMSE: {m_metrics['RMSE']:>6.2f} | MAE: {m_metrics['MAE']:>6.2f} | R2: {m_metrics['R2']:>5.3f} | MBE: {m_metrics['MBE']:>6.2f} | Latency: {t_inf:>6.1f} us")
            
    df_results = pd.DataFrame(results_list)
    df_preds = pd.DataFrame(test_predictions)
    
    return df_results, df_preds, test_df

def compute_stratified_metrics(df_preds, dataset_name):
    """
    Computes residual statistics across Relative Humidity and Concentration strata.
    """
    # 1. RH Bins: <50%, 50-70%, 70-85%, >85%
    rh_bins = [-np.inf, 50.0, 70.0, 85.0, np.inf]
    rh_labels = ['<50% (Dry)', '50-70% (Moderate)', '70-85% (Deliquescence)', '>85% (Severe Wet)']
    df_preds['RH_Bracket'] = pd.cut(df_preds['RH'], bins=rh_bins, labels=rh_labels)
    
    models_to_check = ['Raw Sensor Baseline', 'US EPA Linear Standard', 'Multiple Linear Regression (MLR)', 
                       'Random Forest Regressor', 'Gradient Boosted Decision Trees (GBDT/XGB)', 'Support Vector Regression (SVR - RBF)']
    
    rh_results = []
    for bracket in rh_labels:
        sub = df_preds[df_preds['RH_Bracket'] == bracket]
        if len(sub) == 0:
            continue
        for m in models_to_check:
            if m in sub.columns:
                err = sub[m] - sub['y_true']
                rmse = np.sqrt(mean_squared_error(sub['y_true'], sub[m]))
                mae = mean_absolute_error(sub['y_true'], sub[m])
                mbe = np.mean(err)
                rh_results.append({
                    'Dataset': dataset_name,
                    'Stratum_Type': 'Relative_Humidity',
                    'Bracket': bracket,
                    'Model': m,
                    'Sample_Count': len(sub),
                    'RMSE': rmse,
                    'MAE': mae,
                    'MBE': mbe
                })
                
    # 2. Concentration Bins: Low (<30), Moderate (30-60), High (>60)
    conc_bins = [-np.inf, 30.0, 60.0, np.inf]
    conc_labels = ['Low (<30 ug/m3)', 'Moderate (30-60 ug/m3)', 'High (>60 ug/m3)']
    df_preds['Conc_Bracket'] = pd.cut(df_preds['y_true'], bins=conc_bins, labels=conc_labels)
    
    conc_results = []
    for bracket in conc_labels:
        sub = df_preds[df_preds['Conc_Bracket'] == bracket]
        if len(sub) == 0:
            continue
        for m in models_to_check:
            if m in sub.columns:
                err = sub[m] - sub['y_true']
                rmse = np.sqrt(mean_squared_error(sub['y_true'], sub[m]))
                mae = mean_absolute_error(sub['y_true'], sub[m])
                mbe = np.mean(err)
                conc_results.append({
                    'Dataset': dataset_name,
                    'Stratum_Type': 'PM2.5_Concentration',
                    'Bracket': bracket,
                    'Model': m,
                    'Sample_Count': len(sub),
                    'RMSE': rmse,
                    'MAE': mae,
                    'MBE': mbe
                })
                
    return pd.DataFrame(rh_results), pd.DataFrame(conc_results)

def plot_publication_figures(df_preds1, df_preds2, df_results, df_rh_all):
    """
    Generates 5 publication-ready, IEEE-standard visualization figures.
    """
    print("\nGenerating Publication-Quality Visualizations...")
    
    # -------------------------------------------------------------
    # Figure 1: Time Series Calibration Tracking (Temporal Generalization)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    
    # Dataset 1 India snippet (14 days / 336 hours in winter inversion)
    snippet1 = df_preds1.iloc[100:436].copy()
    time_hrs1 = np.arange(len(snippet1))
    
    axes[0].plot(time_hrs1, snippet1['y_true'], 'k-', linewidth=2.0, label='Regulatory BAM Ground Truth (Met One)', zorder=5)
    axes[0].plot(time_hrs1, snippet1['Raw Sensor Baseline'], color='crimson', linestyle='--', alpha=0.6, linewidth=1.2, label='Raw PurpleAir (PMS5003)')
    axes[0].plot(time_hrs1, snippet1['US EPA Linear Standard'], color='gray', linestyle=':', linewidth=1.4, label='US EPA Linear Baseline')
    axes[0].plot(time_hrs1, snippet1['Random Forest Regressor'], color='forestgreen', alpha=0.85, linewidth=1.5, label='Random Forest (Methodology 1)')
    axes[0].plot(time_hrs1, snippet1['Gradient Boosted Decision Trees (GBDT/XGB)'], color='dodgerblue', alpha=0.9, linewidth=1.6, label='Gradient Boosted Trees (Methodology 2)')
    axes[0].plot(time_hrs1, snippet1['Support Vector Regression (SVR - RBF)'], color='darkorange', alpha=0.8, linewidth=1.4, label='SVR RBF (Methodology 3)')
    axes[0].set_title('(a) Dataset 1: PurpleAir Sensor Network in Urban India (Winter Testing Window)', fontweight='bold')
    axes[0].set_ylabel('PM2.5 Concentration ($\mu g/m^3$)')
    axes[0].set_xlabel('Elapsed Observation Time (Hours)')
    axes[0].legend(loc='upper right', frameon=True, framealpha=0.9)
    axes[0].set_ylim(0, max(snippet1['Raw Sensor Baseline'].max() * 1.05, 250))
    
    # Dataset 2 Italy snippet (14 days / 336 hours)
    snippet2 = df_preds2.iloc[100:436].copy()
    time_hrs2 = np.arange(len(snippet2))
    
    axes[1].plot(time_hrs2, snippet2['y_true'], 'k-', linewidth=2.0, label='Regulatory Ground Truth (ARPA Piemonte)', zorder=5)
    axes[1].plot(time_hrs2, snippet2['Raw Sensor Baseline'], color='crimson', linestyle='--', alpha=0.6, linewidth=1.2, label='Raw Honeywell (HPMA115C0 Median)')
    axes[1].plot(time_hrs2, snippet2['US EPA Linear Standard'], color='gray', linestyle=':', linewidth=1.4, label='US EPA Linear Baseline')
    axes[1].plot(time_hrs2, snippet2['Random Forest Regressor'], color='forestgreen', alpha=0.85, linewidth=1.5, label='Random Forest (Methodology 1)')
    axes[1].plot(time_hrs2, snippet2['Gradient Boosted Decision Trees (GBDT/XGB)'], color='dodgerblue', alpha=0.9, linewidth=1.6, label='Gradient Boosted Trees (Methodology 2)')
    axes[1].plot(time_hrs2, snippet2['Support Vector Regression (SVR - RBF)'], color='darkorange', alpha=0.8, linewidth=1.4, label='SVR RBF (Methodology 3)')
    axes[1].set_title('(b) Dataset 2: Q-SCALE Low-Cost Sensor Benchmark (Turin, Italy Testing Window)', fontweight='bold')
    axes[1].set_ylabel('PM2.5 Concentration ($\mu g/m^3$)')
    axes[1].set_xlabel('Elapsed Observation Time (Hours)')
    axes[1].legend(loc='upper right', frameon=True, framealpha=0.9)
    axes[1].set_ylim(0, max(snippet2['Raw Sensor Baseline'].max() * 1.05, 80))
    
    plt.tight_layout()
    plt.savefig('figure1_time_series_calibration.png', dpi=300)
    plt.close()
    print("Saved figure1_time_series_calibration.png")
    
    # -------------------------------------------------------------
    # Figure 2: 1:1 Scatter Parity Plots
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    models_to_plot = [
        ('Raw Sensor Baseline', 'crimson'),
        ('Random Forest Regressor', 'forestgreen'),
        ('Gradient Boosted Decision Trees (GBDT/XGB)', 'dodgerblue'),
        ('Support Vector Regression (SVR - RBF)', 'darkorange')
    ]
    
    for col_idx, (m_name, color) in enumerate(models_to_plot):
        # Top Row: Dataset 1 India
        ax_top = axes[0, col_idx]
        y_t = df_preds1['y_true'].values
        y_p = df_preds1[m_name].values
        r2 = r2_score(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        
        ax_top.scatter(y_t, y_p, c=color, alpha=0.25, s=12, edgecolors='none')
        max_val1 = max(np.percentile(y_t, 99.5), np.percentile(y_p, 99.5)) * 1.05
        ax_top.plot([0, max_val1], [0, max_val1], 'k--', linewidth=1.5, label='1:1 Parity')
        ax_top.plot([0, max_val1], [0, max_val1 * 1.2], 'k:', alpha=0.5, label='+20% Error')
        ax_top.plot([0, max_val1], [0, max_val1 * 0.8], 'k:', alpha=0.5, label='-20% Error')
        
        # Fit trend line
        m_fit, b_fit = np.polyfit(y_t, y_p, 1)
        ax_top.plot([0, max_val1], [b_fit, m_fit * max_val1 + b_fit], color=color, linewidth=1.8, label=f'Fit: y={m_fit:.2f}x+{b_fit:.1f}')
        
        ax_top.set_xlim(0, max_val1)
        ax_top.set_ylim(0, max_val1)
        ax_top.set_title(f"Dataset 1: {m_name}\n$R^2$ = {r2:.3f} | RMSE = {rmse:.1f} $\mu g/m^3$", fontsize=10, fontweight='bold')
        ax_top.set_xlabel('Reference PM2.5 ($\mu g/m^3$)')
        if col_idx == 0:
            ax_top.set_ylabel('Predicted PM2.5 ($\mu g/m^3$)')
        ax_top.legend(loc='upper left', fontsize=8)
        
        # Bottom Row: Dataset 2 Italy
        ax_bot = axes[1, col_idx]
        y_t2 = df_preds2['y_true'].values
        y_p2 = df_preds2[m_name].values
        r2_2 = r2_score(y_t2, y_p2)
        rmse_2 = np.sqrt(mean_squared_error(y_t2, y_p2))
        
        ax_bot.scatter(y_t2, y_p2, c=color, alpha=0.3, s=14, edgecolors='none')
        max_val2 = max(np.percentile(y_t2, 99.5), np.percentile(y_p2, 99.5)) * 1.05
        ax_bot.plot([0, max_val2], [0, max_val2], 'k--', linewidth=1.5, label='1:1 Parity')
        ax_bot.plot([0, max_val2], [0, max_val2 * 1.2], 'k:', alpha=0.5)
        ax_bot.plot([0, max_val2], [0, max_val2 * 0.8], 'k:', alpha=0.5)
        
        m_fit2, b_fit2 = np.polyfit(y_t2, y_p2, 1)
        ax_bot.plot([0, max_val2], [b_fit2, m_fit2 * max_val2 + b_fit2], color=color, linewidth=1.8, label=f'Fit: y={m_fit2:.2f}x+{b_fit2:.1f}')
        
        ax_bot.set_xlim(0, max_val2)
        ax_bot.set_ylim(0, max_val2)
        ax_bot.set_title(f"Dataset 2: {m_name}\n$R^2$ = {r2_2:.3f} | RMSE = {rmse_2:.1f} $\mu g/m^3$", fontsize=10, fontweight='bold')
        ax_bot.set_xlabel('Reference PM2.5 ($\mu g/m^3$)')
        if col_idx == 0:
            ax_bot.set_ylabel('Predicted PM2.5 ($\mu g/m^3$)')
        ax_bot.legend(loc='upper left', fontsize=8)
        
    plt.tight_layout()
    plt.savefig('figure2_scatter_parity_plots.png', dpi=300)
    plt.close()
    print("Saved figure2_scatter_parity_plots.png")
    
    # -------------------------------------------------------------
    # Figure 3: RH Stratified Residual Analysis (Hygroscopic Suppression)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter for the key comparison models
    plot_models = ['Raw Sensor Baseline', 'US EPA Linear Standard', 'Random Forest Regressor', 'Gradient Boosted Decision Trees (GBDT/XGB)', 'Support Vector Regression (SVR - RBF)']
    rh_sub = df_rh_all[df_rh_all['Model'].isin(plot_models)].copy()
    
    palette = {
        'Raw Sensor Baseline': '#d62728',
        'US EPA Linear Standard': '#7f7f7f',
        'Random Forest Regressor': '#2ca02c',
        'Gradient Boosted Decision Trees (GBDT/XGB)': '#1f77b4',
        'Support Vector Regression (SVR - RBF)': '#ff7f0e'
    }
    
    for i, d_name in enumerate(['Dataset 1 (PurpleAir India)', 'Dataset 2 (Q-SCALE Italy)']):
        d_sub = rh_sub[rh_sub['Dataset'] == d_name]
        sns.barplot(data=d_sub, x='Bracket', y='MBE', hue='Model', ax=axes[i], palette=palette)
        axes[i].axhline(0, color='black', linestyle='--', linewidth=1.0)
        axes[i].set_title(f"Hygroscopic Mean Bias Error (MBE) by RH Strata\n{d_name}", fontweight='bold')
        axes[i].set_xlabel('Relative Humidity Bracket')
        axes[i].set_ylabel('Mean Bias Error (MBE, $\mu g/m^3$)')
        axes[i].tick_params(axis='x', rotation=15)
        if i == 0:
            axes[i].legend(loc='upper left', fontsize=8)
        else:
            axes[i].legend([], frameon=False)
            
    plt.tight_layout()
    plt.savefig('figure3_rh_stratified_residuals.png', dpi=300)
    plt.close()
    print("Saved figure3_rh_stratified_residuals.png")
    
    # -------------------------------------------------------------
    # Figure 4: Feature Ablation Comparison (Sensor-Only vs Sensor+Env)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Compare RMSE and R2 across feature sets
    ablation_df = df_results[~df_results['Model'].isin(['Raw Sensor Baseline', 'US EPA Linear Standard'])].copy()
    
    # Rename models for cleaner labels
    model_short = {
        'Multiple Linear Regression (MLR)': 'MLR',
        'Random Forest Regressor': 'Random Forest',
        'Gradient Boosted Decision Trees (GBDT/XGB)': 'GBDT (XGB)',
        'Support Vector Regression (SVR - RBF)': 'SVR (RBF)'
    }
    ablation_df['Model_Short'] = ablation_df['Model'].map(model_short)
    
    # Panel (a): RMSE comparison
    sns.barplot(data=ablation_df, x='Model_Short', y='RMSE', hue='Feature_Set', ax=axes[0], palette=['#aec7e8', '#1f77b4'])
    axes[0].set_title('(a) RMSE Reduction via Environmental Covariates', fontweight='bold')
    axes[0].set_ylabel('Test RMSE ($\mu g/m^3$)')
    axes[0].set_xlabel('Calibration Model')
    axes[0].legend(title='Feature Set')
    
    # Panel (b): R2 comparison
    sns.barplot(data=ablation_df, x='Model_Short', y='R2', hue='Feature_Set', ax=axes[1], palette=['#ffbb78', '#ff7f0e'])
    axes[1].set_title('(b) Coefficient of Determination ($R^2$) Improvement', fontweight='bold')
    axes[1].set_ylabel('Test $R^2$')
    axes[1].set_xlabel('Calibration Model')
    axes[1].legend(title='Feature Set')
    axes[1].set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig('figure4_feature_ablation_comparison.png', dpi=300)
    plt.close()
    print("Saved figure4_feature_ablation_comparison.png")
    
    # -------------------------------------------------------------
    # Figure 5: Computational Efficiency vs Accuracy Pareto Frontiers
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    
    env_results = df_results[(df_results['Feature_Set'] == 'Sensor+Environmental') & (df_results['Dataset'] == 'Dataset 1 (PurpleAir India)')].copy()
    
    markers = {'Multiple Linear Regression (MLR)': 'o', 'Random Forest Regressor': 's', 
               'Gradient Boosted Decision Trees (GBDT/XGB)': '^', 'Support Vector Regression (SVR - RBF)': 'D'}
    colors = {'Multiple Linear Regression (MLR)': '#7f7f7f', 'Random Forest Regressor': '#2ca02c', 
              'Gradient Boosted Decision Trees (GBDT/XGB)': '#1f77b4', 'Support Vector Regression (SVR - RBF)': '#ff7f0e'}
    
    for _, row in env_results.iterrows():
        m = row['Model']
        if m in markers:
            ax.scatter(row['Inference_Latency_us'], row['RMSE'], 
                       s=max(row['Model_Size_KB'] / 2.5, 60), 
                       c=colors[m], marker=markers[m], edgecolors='black', linewidth=1.5, alpha=0.85,
                       label=f"{model_short[m]} (Size: {row['Model_Size_KB']:.0f} KB)")
            # Add text annotation
            ax.annotate(f"{model_short[m]}\n(RMSE: {row['RMSE']:.1f}, {row['Inference_Latency_us']:.1f} $\mu s$)",
                        (row['Inference_Latency_us'] * 1.08, row['RMSE']),
                        fontsize=9, fontweight='semibold')
            
    ax.set_xscale('log')
    ax.set_title('Edge Computing Pareto Frontier: Inference Latency vs. Prediction RMSE\n(Bubble area proportional to model memory footprint on disk)', fontweight='bold')
    ax.set_xlabel('Inference Latency per Sample ($\mu s$, Log Scale)')
    ax.set_ylabel('Test Root Mean Squared Error (RMSE, $\mu g/m^3$)')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig('figure5_computational_tradeoffs.png', dpi=300)
    plt.close()
    print("Saved figure5_computational_tradeoffs.png")

def main():
    print("=========================================================================")
    print("STARTING FULL MACHINE LEARNING AIR QUALITY SENSOR CALIBRATION BENCHMARK")
    print("=========================================================================")
    
    df1 = prepare_dataset1()
    df2 = prepare_dataset2()
    
    # Run experiments on Dataset 1 (India)
    res1, preds1, test_df1 = run_experiment_on_dataset(df1, 'Dataset 1 (PurpleAir India)', raw_col='PM2.5_raw')
    rh_res1, conc_res1 = compute_stratified_metrics(preds1, 'Dataset 1 (PurpleAir India)')
    
    # Run experiments on Dataset 2 (Italy)
    res2, preds2, test_df2 = run_experiment_on_dataset(df2, 'Dataset 2 (Q-SCALE Italy)', raw_col='PM2.5_raw')
    rh_res2, conc_res2 = compute_stratified_metrics(preds2, 'Dataset 2 (Q-SCALE Italy)')
    
    # Aggregate results
    all_results = pd.concat([res1, res2], ignore_index=True)
    all_rh = pd.concat([rh_res1, rh_res2], ignore_index=True)
    all_conc = pd.concat([conc_res1, conc_res2], ignore_index=True)
    
    # Save CSV outputs
    all_results.to_csv('results_overall_benchmark.csv', index=False)
    all_rh.to_csv('results_rh_stratified.csv', index=False)
    all_conc.to_csv('results_concentration_stratified.csv', index=False)
    print("\nSaved all tabular results to CSV:")
    print(" - results_overall_benchmark.csv")
    print(" - results_rh_stratified.csv")
    print(" - results_concentration_stratified.csv")
    
    # Generate 5 Publication Figures
    plot_publication_figures(preds1, preds2, all_results, all_rh)
    
    # Display formatted tables in terminal
    print("\n" + "="*80)
    print("FINAL QUANTITATIVE BENCHMARK SUMMARY TABLE (SENSOR + ENVIRONMENTAL)")
    print("="*80)
    summary_table = all_results[all_results['Feature_Set'].isin(['Sensor+Environmental', 'Sensor+Env (Fixed EPA)', 'Sensor-Only']) & 
                                (all_results['Model'] != 'Multiple Linear Regression (MLR)')]
    cols_to_show = ['Dataset', 'Model', 'Feature_Set', 'RMSE', 'MAE', 'R2', 'MBE', 'NRMSE_%', 'Inference_Latency_us', 'Model_Size_KB']
    print(all_results[cols_to_show].to_string(index=False))
    
    print("\n=========================================================================")
    print("ALL BENCHMARK EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=========================================================================")

if __name__ == '__main__':
    main()
