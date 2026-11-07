"""
Benchmark Dataset Generator for Low-Cost Air Quality Sensor Calibration
Faithfully implements empirical and physical characteristics described in:
- Campmier et al. (2023), Atmos. Meas. Tech. (Dryad: 10.6078/D1RQ70)
- Chiavassa, Giusto, & Bergadano (2024), IEEE QCE / Zenodo 10.5281/zenodo.11068317
- Barkjohn et al. (2021), Atmos. Meas. Tech. (US EPA National Benchmark)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

def generate_purpleair_india(num_hours=8760):
    """
    Simulates co-located PurpleAir PA-II (Plantower PMS5003 dual channels A & B)
    vs. Regulatory Beta Attenuation Monitor (Met One BAM 1020/1022) across
    contrasting Indian environments (Delhi Urban, Hamirpur Rural, Bengaluru Plateau).
    """
    start_date = datetime(2019, 1, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(hours=i) for i in range(num_hours)]
    
    # 3 distinct co-location sites with differing aerosol composition & climate
    sites = ['Delhi', 'Hamirpur', 'Bengaluru']
    dfs = []
    
    for site in sites:
        hours = np.arange(num_hours)
        day_of_year = (hours // 24) % 365
        hour_of_day = hours % 24
        
        if site == 'Delhi':
            # Sub-tropical mega-city with severe winter smog & post-monsoon crop burning
            # High base PM2.5, extreme winter peaks (Nov-Jan), monsoon washout (Jul-Aug)
            seasonal_base = 95.0 + 75.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0)
            # Monsoon dip (days 180 to 240)
            monsoon_mask = (day_of_year >= 180) & (day_of_year <= 245)
            seasonal_base[monsoon_mask] *= 0.35
            
            # Diurnal cycle: morning rush hour (8-10 AM) and evening boundary layer collapse (8-11 PM)
            diurnal = 25.0 * np.cos(2 * np.pi * (hour_of_day - 8) / 24.0) + 15.0 * np.cos(4 * np.pi * (hour_of_day - 20) / 24.0)
            
            # Autoregressive component for multi-day stagnation events
            ar_noise = np.zeros(num_hours)
            noise = np.random.normal(0, 12.0, num_hours)
            for t in range(1, num_hours):
                ar_noise[t] = 0.85 * ar_noise[t-1] + noise[t]
                
            pm25_ref = np.clip(seasonal_base + diurnal + ar_noise, 8.0, 480.0)
            
            # Temperature: hot pre-monsoon summer (Apr-Jun > 38C), cool winter (Dec-Jan ~8-15C)
            temp_ambient = 25.0 - 12.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0) + 6.0 * np.cos(2 * np.pi * (hour_of_day - 14) / 24.0) + np.random.normal(0, 1.5, num_hours)
            
            # Relative Humidity: high monsoon (80-95%) and foggy winter mornings (80-95%), dry summer (25-40%)
            rh_base = 55.0 - 15.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0)
            rh_base[monsoon_mask] += 25.0
            rh_diurnal = -18.0 * np.cos(2 * np.pi * (hour_of_day - 14) / 24.0)
            rh = np.clip(rh_base + rh_diurnal + np.random.normal(0, 4.0, num_hours), 15.0, 98.0)
            
            # Deliquescence factor for urban sulfate/nitrate rich aerosol
            c_h = 0.28
            
        elif site == 'Hamirpur':
            # Rural Himalayan foothills, lower base PM2.5, biogenic & woodsmoke aerosols
            seasonal_base = 40.0 + 25.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0)
            monsoon_mask = (day_of_year >= 180) & (day_of_year <= 245)
            seasonal_base[monsoon_mask] *= 0.45
            
            diurnal = 12.0 * np.cos(2 * np.pi * (hour_of_day - 9) / 24.0)
            ar_noise = np.zeros(num_hours)
            noise = np.random.normal(0, 6.0, num_hours)
            for t in range(1, num_hours):
                ar_noise[t] = 0.80 * ar_noise[t-1] + noise[t]
            pm25_ref = np.clip(seasonal_base + diurnal + ar_noise, 5.0, 160.0)
            
            temp_ambient = 18.0 - 10.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0) + 5.0 * np.cos(2 * np.pi * (hour_of_day - 14) / 24.0) + np.random.normal(0, 1.2, num_hours)
            rh_base = 60.0 - 10.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0)
            rh_base[monsoon_mask] += 22.0
            rh_diurnal = -15.0 * np.cos(2 * np.pi * (hour_of_day - 14) / 24.0)
            rh = np.clip(rh_base + rh_diurnal + np.random.normal(0, 3.5, num_hours), 20.0, 96.0)
            c_h = 0.22
            
        else: # Bengaluru
            # Southern tropical plateau, moderate climate, vehicular and construction dust
            seasonal_base = 32.0 + 10.0 * np.cos(2 * np.pi * (day_of_year - 30) / 365.0)
            monsoon_mask = (day_of_year >= 160) & (day_of_year <= 250)
            seasonal_base[monsoon_mask] *= 0.60
            
            diurnal = 10.0 * np.cos(2 * np.pi * (hour_of_day - 9) / 24.0)
            ar_noise = np.zeros(num_hours)
            noise = np.random.normal(0, 4.5, num_hours)
            for t in range(1, num_hours):
                ar_noise[t] = 0.78 * ar_noise[t-1] + noise[t]
            pm25_ref = np.clip(seasonal_base + diurnal + ar_noise, 4.0, 95.0)
            
            temp_ambient = 24.0 - 4.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0) + 4.5 * np.cos(2 * np.pi * (hour_of_day - 14) / 24.0) + np.random.normal(0, 1.0, num_hours)
            rh_base = np.full(num_hours, 65.0)
            rh_base[monsoon_mask] += 18.0
            rh_diurnal = -14.0 * np.cos(2 * np.pi * (hour_of_day - 14) / 24.0)
            rh = np.clip(rh_base + rh_diurnal + np.random.normal(0, 3.5, num_hours), 30.0, 96.0)
            c_h = 0.20
            
        # Physical Hygroscopic Growth Factor f(RH) based on Hanel / Köhler equation
        # When RH > 60%, particles deliquesce and grow exponentially
        rh_ratio = rh / 100.0
        # Hygroscopic magnification factor: f(RH) = 1 + c_h * (RH / (100 - RH))^1.2 for RH > 50%
        hygro_growth = np.ones(num_hours)
        high_rh_idx = rh > 50.0
        hygro_growth[high_rh_idx] = 1.0 + c_h * np.power(rh_ratio[high_rh_idx] / (1.0001 - rh_ratio[high_rh_idx]), 1.15)
        hygro_growth = np.clip(hygro_growth, 1.0, 3.2)
        
        # Plantower PMS5003 raw optical response:
        # Optical baseline slope ~1.30 (due to PSL bead calibration assumption vs ambient density)
        alpha = 1.32
        
        # Dual channel simulation: Channel A and Channel B
        # Sensor A
        noise_a = np.random.normal(0, 1.5 + 0.03 * pm25_ref, num_hours)
        pm25_cf1_a = np.clip(alpha * pm25_ref * hygro_growth + noise_a, 0.0, 999.0)
        
        # Sensor B (independent laser/detector pair with slight manufacturing variance)
        alpha_b = alpha * np.random.normal(1.0, 0.03)
        noise_b = np.random.normal(0, 1.5 + 0.03 * pm25_ref, num_hours)
        pm25_cf1_b = np.clip(alpha_b * pm25_ref * hygro_growth + noise_b, 0.0, 999.0)
        
        # Introduce transient channel divergence / clogs in 1.2% of data to test QA/QC channel check
        clog_indices = np.random.choice(num_hours, size=int(0.012 * num_hours), replace=False)
        pm25_cf1_a[clog_indices] += np.random.uniform(25.0, 60.0, size=len(clog_indices))
        
        # Atmospheric formula (PM2.5_ATM): Plantower proprietary lower-concentration curve
        pm25_atm_a = np.where(pm25_cf1_a < 30.0, pm25_cf1_a, 0.66 * pm25_cf1_a + 10.0)
        pm25_atm_b = np.where(pm25_cf1_b < 30.0, pm25_cf1_b, 0.66 * pm25_cf1_b + 10.0)
        
        # Onboard sensor temperature offset (+2.5 to +4.0 C due to internal electronic heating)
        temp_onboard = temp_ambient + 3.2 + np.random.normal(0, 0.3, num_hours)
        
        df_site = pd.DataFrame({
            'timestamp': timestamps,
            'site': site,
            'PM2.5_ref': np.round(pm25_ref, 2),
            'PM2.5_CF1_A': np.round(pm25_cf1_a, 2),
            'PM2.5_CF1_B': np.round(pm25_cf1_b, 2),
            'PM2.5_ATM_A': np.round(pm25_atm_a, 2),
            'PM2.5_ATM_B': np.round(pm25_atm_b, 2),
            'temperature': np.round(temp_onboard, 2),
            'relative_humidity': np.round(rh, 2)
        })
        dfs.append(df_site)
        
    df_india = pd.concat(dfs, ignore_index=True)
    df_india.sort_values(by=['timestamp', 'site'], inplace=True)
    return df_india


def generate_qscale_italy(num_hours=5500):
    """
    Simulates Q-SCALE benchmark dataset (Turin, Italy - Chiavassa et al., 2024 / Zenodo 11068317)
    Co-located 4-Honeywell HPMA115C0 sensor units, DHT22 (T, RH), BME280 (Pressure)
    vs ARPA Piemonte official regulatory station (BAM/gravimetric).
    Period: Nov 2022 to June 2023 (~5500 hours).
    """
    start_date = datetime(2022, 11, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(hours=i) for i in range(num_hours)]
    hours = np.arange(num_hours)
    day_of_year = (hours // 24)
    hour_of_day = hours % 24
    
    # European continental sub-alpine urban environment:
    # High winter pollution (inversion in Po Valley, Nov-Feb: 30-100 ug/m3)
    # Clean spring/summer (March-June: 5-25 ug/m3)
    winter_factor = np.clip(1.0 - (day_of_year / 150.0), 0.0, 1.0)
    base_pm25 = 14.0 + 36.0 * winter_factor
    
    # Boundary layer diurnal stagnation: higher in early morning (7-9 AM) and evening (8-10 PM)
    diurnal = 8.0 * np.cos(2 * np.pi * (hour_of_day - 8) / 24.0) + 5.0 * np.cos(4 * np.pi * (hour_of_day - 20) / 24.0)
    
    # Autoregressive multi-day smog inversion episodes
    ar_noise = np.zeros(num_hours)
    noise = np.random.normal(0, 4.0, num_hours)
    for t in range(1, num_hours):
        ar_noise[t] = 0.88 * ar_noise[t-1] + noise[t]
        
    pm25_ref = np.clip(base_pm25 + diurnal + ar_noise, 2.0, 115.0)
    
    # Meteorological variables in Turin:
    # Sub-alpine winter: T ~ -3 to 8 C, Spring: T ~ 12 to 28 C
    temp = 3.0 + 18.0 * (day_of_year / 230.0) + 4.5 * np.cos(2 * np.pi * (hour_of_day - 14) / 24.0) + np.random.normal(0, 1.1, num_hours)
    
    # RH: High in winter (70-92%), lower in spring (35-65%)
    rh_base = 78.0 - 22.0 * (day_of_year / 230.0)
    rh_diurnal = -12.0 * np.cos(2 * np.pi * (hour_of_day - 14) / 24.0)
    rh = np.clip(rh_base + rh_diurnal + np.random.normal(0, 3.5, num_hours), 25.0, 94.0)
    
    # Barometric pressure (BME280): Winter anticyclone (1015-1030 hPa = strong inversion), spring depressions (985-1010 hPa)
    pressure = 1018.0 - 8.0 * (day_of_year / 230.0) + np.random.normal(0, 3.5, num_hours)
    
    # Honeywell HPMA115C0 optical response:
    # Optical baseline slope ~1.18, lower optical hygroscopic growth coefficient than Delhi dust (c_h ~ 0.17)
    c_h = 0.17
    rh_ratio = rh / 100.0
    hygro_growth = np.ones(num_hours)
    high_rh_idx = rh > 55.0
    hygro_growth[high_rh_idx] = 1.0 + c_h * np.power(rh_ratio[high_rh_idx] / (1.0001 - rh_ratio[high_rh_idx]), 1.10)
    hygro_growth = np.clip(hygro_growth, 1.0, 2.5)
    
    # Simulate 4 independent Honeywell sensors per station unit and take median
    sensor_readings = []
    for s in range(4):
        s_alpha = 1.16 + np.random.normal(0, 0.02)
        s_noise = np.random.normal(0, 1.2 + 0.02 * pm25_ref, num_hours)
        s_val = np.clip(s_alpha * pm25_ref * hygro_growth + s_noise, 0.0, 500.0)
        sensor_readings.append(s_val)
        
    sensor_median = np.median(sensor_readings, axis=0)
    
    df_qscale = pd.DataFrame({
        'timestamp': timestamps,
        'station_id': 'Station_1_Turin',
        'PM2.5_ref': np.round(pm25_ref, 2),
        'PM2.5_sensor_median': np.round(sensor_median, 2),
        'temperature': np.round(temp, 2),
        'relative_humidity': np.round(rh, 2),
        'pressure_hPa': np.round(pressure, 1)
    })
    return df_qscale

if __name__ == '__main__':
    print('Generating Dataset 1: PurpleAir Sensor Network in Urban & Rural India...')
    df_india = generate_purpleair_india(num_hours=8760)
    india_path = 'dataset1_purpleair_india.csv'
    df_india.to_csv(india_path, index=False)
    print(f'Saved {india_path} with {len(df_india)} rows. Sites: {df_india["site"].unique().tolist()}')
    print('India Summary Statistics:')
    print(df_india[['PM2.5_ref', 'PM2.5_CF1_A', 'PM2.5_CF1_B', 'temperature', 'relative_humidity']].describe().round(2))
    
    print('\nGenerating Dataset 2: Q-SCALE Benchmark Dataset (Turin, Italy)...')
    df_qscale = generate_qscale_italy(num_hours=5500)
    qscale_path = 'dataset2_qscale_italy.csv'
    df_qscale.to_csv(qscale_path, index=False)
    print(f'Saved {qscale_path} with {len(df_qscale)} rows.')
    print('Q-SCALE Summary Statistics:')
    print(df_qscale[['PM2.5_ref', 'PM2.5_sensor_median', 'temperature', 'relative_humidity', 'pressure_hPa']].describe().round(2))
    print('\nBenchmark datasets generated successfully!')
