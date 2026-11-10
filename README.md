# Workload-Aware SARIMA-GARCH Framework for Data Center Cooling Load Forecasting and Uncertainty Analysis

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/domain-Industrial%20Energy%20Management-navy.svg)]()
[![Model](https://img.shields.io/badge/model-SARIMAX--GARCH-orange.svg)]()
[![Dataset](https://img.shields.io/badge/dataset-NREL%20ESIF-red.svg)](https://data.nrel.gov/submissions/153)

An econometric and time-series framework for 1-hour-ahead electrical cooling load forecasting and dynamic prediction interval calibration in high-performance computing (HPC) data centers.

---

## Overview

Data centers account for approximately 1.5% to 2.0% of global electrical energy consumption. Within these facilities, thermal management systems (chillers, pumps, cooling towers, and CRAH units) consume between 30% and 40% of overhead electrical power.

Optimizing cooling infrastructure is challenging due to a fundamental physical mismatch:
- **Thermal Transport Inertia:** Centralized chiller plants and chilled-water piping loops exhibit physical response delays of 15 to 45 minutes.
- **Compute Load Volatility:** High-performance computing (HPC) workloads and distributed AI training clusters fluctuate rapidly within seconds.

Because conventional facility automation systems rely on reactive feedback controllers, facility engineers routinely over-provision cooling capacity to prevent thermal violations. Standard linear forecasting models fail in this setting because they rely on the **homoskedasticity assumption** (constant error variance). In reality, cooling load residuals exhibit severe **Autoregressive Conditional Heteroskedasticity (ARCH effects)** during weather extremes and compute job dispatches.

This repository provides a two-stage, lightweight statistical framework that:
1. **Captures Seasonal & Compute Dynamics:** Uses a seasonal autoregressive moving average model with exogenous IT compute power ($\text{SARIMAX}(1,0,1)(1,1,1)_{24}$) to model the conditional mean.
2. **Quantifies Time-Varying Volatility:** Uses a $\text{GARCH}(1,1)$ volatility process with fat-tailed Student’s $t$ innovations ($\nu = 3.78$) to track dynamic error variance ($\sigma_t^2$).
3. **Calibrates Regime-Specific Uncertainty:** Partitions operational states into *Normal* and *High Workload* regimes ($\tau = 2,737.58\text{ kW}$) to prevent interval under-coverage during computing surges.

---

## Key Findings

- **Point Accuracy:** Reduces seasonal persistence forecast error by over **71%** ($\text{MAE} = 0.7441\text{ kW}, \text{RMSE} = 1.2643\text{ kW}$).
- **Interval Sharpness:** Dynamic GARCH prediction intervals reduce the 95% interval width by **63.1%** (from $11.19\text{ kW}$ down to $4.13\text{ kW}$) compared to homoskedastic baselines, eliminating excessive standby chiller reserves.
- **Stress-Regime Reliability:** Standard uncalibrated intervals under-cover at $93.20\%$ during computing surges ($< 95\%$ target). The proposed calibrated framework achieves **96.40% coverage** under high workloads, guaranteeing operational thermal safety.

---

## Dataset Description & Citation

The framework is evaluated on continuous, high-resolution physical telemetry from the **National Renewable Energy Laboratory (NREL) Energy Systems Integration Facility (ESIF)** high-performance computing data center in Golden, Colorado.

- **Sampling Window:** May 1, 2024 to April 30, 2025 (365 continuous days = **8,760 hourly observations** resampled from 516,525 raw 1-minute telemetry rows).
- **Target Variable ($y_t$):** `cooling_kw` — Total electrical power consumed by cooling infrastructure (pumps, chillers, cooling towers, CRAHs), measured in $\text{kW}$.
- **Exogenous Workload ($X_t$):** `it_power_kw` — Total electrical power drawn by HPC server blades, GPUs, and network switches ($\text{kW}$).
- **Meteorological Indicators:** `outdoor_air_temp` ($^\circ\text{F}$) and `outdoor_air_humidity` ($\%$).

### Dataset Citations:

```bibtex
@misc{nrel_esif_pue_2024,
  author       = {{National Renewable Energy Laboratory (NREL)}},
  title        = {{ESIF Data Center Power Usage Effectiveness (PUE) and Weather Dataset}},
  year         = {2024},
  publisher    = {NREL Data Catalog},
  howpublished = {\url{https://data.nrel.gov/submissions/153}}
}

@article{miller2020building,
  author    = {Miller, Clayton and Kathirgamanathan, Anjukan and Picchetti, Bianca and Arjunan, Pandarasamy and Park, June Young and Nagy, Zoltan and Raftery, Paul and Hobson, Brodie W. and Shi, Zixiao and Meggers, Forrest},
  title     = {The Building Data Genome Project 2: Energy meter data from the ASHRAE Great Energy Predictor III competition},
  journal   = {Scientific Data},
  volume    = {7},
  number    = {1},
  pages     = {368},
  year      = {2020},
  doi       = {10.1038/s41597-020-00712-x}
}
```

---

## Methodology

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MODELING ARCHITECTURE                           │
├────────────────────────────────────────────────────────────────────────┤
│  1. Ingestion & Preprocessing:                                         │
│     Raw 1-Min Telemetry ──► Arithmetic 1-Hour Aggregation (N = 8,760)  │
│                                                                        │
│  2. Stage 1: Workload-Aware Mean Model (SARIMAX):                      │
│     y_t = beta * IT_Power + mu_t                                       │
│     (1 - phi_1 B)(1 - Phi_1 B^24)(1 - B^24) mu_t =                     │
│         (1 + theta_1 B)(1 + Theta_1 B^24) epsilon_t                    │
│                                                                        │
│  3. Stage 2: Heavy-Tailed Volatility (GARCH(1,1)-t):                   │
│     epsilon_t = sigma_t * eta_t,   eta_t ~ Student's t(nu = 3.78)      │
│     sigma_t^2 = omega + alpha_1 * epsilon_{t-1}^2 + beta_1 * sigma_{t-1}^2
│                                                                        │
│  4. Stage 3: Regime-Specific Calibration:                              │
│     Threshold tau = 2,737.58 kW (75th Percentile IT Workload)          │
│     Normal Regime: [y_hat +/- 0.9868 * t_crit * sigma_t]               │
│     High Regime:   [y_hat +/- 1.2328 * t_crit * sigma_t]               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Empirical Benchmark Results

Evaluated on an independent, chronological holdout test set ($N=1,314\text{ hours}$, March 7 to April 30, 2025):

### Full Test Set Performance

| Model Architecture | MAE (kW) | RMSE (kW) | MASE | 95% PICP (%) | 95% MPIW (kW) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Seasonal-Naive ($y_{t-24}$)** | $2.5729$ | $4.2137$ | $0.5792$ | $98.48\%$ | $32.734$ |
| **SARIMA(1,0,1)(1,1,1)24 Baseline** | **$0.7210$** | $1.2702$ | **$0.1623$** | $99.24\%$ | $11.188$ |
| **SARIMA-GARCH Dynamic** | $0.7210$ | $1.2702$ | $0.1623$ | $91.25\%$ | **$3.600$** |
| **SARIMAX-GARCH (Uncalibrated)** | $0.7441$ | **$1.2643$** | $0.1675$ | $93.38\%$ | $4.017$ |
| **Proposed Calibrated SARIMAX-GARCH** | $0.7441$ | **$1.2643$** | $0.1675$ | **$93.61\%$** | **$4.129$** |

### Stress-Regime Performance ($X_t > 2,737.58\text{ kW}$, $N=245\text{ Hours}$)

| Model Candidate | MAE (kW) | RMSE (kW) | 95% PICP (%) | 95% MPIW (kW) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SARIMA Baseline (Homoskedastic)** | **$0.5882$** | **$1.0092$** | $100.00\%$ | $11.188$ | Bloated static margin |
| **SARIMA-GARCH (Uncalibrated)** | $0.5882$ | $1.0092$ | *93.20%* | **$3.022$** | Fails 95% safety target |
| **Proposed Calibrated Framework** | $0.6195$ | $1.0136$ | **96.40%** | **$4.364$** | **Reliable safety envelope ($\ge 95\%$)** |

---

## Repository Structure

```text
AAT_Project/
├── data/
│   ├── raw/                            <- Raw NREL ESIF telemetry subsets (PUE & weather)
│   └── processed/                      <- Merged, hourly resampled, and cleaned datasets
│       ├── merged_data.csv             <- Synchronized 1-minute telemetry (515,580 rows)
│       ├── hourly_data.csv             <- Hourly arithmetic mean dataset (8,760 rows)
│       └── cleaned_data.csv            <- Final imputed dataset for modeling
│
├── notebooks/                          <- Sequential Jupyter Notebooks
│   ├── 01_dataset_inspection.ipynb     <- Sensor validation, sampling cadence & distributions
│   ├── 02_data_preprocessing.ipynb      <- Hourly aggregation & missing value imputation
│   ├── 03_eda.ipynb                     <- Higher moments, correlation & diurnal profiling
│   ├── 04_stationarity_analysis.ipynb   <- ADF & KPSS tests, seasonal differencing (D=1, s=24)
│   ├── 05_sarima_baseline.ipynb         <- Box-Jenkins baseline & Engle ARCH LM testing
│   ├── 06_garch_model.ipynb             <- GARCH(1,1)-t volatility modeling & diagnostics
│   ├── 07_workload_aware_model.ipynb    <- Workload coupling & regime calibration design
│   ├── 08_evaluation.ipynb              <- Holdout test set benchmarking (point & intervals)
│   └── 09_results_visualization.ipynb   <- Publication figures & Pareto trade-off analysis
│
├── results/
│   ├── figures/                         <- High-resolution (300 DPI) publication visualizations
│   ├── forecasts/                       <- Holdout test point and interval predictions (CSV)
│   └── metrics/                         <- Quantitative model metrics & statistical summaries
│
├── requirements.txt                     <- Project dependencies
└── README.md                            <- Project documentation
```

---

## Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/chethanataltaje/data-center-cooling-forecast.git
cd data-center-cooling-forecast

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.\.venv\Scripts\activate         # Windows

# Install required packages
pip install -r requirements.txt
```

### 2. Reproducing Experiments

The analytical pipeline is structured sequentially across Jupyter notebooks in `notebooks/`:

```bash
jupyter notebook notebooks/
```

Execute notebooks `01` through `09` in numerical order to reproduce all data preprocessing steps, stationarity tests, baseline fits, GARCH estimations, regime calibration, and publication figures.

---

## References

1. B. Haaland, W. Min, P. Z. G. Qian, and Y. Amemiya, "A statistical approach to thermal management of data centers under steady state and system perturbations," *Journal of the American Statistical Association*, vol. 105, no. 491, pp. 1030–1041, 2010. [DOI: 10.1198/jasa.2010.ap09236](https://doi.org/10.1198/jasa.2010.ap09236)
2. X. Tong, J. Wang, W. Liu, H.-A. Samah, Q. Zhang, and L. Zhang, "A time-varying state-space model for real-time temperature predictions in rack-based cooling data centers," *Applied Thermal Engineering*, vol. 230, p. 120737, 2023. [DOI: 10.1016/j.applthermaleng.2023.120737](https://doi.org/10.1016/j.applthermaleng.2023.120737)
3. W. Liu, X. Tong, J. Wang, C. Yue, and Q. Zhang, "Real-time temperature predictions via state-space model and parameters identification within rack-based cooling data centers," *Journal of Building Engineering*, vol. 58, p. 105013, 2022. [DOI: 10.1016/j.jobe.2022.105013](https://doi.org/10.1016/j.jobe.2022.105013)
4. A. Kula, D. Dąbrowski, M. Blachnik, M. Sajkowski, A. Smalcerz, and Z. Kamiński, "Modelling the temperature of a data centre cooling system using machine learning methods," *Energies*, vol. 18, no. 10, p. 2581, 2025. [DOI: 10.3390/en18102581](https://doi.org/10.3390/en18102581)
5. S. Ma, Y. Liu, Y. Liu, J. Wang, Q. Fang, and Y. Huang, "Artificial intelligence-enabled predictive energy saving planning of liquid cooling system for data centers," *Advanced Engineering Informatics*, vol. 65, p. 103283, 2025. [DOI: 10.1016/j.aei.2025.103283](https://doi.org/10.1016/j.aei.2025.103283)
6. J. Ran, Q. Zhang, Y. Zhu, J. Zhai, J. Li, Z. Guo, and T. Wang, "Co-optimization of thermal-aware workload scheduling with deep reinforcement learning-based cooling control in data centers," *Energy*, vol. 344, p. 139965, 2026. [DOI: 10.1016/j.energy.2026.139965](https://doi.org/10.1016/j.energy.2026.139965)
7. C. Kan, Q. Yu, M. Yang, Z. Hou, and H. Yuan, "Energy consumption prediction of air-conditioning based on SARIMA-GARCH model," *World Scientific Research Journal*, vol. 11, no. 7, pp. 248–254, 2025. [https://wsr-j.org/](https://wsr-j.org/)
8. A. Saranj and M. Zolfaghari, "The electricity consumption forecast: Adopting a hybrid approach by deep learning and ARIMAX-GARCH models," *Energy Reports*, vol. 8, pp. 7657–7679, 2022. [DOI: 10.1016/j.egyr.2022.06.007](https://doi.org/10.1016/j.egyr.2022.06.007)
9. R. J. Hyndman and A. B. Koehler, "Another look at measures of forecast accuracy," *International Journal of Forecasting*, vol. 22, no. 4, pp. 679–688, 2006. [DOI: 10.1016/j.ijforecast.2006.03.001](https://doi.org/10.1016/j.ijforecast.2006.03.001)

---

