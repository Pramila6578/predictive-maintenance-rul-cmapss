# Predictive Maintenance: Remaining Useful Life (RUL) Prediction

Predicting how many operating cycles remain before a turbofan jet engine fails, using NASA's C-MAPSS turbofan degradation dataset (FD001).

## Problem Statement
Aircraft and industrial equipment degrade gradually during operation. Instead of waiting for failure or maintaining on a fixed calendar schedule, this project builds a Machine Learning model that estimates **Remaining Useful Life (RUL)** — the number of operating cycles left before failure — directly from sensor readings.

This mirrors real predictive-maintenance use cases in manufacturing: the same approach (tracking sensor drift over time to predict time-to-failure) applies to motors, rolling mills, and rotating machinery in industrial plants.

## Dataset
NASA C-MAPSS Turbofan Engine Degradation Simulation — subset FD001
- 100 training engines, 100 test engines
- 21 sensor measurements, 3 operational settings per engine per cycle
- Source: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) / also available on [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

## Approach
1. Data understanding and cleaning
2. Feature engineering — RUL calculated from run-to-failure histories
3. Exploratory Data Analysis — sensor variance, correlation with RUL, degradation trends
4. Feature selection — dropped 7 near-zero-variance sensors
5. Train/test split **by engine ID** (not random rows) to avoid data leakage across correlated time-series records
6. Model comparison: Linear Regression (baseline) vs. Random Forest

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Random Forest** | **30.81** | **23.51** | **0.780** |
| Linear Regression | 31.68 | 25.17 | 0.767 |

Random Forest improved RMSE by ~3% over the baseline. The top predictive features were `cycle` (63.5% importance) and `sensor_11` (16.2%).

## Business Impact
An RMSE of ~31 cycles means predictions are typically off by about 31 operating cycles. For engines with a typical lifespan of 130–360 cycles, this provides a meaningful early-warning window — enough to shift maintenance from a fixed schedule to a condition-based one, reducing unplanned downtime.

## Tech Stack
Python · Pandas · NumPy · Matplotlib · Scikit-learn

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook CMAPSS_RUL_Prediction.ipynb
```
The dataset (`train_FD001.txt`) is included at the root of this repo — no separate download needed. Original source: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) / [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps).

## About
Built as part of a self-directed data analytics portfolio, connecting 5+ years of hands-on industrial automation (PLC/SCADA/Level 2 systems) experience to applied Machine Learning.

**Author:** Chaitanya Naga Pramila R
[LinkedIn](https://www.linkedin.com/in/pramila-r) · [GitHub](https://github.com/Pramila6578) · [Tableau Public](https://public.tableau.com/app/profile/pramila.r7292/vizzes)
