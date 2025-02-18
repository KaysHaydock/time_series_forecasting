## Project Overview

This project investigates **time series forecasting** for **paediatric emergency department attendances**. The dataset contains **daily ED attendance counts** from **01/04/2014 to 19/02/2017**. The objective is to develop and compare forecasting models to predict attendances for the next **28 days** and evaluate their performance.

### **Goals**
- **Understand seasonal trends** in paediatric ED attendances (weekly & monthly patterns).
- **Develop and evaluate forecasting models** (Naive, ARIMA, and Prophet).
- **Compare models using performance metrics** (MAE and Winkler scores for prediction intervals).
- **Make recommendations** for staff planning based on forecast accuracy.

---

## Key Files

### **Jupyter Notebooks**
- `00_introduction.ipynb` – Project overview.
- `01_data_exploration.ipynb` – Exploratory data analysis (EDA), visualising trends, seasonality and outliers.
- `02_naive_model.ipynb` – Implements **Naive 1 and Seasonal Naive models** as benchmarks.
- `03_arima_model.ipynb` – Develops and evaluates an **ARIMA models** for forecasting.
- `04_prophet_model.ipynb` – Implements **Facebook Prophet models** for capturing trends and seasonality.
- `05_final_report.ipynb` – Main report, insights and recommendations.
- `06_references.ipynb` – References.

### **Folder Structure**
```bash
project/
├── README.md
├── binder/
│   └── environment.yml
├── 00_introduction.ipynb
├── 01_data_exploration.ipynb
├── 02_naive_model.ipynb
├── 03_arima_model.ipynb
├── 04_prophet.ipynb
├── 05_final_report.ipynb
├── 06_references.ipynb
├── images/
│   └── auto_arima_graph.png
│   ├── naive_graph.png
│   ├── box_plots.png
│   ├── prophet_predictions.png
│   └── prophet_graph.png
├── input/
│   └── model_runs.keras
├── paediatrics_train.csv
└──  ts_utils.py
```


### **Data Files**
- `paediatrics_train.csv` – The dataset containing **daily paediatric ED attendances**.

### **Scripts**
- `ts_utils.py` – Utility functions for time series preprocessing, modeling and evaluation.

---

## Methodology

### **1. Data Exploration**
- Visualising **monthly and weekly seasonality**.
- Identifying **trends and outliers**.
- Applying **stationarity tests (ADF, KPSS)**.

### **2. Forecasting Models**
| Model | Description |
|-------|------------|
| **Naive** | Baseline models for benchmarking, investigated Naive1 & SeasonalNaive |
| **ARIMA** | Classical time series model (Auto-Regressive Integrated Moving Average) |
| **FB Prophet** | Forecasting model designed for handling seasonality and holidays |

### **3. Model Evaluation**
- **MAE (Mean Absolute Error)** – Measures average forecast error.
- **Winkler Score (80% & 90%)** – Evaluates prediction intervals.

---

## Installation & Setup

To run the notebooks, set up the required environment:

### **1. Conda Environment**
```bash
conda env create -f binder/environment.yml
```
```bash
conda activate hds_forecast
```

Author: Kayleigh Haydock
