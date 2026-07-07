# End-to-End Sales Forecasting & Demand Intelligence System

## Project Overview

This project is an end-to-end Sales Forecasting and Demand Intelligence System developed using Python, Machine Learning, and Streamlit.

The application analyzes historical sales data, forecasts future sales, detects anomalies, and segments products based on demand patterns. An interactive dashboard enables users to explore insights through visualizations and forecasting tools.

---

## Features

### Sales Overview Dashboard
- Total Sales KPI
- Total Orders
- Total Customers
- Average Order Value
- Year-wise Sales Analysis
- Monthly Sales Trend
- Region Filter
- Category Filter

### Forecast Explorer
- Sales Forecasting using XGBoost
- Category-wise Forecast
- Actual vs Forecast Comparison
- MAE and RMSE Evaluation
- Forecast Results Table

### Sales Anomaly Detection
- Weekly Sales Analysis
- Automatic Anomaly Detection
- Highlighted Outliers
- Anomaly Summary Table

### Product Demand Segmentation
- Product Clustering using K-Means
- PCA Visualization
- Cluster Assignment
- Product Demand Analysis

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Prophet
- Statsmodels
- Streamlit

---

## Machine Learning Models

- SARIMA
- Prophet
- XGBoost Regressor
- K-Means Clustering
- PCA
- Z-Score Based Anomaly Detection

---

## Project Structure

```
SalesForecasting_Sathvika/
│
├── analysis.ipynb
├── app.py
├── train.csv
├── vgsales.csv
├── requirements.txt
├── README.md
└── Charts/
```

---

## How to Run

### Install the required packages

```bash
pip install -r requirements.txt
```

### Launch the dashboard

```bash
streamlit run app.py
```

---

## Dashboard Modules

- Sales Overview
- Forecast Explorer
- Sales Anomaly Report
- Product Demand Segmentation

---

## Results

- Built an interactive Streamlit dashboard for sales analysis.
- Forecasted future sales using machine learning models.
- Detected unusual sales behavior using anomaly detection.
- Segmented products using K-Means clustering.
- Generated interactive visualizations for business insights.

---
