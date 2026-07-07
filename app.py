import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    layout="wide"
)

st.title("📈 End-to-End Sales Forecasting & Demand Intelligence System")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        dayfirst=True
    )

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.month_name()

    return df


sales = load_data()

# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.radio(
    "Select Page",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)

# ====================================================
# PAGE 1
# ====================================================

if page == "Sales Overview":

    st.header("Sales Overview Dashboard")

    total_sales = sales["Sales"].sum()

    total_orders = sales["Order ID"].nunique()

    total_customers = sales["Customer ID"].nunique()

    avg_order = sales["Sales"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Sales", f"${total_sales:,.0f}")
    c2.metric("Orders", total_orders)
    c3.metric("Customers", total_customers)
    c4.metric("Average Order", f"${avg_order:.2f}")

    st.divider()

    # -------------------------
    # Filters
    # -------------------------

    region = st.sidebar.selectbox(
        "Region",
        ["All"] + sorted(sales["Region"].unique().tolist())
    )

    category = st.sidebar.selectbox(
        "Category",
        ["All"] + sorted(sales["Category"].unique().tolist())
    )

    filtered = sales.copy()

    if region != "All":
        filtered = filtered[
            filtered["Region"] == region
        ]

    if category != "All":
        filtered = filtered[
            filtered["Category"] == category
        ]

    # -------------------------
    # Sales by Year
    # -------------------------

    yearly = filtered.groupby("Year")["Sales"].sum()

    fig, ax = plt.subplots(figsize=(8,4))

    yearly.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Total Sales by Year")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    # -------------------------
    # Monthly Trend
    # -------------------------

    monthly = filtered.groupby(
        pd.Grouper(
            key="Order Date",
            freq="ME"
        )
    )["Sales"].sum()

    fig2, ax2 = plt.subplots(figsize=(12,4))

    ax2.plot(
        monthly.index,
        monthly.values,
        marker="o"
    )

    ax2.set_title("Monthly Sales Trend")

    st.pyplot(fig2)

    st.subheader("Sales Data")

    st.dataframe(filtered.head(20))

elif page == "Forecast Explorer":

    st.header("Forecast Explorer")

    forecast_type = st.selectbox(
        "Forecast Based On",
        ["Category", "Region"]
    )

    horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    if forecast_type == "Category":

        selected = st.selectbox(
            "Select Category",
            sorted(sales["Category"].unique())
        )

        data = sales[
            sales["Category"] == selected
        ]

    else:

        selected = st.selectbox(
            "Select Region",
            sorted(sales["Region"].unique())
        )

        data = sales[
            sales["Region"] == selected
        ]

    monthly = data.groupby(
        pd.Grouper(
            key="Order Date",
            freq="ME"
        )
    )["Sales"].sum().reset_index()

    monthly["Lag1"] = monthly["Sales"].shift(1)
    monthly["Lag2"] = monthly["Sales"].shift(2)
    monthly["Lag3"] = monthly["Sales"].shift(3)

    monthly["RollingMean3"] = (
        monthly["Sales"]
        .rolling(3)
        .mean()
    )

    monthly["Month"] = (
        monthly["Order Date"]
        .dt.month
    )

    monthly["Quarter"] = (
        monthly["Order Date"]
        .dt.quarter
    )

    monthly = monthly.dropna()

    from xgboost import XGBRegressor

    features = [
        "Lag1",
        "Lag2",
        "Lag3",
        "RollingMean3",
        "Month",
        "Quarter"
    ]

    X = monthly[features]
    y = monthly["Sales"]

    X_train = X[:-horizon]
    X_test = X[-horizon:]

    y_train = y[:-horizon]
    y_test = y[-horizon:]

    model = XGBRegressor(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error
    )

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    fig, ax = plt.subplots(
        figsize=(12,5)
    )

    ax.plot(
        y_test.index,
        y_test,
        marker="o",
        label="Actual"
    )

    ax.plot(
        y_test.index,
        pred,
        marker="o",
        label="Forecast"
    )

    ax.set_title(
        f"{selected} Forecast"
    )

    ax.legend()

    st.pyplot(fig)

    c1, c2 = st.columns(2)

    c1.metric(
        "MAE",
        f"{mae:,.2f}"
    )

    c2.metric(
        "RMSE",
        f"{rmse:,.2f}"
    )

    result = pd.DataFrame({

        "Month":

        monthly["Order Date"]
        .tail(horizon)
        .dt.strftime("%b-%Y"),

        "Actual":

        y_test.values,

        "Forecast":

        pred

    })

    st.subheader(
        "Forecast Table"
    )

    st.dataframe(result)

elif page == "Anomaly Report":

    st.header("Sales Anomaly Report")

    weekly = sales.groupby(
        pd.Grouper(
            key="Order Date",
            freq="W"
        )
    )["Sales"].sum().reset_index()

    iso = IsolationForest(
        contamination=0.03,
        random_state=42
    )

    weekly["Anomaly"] = iso.fit_predict(
        weekly[["Sales"]]
    )

    anomalies = weekly[
        weekly["Anomaly"] == -1
    ]

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        weekly["Order Date"],
        weekly["Sales"],
        label="Weekly Sales"
    )

    ax.scatter(
        anomalies["Order Date"],
        anomalies["Sales"],
        color="red",
        s=80,
        label="Anomaly"
    )

    ax.set_title("Weekly Sales Anomaly Detection")

    ax.legend()

    st.pyplot(fig)

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomalies[
            ["Order Date", "Sales"]
        ]
    )

    st.metric(
        "Total Anomalies",
        len(anomalies)
    )

elif page == "Product Demand Segments":

    st.header("Product Demand Segmentation")

    cluster_df = sales.groupby(
        "Sub-Category"
    ).agg(
        Total_Sales=("Sales", "sum"),
        Avg_Order_Value=("Sales", "mean")
    ).reset_index()

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        cluster_df[
            [
                "Total_Sales",
                "Avg_Order_Value"
            ]
        ]
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    cluster_df["Cluster"] = kmeans.fit_predict(
        scaled
    )

    pca = PCA(
        n_components=2
    )

    pca_data = pca.fit_transform(
        scaled
    )

    cluster_df["PC1"] = pca_data[:,0]
    cluster_df["PC2"] = pca_data[:,1]

    fig, ax = plt.subplots(figsize=(10,6))

    scatter = ax.scatter(
        cluster_df["PC1"],
        cluster_df["PC2"],
        c=cluster_df["Cluster"],
        cmap="viridis",
        s=120
    )

    for i, txt in enumerate(
        cluster_df["Sub-Category"]
    ):
        ax.annotate(
            txt,
            (
                cluster_df["PC1"].iloc[i],
                cluster_df["PC2"].iloc[i]
            ),
            fontsize=8
        )

    ax.set_title(
        "Product Demand Segments"
    )

    st.pyplot(fig)

    st.subheader("Cluster Table")

    st.dataframe(cluster_df)
