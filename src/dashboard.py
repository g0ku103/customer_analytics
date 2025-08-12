import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set page config
st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")

# Header with timestamp
st.title("Customer Revenue Forecasting & Segmentation Dashboard")
st.subheader(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")

# Tabs for sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["EDA", "RFM Analysis", "Clustering", "Revenue Forecasting", "LLM Insights"])

with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    # Load EDA plots
    st.image("visualizations/distributions.png", caption="Distributions of Quantity, UnitPrice, TotalPrice")
    st.image("visualizations/monthly_revenue.png", caption="Monthly Revenue Trend")
    st.image("visualizations/country_revenue.png", caption="Top 10 Countries by Revenue")
    # Load customer metrics summary
    df_eda = pd.read_csv("data/processed/cleaned_data.csv")
    st.dataframe(df_eda.head())
    st.write("Customer Metrics Summary from EDA:")
    customer_metrics = df_eda.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum',
        'InvoiceDate': 'max'
    }).rename(columns={'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary', 'InvoiceDate': 'LastPurchase'})
    st.write(customer_metrics.describe())

with tab2:
    st.header("RFM Analysis")
    # Load RFM data
    df_rfm = pd.read_csv("data/processed/rfm_data.csv")
    st.dataframe(df_rfm.head())
    st.image("visualizations/rfm_segments.png", caption="RFM Segment Distribution")
    st.write("RFM Segment Counts:")
    st.write(df_rfm['Segment'].value_counts())

with tab3:
    st.header("Clustering")
    # Load clustered data
    df_clustered = pd.read_csv("data/processed/clustered_rfm_data.csv")
    st.dataframe(df_clustered.head())
    st.image("visualizations/elbow_plot.png", caption="Elbow Method for Optimal Clusters")
    st.image("visualizations/customer_clusters.png", caption="Clusters by Recency vs. Monetary")
    st.write("Cluster Counts:")
    st.write(df_clustered['Cluster'].value_counts())

with tab4:
    st.header("Revenue Forecasting")
    # Load forecast data
    df_forecast = pd.read_csv("data/processed/forecast_data.csv")
    st.dataframe(df_forecast.head())
    st.image("visualizations/revenue_predictions.png", caption="Actual vs. Predicted Revenue")
    st.write("Model Performance (from Step 7):")
    st.write("Linear Regression - MSE: 68142695.62, R2: 0.33")
    st.write("Random Forest - MSE: 38344259.88, R2: 0.63")

with tab5:
    st.header("LLM-Powered Insights")
    # Load insights text
    insights_path = "insights/insights.txt"
    if os.path.exists(insights_path):
        with open(insights_path, "r") as f:
            insights = f.read()
        st.write(insights)
    else:
        st.write("Insights file not found. Run Step 8 to generate.")
    st.image("visualizations/insight_visualizations.png", caption="Insight Visualizations")

# Footer
st.markdown("---")
st.markdown("Built by Gokul S Babu | Project Portfolio | Contact: gokulsbabu03@gmail.com | LinkedIn: https://www.linkedin.com/in/gokul-s-babu-72b526250/")