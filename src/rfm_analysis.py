import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set plot style
sns.set_style("whitegrid")

def load_customer_metrics(file_path):
    """Load or compute customer metrics from CSV."""
    try:
        df = pd.read_csv(file_path)
        print("Loaded Data Shape:", df.shape)
        # Compute customer metrics if not pre-aggregated
        if not all(col in df.columns for col in ['LastPurchase', 'Frequency', 'Monetary']):
            print("Computing customer metrics from transaction data...")
            customer_metrics = df.groupby('CustomerID').agg({
                'InvoiceDate': 'max',  # Last purchase date
                'InvoiceNo': 'nunique',  # Frequency
                'TotalPrice': 'sum'     # Monetary
            }).rename(columns={
                'InvoiceDate': 'LastPurchase',
                'InvoiceNo': 'Frequency',
                'TotalPrice': 'Monetary'
            }).reset_index()
            print("Computed Customer Metrics Shape:", customer_metrics.shape)
            return customer_metrics
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_rfm(df):
    """Calculate RFM metrics and scores for customer segmentation."""
    # Current date (set to dataset end for historical context)
    current_date = datetime(2011, 12, 9)  # Reverted from datetime.now()
    df['LastPurchase'] = pd.to_datetime(df['LastPurchase'], errors='coerce')
    df = df.dropna(subset=['LastPurchase'])
    rfm_table = df.copy()
    rfm_table['Recency'] = (current_date - rfm_table['LastPurchase']).dt.days
    rfm_table['Frequency'] = rfm_table['Frequency']
    rfm_table['Monetary'] = rfm_table['Monetary']

    # Assign RFM scores (1-5) based on quantiles
    rfm_table['R_Score'] = pd.qcut(rfm_table['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    rfm_table['F_Score'] = pd.qcut(rfm_table['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    rfm_table['M_Score'] = pd.qcut(rfm_table['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    rfm_table['RFM_Score'] = rfm_table['R_Score'].astype(str) + rfm_table['F_Score'].astype(str) + rfm_table['M_Score'].astype(str)

    # Define segments
    def segment_rfm(rfm):
        if rfm == '555': return 'High-Value'
        elif rfm in ['111', '112', '121', '211', '122']: return 'At-Risk'
        elif rfm in ['543', '544', '553', '554']: return 'Loyal'
        else: return 'Others'
    rfm_table['Segment'] = rfm_table['RFM_Score'].apply(segment_rfm)

    return rfm_table    

def visualize_segments(rfm_table, save_path):
    """Visualize segment distribution."""
    plt.figure(figsize=(10, 5))
    sns.countplot(data=rfm_table, x='Segment')
    plt.title('Customer Segment Distribution')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'rfm_segments.png'))
    plt.close()
    print("Segment distribution plot saved to visualizations/rfm_segments.png")

def save_rfm_data(rfm_table, output_path):
    """Save RFM data to CSV."""
    try:
        rfm_table.to_csv(output_path, index=False)
        print(f"RFM data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    df = load_customer_metrics("data/processed/cleaned_data.csv")
    if df is not None:
        os.makedirs("visualizations", exist_ok=True)
        rfm_table = calculate_rfm(df)
        visualize_segments(rfm_table, "visualizations")
        save_rfm_data(rfm_table, "data/processed/rfm_data.csv")