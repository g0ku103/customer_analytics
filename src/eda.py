import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set_style("whitegrid")

def load_cleaned_data(file_path):
    """Load cleaned dataset from CSV."""
    try:
        df = pd.read_csv(file_path)
        print("Cleaned Data Shape:", df.shape)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_distributions(df, save_path):
    """Analyze and plot distributions of Quantity, UnitPrice, TotalPrice."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Quantity
    sns.histplot(df['Quantity'], bins=50, ax=axes[0])
    axes[0].set_title('Quantity Distribution')
    axes[0].set_xlabel('Quantity')
    # UnitPrice
    sns.histplot(df['UnitPrice'], bins=50, ax=axes[1])
    axes[1].set_title('UnitPrice Distribution')
    axes[1].set_xlabel('UnitPrice')
    # TotalPrice
    sns.histplot(df['TotalPrice'], bins=50, ax=axes[2])
    axes[2].set_title('TotalPrice Distribution')
    axes[2].set_xlabel('TotalPrice')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'distributions.png'))
    plt.close()
    print("Distribution plot saved to visualizations/distributions.png")

def analyze_time_series(df, save_path):
    """Analyze and plot monthly revenue trends."""
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
    monthly_revenue = df.groupby('YearMonth')['TotalPrice'].sum().reset_index()
    monthly_revenue['YearMonth'] = monthly_revenue['YearMonth'].astype(str)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly_revenue, x='YearMonth', y='TotalPrice')
    plt.title('Monthly Revenue Trend')
    plt.xlabel('Year-Month')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'monthly_revenue.png'))
    plt.close()
    print("Monthly revenue plot saved to visualizations/monthly_revenue.png")

def analyze_country(df, save_path):
    """Analyze and plot revenue by country."""
    country_revenue = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=country_revenue.values, y=country_revenue.index)
    plt.title('Top 10 Countries by Revenue')
    plt.xlabel('Total Revenue')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'country_revenue.png'))
    plt.close()
    print("Country revenue plot saved to visualizations/country_revenue.png")

def analyze_customer_metrics(df):
    """Compute customer-level metrics for RFM analysis."""
    customer_metrics = df.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',  # Purchase frequency
        'TotalPrice': 'sum',     # Total spend
        'InvoiceDate': 'max'     # Most recent purchase
    }).rename(columns={
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary',
        'InvoiceDate': 'LastPurchase'
    })
    print("\nCustomer Metrics Summary:\n", customer_metrics.describe())
    return customer_metrics

if __name__ == "__main__":
    # Load data
    df = load_cleaned_data("data/processed/cleaned_data.csv")
    if df is not None:
        # Create visualizations folder if it doesn't exist
        os.makedirs("visualizations", exist_ok=True)
        # Run analyses
        analyze_distributions(df, "visualizations")
        analyze_time_series(df, "visualizations")
        analyze_country(df, "visualizations")
        customer_metrics = analyze_customer_metrics(df)