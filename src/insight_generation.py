import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import r2_score
import joblib

sns.set_style("whitegrid")

def load_and_combine_data(rfm_file , clustered_file , forecast_file):
    try:
        rfm_df = pd.read_csv(rfm_file)
        clustered_df = pd.read_csv(clustered_file)
        forecast_df = pd.read_csv(forecast_file)
        
        # Merge dataframes
        df = pd.merge(rfm_df, clustered_df[['CustomerID', 'Cluster']], on='CustomerID', how='left')
        df = pd.merge(df, forecast_df[['CustomerID', 'Predicted_RandomForest']], on='CustomerID', how='left')
        # Rename for consistency
        df['Predicted_Monetary'] = df['Predicted_RandomForest'].fillna(df['Monetary'])
        print("Combined Data Shape:", df.shape)
        return df
    except Exception as e:
        print(f"Error loading or combining data: {e}")
        return None
    
def generate_insights(df):
    insights = []

    # Segment analysis
    segment_counts = df['Segment'].value_counts()
    high_value_count = segment_counts.get('High-Value', 0)
    at_risk_count = segment_counts.get('At-Risk', 0)
    insights.append(f"Total customers: {len(df)}. High-Value customers: {high_value_count} ({high_value_count/len(df)*100:.1f}%), At-Risk: {at_risk_count} ({at_risk_count/len(df)*100:.1f}%).")

    # Cluster analysis
    cluster_counts = df['Cluster'].value_counts()
    for cluster, count in cluster_counts.items():
        avg_monetary = df[df['Cluster'] == cluster]['Monetary'].mean()
        insights.append(f"Cluster {cluster} has {count} customers with average revenue of ${avg_monetary:.2f}.")

    # Revenue forecasting insight
    avg_actual = df['Monetary'].mean()
    avg_predicted = df['Predicted_Monetary'].mean()
    insights.append(f"Average actual revenue: ${avg_actual:.2f}, Average predicted revenue: ${avg_predicted:.2f}. "
                    f"Difference suggests {((avg_predicted - avg_actual) / avg_actual * 100):.1f}% change.")

    # Recommendations
    if high_value_count / len(df) < 0.05:
        insights.append("Recommendation: Focus marketing on retaining and upselling High-Value customers.")
    if at_risk_count / len(df) > 0.2:
        insights.append("Recommendation: Implement re-engagement campaigns for At-Risk customers.")
    if avg_predicted > avg_actual:
        insights.append("Recommendation: Invest in strategies to sustain predicted revenue growth.")

    return insights


def visualize_insights(df, insights):
    """Visualize key insights."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='Segment')
    plt.title('Customer Segment Distribution')
    plt.xlabel('Segment')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x='Cluster', y='Monetary', estimator=np.mean, errorbar=None)
    plt.title('Average Revenue by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Monetary Value ($)')
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", "insight_visualizations.png"))
    plt.close()
    print("Insight visualizations saved to visualizations/insight_visualizations.png")


def save_insights(insights, output_path="insights/insights.txt"):
    """Save generated insights to a file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("\n".join(insights))
    print(f"Insights saved to {output_path}")

if __name__ == "__main__":
    # Load and combine data (simulating forecast data)
    df = load_and_combine_data("data/processed/rfm_data.csv", "data/processed/clustered_rfm_data.csv", "data/processed/forecast_data.csv")
    if df is not None:
        os.makedirs("visualizations", exist_ok=True)
        os.makedirs("insights", exist_ok=True)
        # Generate insights
        insights = generate_insights(df)
        for insight in insights:
            print(insight)
        # Visualize insights
        visualize_insights(df, insights)
        # Save insights
        save_insights(insights)