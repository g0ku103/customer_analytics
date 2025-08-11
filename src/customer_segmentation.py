import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Set plot style
sns.set_style("whitegrid")

def load_rfm_data(file_path):
    """Load RFM data from CSV."""
    try:
        df = pd.read_csv(file_path)
        print("Loaded RFM Data Shape:", df.shape)
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def preprocess_data(df):
    """Preprocess RFM data for clustering."""
    # Select relevant features
    features = df[['Recency', 'Frequency', 'Monetary']]
    # Handle any missing values
    features = features.dropna()
    if features.empty:
        raise ValueError("No data available after dropping NA values")
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print("Preprocessed Features Shape:", scaled_features.shape)
    return scaled_features, scaler

def determine_optimal_clusters(scaled_features):
    """Determine optimal number of clusters using the elbow method."""
    inertias = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, marker='o')
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.savefig(os.path.join("visualizations", "elbow_plot.png"))
    plt.close()
    print("Elbow plot saved to visualizations/elbow_plot.png")
    # Suggest optimal k (e.g., where elbow bends, typically 3-5)
    optimal_k = 4  # Adjust based on elbow plot
    return optimal_k

def perform_clustering(scaled_features, optimal_k):
    """Perform K-Means clustering."""
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    return cluster_labels

def visualize_clusters(df, cluster_labels):
    """Visualize clusters using a scatter plot."""
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_clustered, x='Recency', y='Monetary', hue='Cluster', palette='deep', s=100)
    plt.title('Customer Segments (Recency vs. Monetary)')
    plt.xlabel('Recency (Days)')
    plt.ylabel('Monetary Value ($)')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", "customer_clusters.png"))
    plt.close()
    print("Cluster visualization saved to visualizations/customer_clusters.png")

def save_clustered_data(df, cluster_labels, output_path):
    """Save clustered data to CSV."""
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    try:
        df_clustered.to_csv(output_path, index=False)
        print(f"Clustered data saved to {output_path}")
    except Exception as e:
        print(f"Error saving clustered data: {e}")

if __name__ == "__main__":
    # Load data
    rfm_data = load_rfm_data("data/processed/rfm_data.csv")
    if rfm_data is not None:
        # Preprocess
        scaled_features, scaler = preprocess_data(rfm_data)
        # Determine optimal clusters
        optimal_k = determine_optimal_clusters(scaled_features)
        print(f"Optimal number of clusters: {optimal_k}")
        # Perform clustering
        cluster_labels = perform_clustering(scaled_features, optimal_k)
        # Visualize
        visualize_clusters(rfm_data, cluster_labels)
        # Save results
        save_clustered_data(rfm_data, cluster_labels, "data/processed/clustered_rfm_data.csv")