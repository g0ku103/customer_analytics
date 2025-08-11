import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
import os
import joblib

sns.set_style("white")


def load_and_preprocess_data(rfm_file , clustered_file):
    try:
        rfm_df = pd.read_csv(rfm_file)
        clustered_df = pd.read_csv(clustered_file)

        df=pd.merge(rfm_df,clustered_df[['CustomerID','Cluster']])

        df['Cluster'] = df['Cluster'].fillna(-1).astype(int)

        features = df[['Recency','Frequency','Cluster']]
        target = df['Monetary']

        features = features.dropna()
        target = target[features.index]

        print('Processed Data Shape : ',features.shape)
        return features , target
    except Exception as e:
        print(f'Error loading or preprocessing data : {e}')
        return None , None
    

def train_and_evaluate_model(features , target):

    X_train ,X_test ,y_train ,y_test = train_test_split(features,target,test_size=0.2,random_state=42)
    print('Training set shape : ',X_test.shape,'Test set shape : ',X_train.shape)

    #linear regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train,y_train)
    lr_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test,lr_pred)
    lr_mse = mean_squared_error(y_test,lr_pred)

    print(f'Linear Regression - MSE : {lr_mse}, R2 : {lr_r2:.2f}')

    #RandomForest Regressor
    rf_model = RandomForestRegressor(n_estimators=100 , random_state=42)
    rf_model.fit(X_train,y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test,rf_pred)
    rf_r2 = r2_score(y_test,rf_pred)
    print(f'Random Forest - MSE : {rf_mse} , R2 : {rf_r2:.2f}')

    return lr_model,rf_model ,X_test,y_test,lr_pred,rf_pred

#visualization
def visualize_predictions(X_test,y_test,lr_pred,rf_pred):

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(X_test['Recency'], y_test, color='blue', label='Actual')
    plt.scatter(X_test['Recency'], lr_pred, color='red', label='Predicted (Linear)')
    plt.title('Linear Regression: Actual vs. Predicted')
    plt.xlabel('Recency (Days)')
    plt.ylabel('Monetary Value ($)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(X_test['Recency'], y_test, color='blue', label='Actual')
    plt.scatter(X_test['Recency'], rf_pred, color='green', label='Predicted (Random Forest)')
    plt.title('Random Forest: Actual vs. Predicted')
    plt.xlabel('Recency (Days)')
    plt.ylabel('Monetary Value ($)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", "revenue_predictions.png"))
    plt.close()
    print("Prediction visualization saved to visualizations/revenue_predictions.png")


def save_model_and_results(lr_model, rf_model, output_dir="models"):
    """Save trained models and results."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(lr_model, os.path.join(output_dir, "linear_regression_model.joblib"))
    joblib.dump(rf_model, os.path.join(output_dir, "random_forest_model.joblib"))
    print(f"Models saved to {output_dir}")

if __name__ == "__main__":
    # Load and preprocess data
    features, target = load_and_preprocess_data("data/processed/rfm_data.csv", "data/processed/clustered_rfm_data.csv")
    if features is not None and target is not None:
        os.makedirs("visualizations", exist_ok=True)
        # Train and evaluate models
        lr_model, rf_model, X_test, y_test, lr_pred, rf_pred = train_and_evaluate_model(features, target)
        # Save predictions
        test_df = pd.DataFrame({
            'CustomerID': pd.read_csv("data/processed/clustered_rfm_data.csv")['CustomerID'].iloc[X_test.index],
            'Actual_Monetary': y_test,
            'Predicted_Linear': lr_pred,
            'Predicted_RandomForest': rf_pred
        })
        test_df.to_csv("data/processed/forecast_data.csv", index=False)
        print("Forecast data saved to data/processed/forecast_data.csv")
        # Visualize predictions
        visualize_predictions(X_test, y_test, lr_pred, rf_pred)
        # Save models
        save_model_and_results(lr_model, rf_model)


