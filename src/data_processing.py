import pandas as pd

def load_data(file_path):
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        print("Dataset Shape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nMissing Values:\n", df.isnull().sum())
        print("\nData Types:\n", df.dtypes)
        df.head().to_csv("data/processed/sample_data.csv", index=False)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    # Create a copy to avoid modifying the original
    df = df.copy()
    print("Initial Shape:", df.shape)

    # Drop rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    print("Shape after dropping missing CustomerID:", df.shape)

    # Convert CustomerID to string and remove decimal
    df.loc[:, 'CustomerID'] = df['CustomerID'].astype(float).astype(int).astype(str)
    print("CustomerID converted to string")

    # Handle missing Description (fill with 'Unknown')
    df.loc[:, 'Description'] = df['Description'].fillna('Unknown')
    print("Missing Descriptions filled with 'Unknown'")

    # Remove invalid data (negative Quantity or UnitPrice)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    print("Shape after removing negative Quantity/UnitPrice:", df.shape)

    # Create TotalPrice column
    df.loc[:, 'TotalPrice'] = df['Quantity'] * df['UnitPrice']
    print("Added TotalPrice column")

    # Remove duplicates
    df = df.drop_duplicates()
    print("Shape after removing duplicates:", df.shape)

    # Verify data types
    print("\nFinal Data Types:\n", df.dtypes)
    print("\nFinal Missing Values:\n", df.isnull().sum())

    return df

def save_cleaned_data(df, output_path):
    """
    Save the cleaned dataset to a CSV file.
    Args:
        df (pd.DataFrame): Cleaned dataset.
        output_path (str): Path to save the CSV.
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    # Load the dataset
    file_path = "data/raw/Online Retail.xlsx"
    df = load_data(file_path)
    if df is not None:
        print("\nFirst 5 rows:\n", df.head())
        # Clean the dataset
        df_cleaned = clean_data(df)
        # Save cleaned data
        save_cleaned_data(df_cleaned, "data/processed/cleaned_data.csv")
        # Display sample
        print("\nFirst 5 rows of cleaned data:\n", df_cleaned.head())