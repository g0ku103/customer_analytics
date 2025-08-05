import pandas as pd

def load_data(file_path):

    try:
        if file_path.endswith('.xlsx'):
            df=pd.read_excel(file_path)
        else:
            df.read_csv(file_path)
        print("Dataset Shape ",df.shape)
        print("\nColumn: ",df.columns.tolist())
        print("\n Missinf Value :",df.isnull().sum())
        print("Dataset type : ",df.dtypes)
        df.head().to_csv("data/processed/sample_data.csv",index=False)
        return df
    except Exception as e:
        print(f'Error Loading data : {e}')
        return None
if __name__ == "__main__":
    file_path="data/raw/Online Retail.xlsx"
    df=load_data(file_path)
    if df is not None:
        print("\n First 5 Row\n",df.head())