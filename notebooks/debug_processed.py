import pandas as pd

df = pd.read_csv("data/processed/processed_data.csv")
print("COLUMNS:", df.columns)
print(df.head())
