import pandas as pd

df = pd.read_excel(
    r"C:\Users\Akshat\OneDrive\Desktop\shl\data\raw\Gen_AI Dataset (1).xlsx"
)


print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nSample rows:")
print(df.head())
