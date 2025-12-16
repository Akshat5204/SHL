import pandas as pd
import os

print("=== Preparing processed_data.csv ===")

input_path = "data/raw/Gen_AI Dataset (1).xlsx"
output_path = "data/processed/processed_data.csv"

# Ensure folder exists
os.makedirs("data/processed", exist_ok=True)

# Load Excel
df = pd.read_excel(input_path)

print("Original columns:", list(df.columns))
print("Rows:", len(df))

# 🔑 CREATE search_text COLUMN
df["search_text"] = df.astype(str).agg(" ".join, axis=1)

print("Columns AFTER adding search_text:", list(df.columns))

# Save
df.to_csv(output_path, index=False)

print("✅ processed_data.csv created successfully")
