import pandas as pd

# --- Step 1: Load your data ---
# Example: If your file is a CSV
df = pd.read_csv("data/financial_fraud_detection_dataset.csv")

print("Original shape:", df.shape)

# --- Step 2: Randomly sample 3,000,00 rows (20% of 5,000,000) ---
df_reduced = df.sample(n=5_000_00, random_state=42)

print("Reduced shape:", df_reduced.shape)

# --- Step 3: Save the reduced dataset ---

df_reduced.to_csv("fraud_data.csv", index=False)


print("Reduced dataset saved as reduced_data.csv")
