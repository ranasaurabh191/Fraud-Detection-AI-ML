# scripts/load_data.py
import pandas as pd

# Load dataset
df = pd.read_csv('./data/creditcard.csv')

# Preview data
print(df.head())

# Save preprocessed data
df.to_csv('./data/processed_transactions.csv', index=False)
