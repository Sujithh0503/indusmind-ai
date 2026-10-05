import pandas as pd

df = pd.read_csv('data/ai4i2020.csv')

print("--- DATASET OVERVIEW ---")
print(df.info())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- MACHINE FAILURE CLASS DISTRIBUTION ---")
if 'Machine failure' in df.columns:
    print(df['Machine failure'].value_counts(normalize=True) * 100)

print("\n--- SPECIFIC FAILURE MODES DISTRIBUTION ---")
failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
for col in failure_cols:
    if col in df.columns:
        print(f"{col}: {df[col].sum()} positive cases")