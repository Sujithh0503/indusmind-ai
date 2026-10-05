import os
import pandas as pd
from ucimlrepo import fetch_ucirepo 

print("Fetching AI4I 2020 Predictive Maintenance Dataset...")
ai4i = fetch_ucirepo(id=601) 

X = ai4i.data.features 
y = ai4i.data.targets 

df = pd.concat([X, y], axis=1)

os.makedirs('data', exist_ok=True)
df.to_csv('data/ai4i2020.csv', index=False)

print("Dataset successfully saved to data/ai4i2020.csv!")
print(f"Shape: {df.shape}")