import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

print("--- DAY 2: TRAINING PIPELINE STARTED ---")

# 1. Load Data
df = pd.read_csv('data/ai4i2020.csv')

# Clean column names (strip spaces, remove brackets for XGBoost compatibility)
df.columns = [
    c.strip()
    .replace(' [K]', '')
    .replace(' [rpm]', '')
    .replace(' [Nm]', '')
    .replace(' [min]', '')
    for c in df.columns
]

# Rename columns to standard clean names
rename_dict = {
    'Air temperature': 'Air_temperature',
    'Process temperature': 'Process_temperature',
    'Rotational speed': 'Rotational_speed',
    'Torque': 'Torque',
    'Tool wear': 'Tool_wear',
    'Machine failure': 'Machine_failure'
}
df.rename(columns=rename_dict, inplace=True)

# 2. Domain Feature Engineering
print("Engineering industrial domain features...")

# Temperature Difference (Process Temp - Air Temp)
df['Temp_Diff'] = df['Process_temperature'] - df['Air_temperature']

# Mechanical Power Output (W) = Torque (Nm) * Speed (rad/s)
df['Power_W'] = df['Torque'] * (df['Rotational_speed'] * (2 * np.pi / 60))

# Overstrain Factor = Tool Wear (min) * Torque (Nm)
df['Overstrain_Factor'] = df['Tool_wear'] * df['Torque']

# 3. Define Feature Sets (Clean names with no brackets/special chars)
feature_cols = [
    'Air_temperature', 'Process_temperature', 
    'Rotational_speed', 'Torque', 'Tool_wear',
    'Temp_Diff', 'Power_W', 'Overstrain_Factor'
]

X = df[feature_cols]
y_binary = df['Machine_failure']

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 4. Train Unsupervised Anomaly Detector (Isolation Forest)
print("Training Isolation Forest on normal operating conditions...")
X_train_normal = X_train[y_train == 0]
iso_forest = IsolationForest(contamination=0.04, random_state=42)
iso_forest.fit(X_train_normal)

# 5. Train Supervised XGBoost Failure Classifier
print("Training Imbalanced XGBoost Classifier...")

# Handle class imbalance automatically using scale_pos_weight
ratio = (len(y_train) - sum(y_train)) / sum(y_train)

xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=ratio,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# 6. Evaluate Model
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

print("\n--- MODEL EVALUATION METRICS ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save Trained Artifacts to models/
os.makedirs('models', exist_ok=True)
joblib.dump(xgb_model, 'models/xgb_failure_model.pkl')
joblib.dump(iso_forest, 'models/iso_forest.pkl')
joblib.dump(feature_cols, 'models/feature_cols.pkl')

print("\nModel artifacts successfully saved to 'models/' directory!")