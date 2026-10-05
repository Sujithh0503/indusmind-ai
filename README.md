# ⚡ IndusMind AI: Intelligent Predictive Maintenance Dashboard

> **Physics-Aware Machine Diagnostics & Real-Time Fault Explainability for Industrial CNC Spindles**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-20B2AA?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

---

## 📌 Problem Statement
In high-precision manufacturing, unplanned equipment breakdowns cause millions in lost throughput and damaged workpieces. Traditional maintenance relies on rigid scheduled intervals or primitive single-sensor thresholds. Standard machine learning models offer risk scores, but their "black-box" nature prevents plant operators from trusting AI-generated alerts without operational context.

**IndusMind AI** resolves this trust gap by combining real-time failure prediction with physics-based stress modeling and **SHAP root-cause explanations**, empowering technicians with actionable, transparent maintenance steps.

---

## ✨ Key Features

- **⚡ Sub-10ms Inference Engine:** Real-time scoring of machine health indices ($0-100\%$) and failure probabilities.
- **🧠 Dual-Model ML Architecture:**
  - **Supervised Failure Classifier (XGBoost):** Trained with `scale_pos_weight` to address extreme class imbalance in industrial failure datasets.
  - **Unsupervised Anomaly Detector (Isolation Forest):** Baselines normal telemetry dynamics to catch unmodeled, novel system behaviors.
- **🔬 Physics-Aware Feature Engineering:** Models domain-specific mechanical behavior:
  - *Thermal Difference ($\Delta T$):* Process Temperature vs. Air Temperature.
  - *Mechanical Power Output ($P_{\text{watts}}$):* Torque $\times$ Rotational Speed.
  - *Overstrain Factor:* Cumulative Tool Wear $\times$ Torque load.
- **🔍 Explainable AI (TreeSHAP):** Local feature attribution waterfall plots break down exactly *why* a machine is at risk.
- **🛑 Prescriptive Action Directives:** Converts risk probability thresholds directly into field instructions (e.g., immediate shutdown vs. scheduled 4-hour inspection).

---

## 🛠️ Tech Stack

- **Frontend & UI:** Streamlit, Plotly, Custom CSS
- **Machine Learning:** Scikit-Learn, XGBoost, SHAP
- **Data & Computation:** Pandas, NumPy, Matplotlib, Joblib
- **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset

---

## 📂 Repository Structure

```text
indusmind-ai/
├── models/                     # Saved model artifacts
│   ├── xgb_failure_model.pkl
│   ├── iso_forest.pkl
│   └── feature_cols.pkl
├── app.py                      # Main Streamlit web application
├── train.py                    # Model training pipeline & feature engineering
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
