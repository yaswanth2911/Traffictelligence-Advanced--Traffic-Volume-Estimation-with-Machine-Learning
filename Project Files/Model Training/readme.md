
# 🤖 Model Training – TrafficTelligence

This directory contains all the core files related to training and saving the machine learning model used in the **TrafficTelligence** project for advanced traffic volume estimation.

---

## 📁 Files Included

| File Name           | Description                                                  |
|---------------------|--------------------------------------------------------------|
| `project_model.ipynb` | Jupyter notebook used for training and evaluating the model |
| `scaler.joblib`     | StandardScaler object used for input normalization           |
| `encoders.joblib`   | Encoders used to preprocess categorical data (if any)        |

---

## ⚙️ Model Overview

- Uses **Gradient Boosting Regressor** and other ML models for traffic prediction.
- Trained on cleaned and feature-engineered datasets (e.g., traffic volume, time, weather).
- Outputs continuous traffic volume values based on historical and contextual data.

---

## 🔍 Features & Preprocessing

- **Feature Engineering**: Extracted date-time features like hour, day, month.
- **Scaling**: Numerical features scaled using `StandardScaler`.
- **Categorical Encoding**: Label encoding used for object/string features.

---

## 📦 Dependencies

- `scikit-learn`
- `joblib`
- `pandas`
- `numpy`
- `matplotlib` (optional for plotting)
- `seaborn` (optional for EDA)

You can install dependencies via:

```bash
pip install -r requirements.txt
