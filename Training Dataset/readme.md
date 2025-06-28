# 📂 Training Dataset – TrafficTelligence

This folder contains the core dataset used to train and evaluate machine learning models for the **TrafficTelligence** project.

---

## 📄 File Description

| File Name              | Description                                                      |
|------------------------|------------------------------------------------------------------|
| `traffic volume.csv`   | Main dataset containing timestamped traffic volume data along with weather, holiday, and temporal features |

---

## 📊 Dataset Overview

The `traffic volume.csv` includes the following key features:

- **date_time** – Timestamp of data recording
- **traffic_volume** – Target variable (number of cars per hour)
- **temperature, rain_1h, snow_1h, clouds_all** – Weather indicators
- **holiday** – Whether the date was a US holiday
- **weather_main, weather_description** – Textual weather context
- **hour, day_of_week** – Extracted features for time-based modeling

This dataset is ideal for regression tasks involving time series and weather-aware traffic prediction.

---

## 🔍 Usage

This dataset is used within:

📁 [`Project Files/Model Training`](../Project%20Files/Model%20Training)  
- Preprocessing
- Feature engineering
- Training Gradient Boosting and LSTM models

---

## 📝 Source

This dataset was sourced and adapted from a public **Kaggle** dataset titled:  
> *Metro Interstate Traffic Volume Dataset*

All data rights belong to the original publisher. It is used here for **educational and research purposes** only.

---

## 📄 License

This dataset is part of the **TrafficTelligence** project under the  
**SmartInternz AI/ML Virtual Internship 2025** and shared solely for academic use.

