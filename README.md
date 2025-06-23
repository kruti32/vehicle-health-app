# 🚗 Vehicle Health Prediction App

This is a Streamlit-based web application that predicts the **health of a vehicle's engine** using sensor data. It uses a trained machine learning model to classify engine condition as **Healthy (1)** or **Unhealthy (0)** based on real-time or uploaded input data.

---

## 📦 Features

- Upload engine sensor data in CSV format
- Get real-time predictions of engine condition
- Visualize results in tables and bar charts
- Download prediction output
- View accuracy and classification report (if true labels are available)

---

## 📁 Sample Data Format

The CSV should contain the following columns:
- Coolant temp
- lub oil temp
- Fuel pressure
- Lub oil pressure
- Engine rpm
- Engine Condition (optional – used to show accuracy)

---

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/vehicle-health-app.git
   cd vehicle-health-app
