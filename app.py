import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Page config
st.set_page_config(page_title="Vehicle Health Predictor", layout="centered")

# Sidebar layout
with st.sidebar:
    st.title("📁 Upload Data")
    uploaded_file = st.file_uploader("Upload engine_data.csv", type=["csv"])
    st.info("This model predicts engine health using sensor readings. 1 = Healthy, 0 = Unhealthy")

# Title
st.title("🚗 Vehicle Health Prediction App")

# Load the model
try:
    model = joblib.load('vehicle_health_model.pkl')
except FileNotFoundError:
    st.error("❌ Model file not found. Please place 'vehicle_health_model.pkl' in the app folder.")
    st.stop()

# Function to highlight predictions
@st.cache_data
def highlight_condition(val):
    if val == 1:
        return 'background-color: #d4edda'  # green
    elif val == 0:
        return 'background-color: #f8d7da'  # red
    return ''

# When a file is uploaded
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data (Top 5 rows)")
    st.write(df.head())

    if st.button("🚀 Predict"):
        try:
            has_actual = 'Engine Condition' in df.columns

            if has_actual:
                X = df.drop(columns=['Engine Condition'])
                y_true = df['Engine Condition']
            else:
                X = df

            # Feature engineering (must match training)
            X['Temp_Diff'] = X['Coolant temp'] - X['lub oil temp']
            X['Pressure_Ratio'] = X['Fuel pressure'] / (X['Lub oil pressure'] + 1)
            X['Load_Temp_Ratio'] = X['Engine rpm'] / (X['lub oil temp'] + 1)

            # Predict
            predictions = model.predict(X)
            df['Predicted Condition'] = predictions

            # Display styled results
            st.subheader("🔍 Prediction Results")
            styled_df = df.style.applymap(highlight_condition, subset=['Predicted Condition'])
            st.dataframe(styled_df)

            # Show distribution chart
            st.subheader("📊 Prediction Distribution")
            st.bar_chart(df['Predicted Condition'].value_counts())

            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results", csv, "predictions.csv", "text/csv")

            # Show accuracy if true labels exist
            if has_actual:
                acc = accuracy_score(y_true, predictions)
                report_dict = classification_report(y_true, predictions, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.subheader(f"✅ Accuracy: {acc * 100:.2f}%")
                st.subheader("📊 Classification Report (Detailed Table)")
                st.dataframe(report_df.style.format("{:.2f}"))

        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")
