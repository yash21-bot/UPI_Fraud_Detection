import streamlit as st
import pandas as pd
import joblib
import hashlib
from xgboost import XGBClassifier
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("xgboost_model.pkl")
model_features = model.get_booster().feature_names

st.set_page_config(page_title="UPI Fraud Detection", layout="wide")
st.title("🛡️ UPI Fraud Detection System")
st.markdown("Detect potential fraudulent transactions by uploading a CSV file or entering data manually.")

# ---------------------- Helper functions ----------------------

def hash_upi(upi):
    return int(hashlib.sha256(str(upi).encode('utf-8')).hexdigest(), 16) % (10 ** 8)

def preprocess(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Weekday'] = df['Timestamp'].dt.weekday
    df['Month'] = df['Timestamp'].dt.month

    df['Sender_Provider'] = df['Sender UPI ID'].apply(lambda x: str(x).split('@')[-1])
    df['Receiver_Provider'] = df['Receiver UPI ID'].apply(lambda x: str(x).split('@')[-1])

    df['Sender_UPI_Code'] = df['Sender UPI ID'].apply(hash_upi)
    df['Receiver_UPI_Code'] = df['Receiver UPI ID'].apply(hash_upi)

    df['Is_Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['Is_High_Amount'] = df['Amount (INR)'].apply(lambda x: 1 if x > 5000 else 0)
    df['Time_Segment'] = df['Hour'].apply(lambda x: 0 if x < 6 else 1 if x < 12 else 2 if x < 18 else 3)

    df = pd.get_dummies(df, columns=['Sender_Provider', 'Receiver_Provider'])

    # Ensure all model features exist in dataframe
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]

    return df

# ---------------------- CSV Upload Section ----------------------

st.header("📤 Upload Transaction CSV")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        raw_data = pd.read_csv(uploaded_file)
        st.subheader("🔍 Uploaded Data Preview")
        st.dataframe(raw_data.head())

        data = raw_data.copy()
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])

        X_input = preprocess(data)

        st.subheader("⚙️ Running Prediction...")
        y_pred = model.predict(X_input)
        data['Predicted_Status'] = y_pred
        data['Predicted_Status'] = data['Predicted_Status'].map({0: 'SUCCESS', 1: 'FAILED'})

        st.subheader("📊 Prediction Results")
        st.dataframe(data[['Amount (INR)', 'Sender UPI ID', 'Receiver UPI ID', 'Predicted_Status']])

        st.subheader("🔢 Prediction Summary")
        st.write(data['Predicted_Status'].value_counts())

        # Visualize
        fig, ax = plt.subplots()
        sns.countplot(data['Predicted_Status'], palette='coolwarm', ax=ax)
        ax.set_title("📈 Prediction Distribution")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# ---------------------- Manual Input Section ----------------------

st.header("📝 Manual Transaction Entry")

with st.form("manual_form"):
    amount = st.number_input("Transaction Amount (INR)", min_value=1.0, step=1.0)
    sender_upi = st.text_input("Sender UPI ID (e.g., user@oksbi)")
    receiver_upi = st.text_input("Receiver UPI ID (e.g., merchant@okhdfcbank)")
    txn_date = st.date_input("Transaction Date")
    txn_time = st.time_input("Transaction Time")
    submit = st.form_submit_button("Predict")

if submit:
    try:
        timestamp = datetime.combine(txn_date, txn_time)

        manual_data = pd.DataFrame([{
            'Amount (INR)': amount,
            'Sender UPI ID': sender_upi,
            'Receiver UPI ID': receiver_upi,
            'Timestamp': timestamp
        }])

        processed = preprocess(manual_data)
        prediction = model.predict(processed)[0]

        result = "FAILED" if prediction == 1 else "SUCCESS"
        color = "red" if prediction == 1 else "green"
        st.markdown(f"### 🧾 Prediction: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error in manual prediction: {e}")
 