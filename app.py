import streamlit as st
import pandas as pd
import joblib
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("xgboost_model.pkl")

st.set_page_config(page_title="UPI Fraud Detection", layout="wide")
st.title("🛡️ UPI Fraud Detection System")
st.markdown("Upload a transaction CSV file or enter data manually to detect potential frauds using the trained XGBoost model.")

# ----------------------------------------
# UPI ID Hash Function
def hash_upi(upi):
    return int(hashlib.sha256(str(upi).encode('utf-8')).hexdigest(), 16) % (10 ** 8)

# ----------------------------------------
# CSV Upload Section
uploaded_file = st.file_uploader("📤 Upload transaction CSV", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("🔍 Uploaded Data Preview")
        st.dataframe(data.head())

        df = data.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        df['Hour'] = df['Timestamp'].dt.hour
        df['Day'] = df['Timestamp'].dt.day
        df['Weekday'] = df['Timestamp'].dt.weekday
        df['Month'] = df['Timestamp'].dt.month
        df.drop(columns=['Timestamp', 'Transaction ID', 'Sender Name', 'Receiver Name'], inplace=True)

        df['Sender_Provider'] = df['Sender UPI ID'].apply(lambda x: str(x).split('@')[-1])
        df['Receiver_Provider'] = df['Receiver UPI ID'].apply(lambda x: str(x).split('@')[-1])

        df = pd.get_dummies(df, columns=['Sender_Provider', 'Receiver_Provider'])

        df['Is_Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
        df['Is_High_Amount'] = df['Amount (INR)'].apply(lambda x: 1 if x > 5000 else 0)
        df['Time_Segment'] = df['Hour'].apply(lambda x: 0 if x < 6 else 1 if x < 12 else 2 if x < 18 else 3)

        df['Sender_UPI_Code'] = data['Sender UPI ID'].apply(hash_upi)
        df['Receiver_UPI_Code'] = data['Receiver UPI ID'].apply(hash_upi)

        drop_cols = ['Status', 'Sender UPI ID', 'Receiver UPI ID']
        X_input = df.drop(columns=[col for col in drop_cols if col in df.columns])

        y_pred = model.predict(X_input)
        data['Predicted_Status'] = y_pred
        data['Predicted_Status'] = data['Predicted_Status'].map({0: 'SUCCESS', 1: 'FAILED'})

        st.subheader("📊 Prediction Results")
        st.dataframe(data[['Amount (INR)', 'Sender UPI ID', 'Receiver UPI ID', 'Predicted_Status']])

        st.subheader("🔢 Prediction Summary")
        st.write(data['Predicted_Status'].value_counts())

        fig, ax = plt.subplots()
        sns.countplot(data['Predicted_Status'], palette='coolwarm', ax=ax)
        ax.set_title("📈 Predicted Transaction Status Distribution")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error processing data: {e}")

# ----------------------------------------
# Manual Input Form
st.markdown("---")
st.subheader("📝 Manually Enter Transaction Details")

with st.form("manual_form"):
    amount = st.number_input("Transaction Amount (INR)", min_value=1.0, value=500.0)
    sender_upi = st.text_input("Sender UPI ID (e.g., user@oksbi)")
    receiver_upi = st.text_input("Receiver UPI ID (e.g., merchant@okhdfcbank)")
    transaction_date = st.date_input("Transaction Date")
    transaction_time = st.time_input("Transaction Time")

    submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        try:
            # Create datetime
            timestamp = datetime.combine(transaction_date, transaction_time)
            hour = timestamp.hour
            day = timestamp.day
            weekday = timestamp.weekday()
            month = timestamp.month

            sender_provider = sender_upi.split('@')[-1]
            receiver_provider = receiver_upi.split('@')[-1]

            is_weekend = 1 if weekday >= 5 else 0
            is_high_amount = 1 if amount > 5000 else 0
            time_segment = 0 if hour < 6 else 1 if hour < 12 else 2 if hour < 18 else 3

            sender_hash = hash_upi(sender_upi)
            receiver_hash = hash_upi(receiver_upi)

            # Match feature columns from training
            input_dict = {
                'Amount (INR)': amount,
                'Day': day,
                'Hour': hour,
                'Weekday': weekday,
                'Month': month,
                'Is_Weekend': is_weekend,
                'Is_High_Amount': is_high_amount,
                'Time_Segment': time_segment,
                'Sender_UPI_Code': sender_hash,
                'Receiver_UPI_Code': receiver_hash,
            }

            # Add all known provider combinations with 0
            providers = ['okhdfcbank', 'okicici', 'oksbi', 'okybl']
            for prov in providers:
                input_dict[f'Sender_Provider_{prov}'] = 1 if prov == sender_provider else 0
                input_dict[f'Receiver_Provider_{prov}'] = 1 if prov == receiver_provider else 0

            input_df = pd.DataFrame([input_dict])
            prediction = model.predict(input_df)[0]

            st.markdown("### 🎯 Prediction Result")
            if prediction == 1:
                st.error("❌ Prediction: This transaction is likely **FAILED**.")
            else:
                st.success("✅ Prediction: This transaction is likely **SUCCESSFUL**.")

        except Exception as e:
            st.error(f"❌ Error in manual prediction: {e}")
