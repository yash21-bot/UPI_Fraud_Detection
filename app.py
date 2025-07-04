import streamlit as st
import pandas as pd
import joblib
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load model and required features ---
model = joblib.load("xgboost_model.pkl")
model_features = joblib.load("model_features.pkl")  # Ensures feature alignment

# --- UPI ID Hash Function ---
def hash_upi(upi):
    return int(hashlib.sha256(str(upi).encode('utf-8')).hexdigest(), 16) % (10 ** 8)

# --- Page Setup ---
st.set_page_config(page_title="UPI Fraud Detection", layout="wide")
st.title("🛡️ UPI Fraud Detection System")
st.markdown("Upload a transaction CSV file or enter data manually to detect potential frauds using the trained XGBoost model.")

# --- Preprocessing Function ---
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

    # Ensure all model features are present
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    return df[model_features]

# --- CSV Upload Section ---
st.header("📤 Upload Transaction CSV")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("🔍 Uploaded Data Preview")
        st.dataframe(data.head())

        df = data.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        X_input = preprocess(df)
        y_pred = model.predict(X_input)

        df['Predicted_Status'] = y_pred
        df['Predicted_Status'] = df['Predicted_Status'].map({0: 'SUCCESS', 1: 'FAILED'})

        # Save for chatbot access
        st.session_state["prediction_df"] = df

        st.subheader("📊 Prediction Results")
        st.dataframe(df[['Amount (INR)', 'Sender UPI ID', 'Receiver UPI ID', 'Predicted_Status']])

        st.subheader("🔢 Prediction Summary")
        st.write(df['Predicted_Status'].value_counts())

        fig, ax = plt.subplots()
        sns.countplot(df['Predicted_Status'], palette='coolwarm', ax=ax)
        ax.set_title("📈 Prediction Distribution")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error processing data: {e}")

# --- Manual Input Section ---
st.markdown("---")
st.subheader("📝 Manually Enter Transaction Details")

with st.form("manual_form"):
    amount = st.number_input("Transaction Amount (INR)", min_value=1.0, value=500.0)
    sender_upi = st.text_input("Sender UPI ID (e.g., user@oksbi)")
    receiver_upi = st.text_input("Receiver UPI ID (e.g., merchant@okhdfcbank)")
    txn_date = st.date_input("Transaction Date")
    txn_time = st.time_input("Transaction Time")

    submit = st.form_submit_button("🚀 Predict")

    if submit:
        try:
            timestamp = datetime.combine(txn_date, txn_time)
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

            # One-hot encode sender/receiver providers
            for col in model_features:
                if col.startswith('Sender_Provider_'):
                    input_dict[col] = 1 if col == f"Sender_Provider_{sender_provider}" else 0
                if col.startswith('Receiver_Provider_'):
                    input_dict[col] = 1 if col == f"Receiver_Provider_{receiver_provider}" else 0

            input_df = pd.DataFrame([input_dict])

            # Ensure all model features are present
            for col in model_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model_features]

            prediction = model.predict(input_df)[0]
            result = "FAILED" if prediction == 1 else "SUCCESS"
            color = "red" if prediction == 1 else "green"
            st.markdown(f"### 🧾 Prediction: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error in manual prediction: {e}")

# --- 🤖 Chatbot Assistant Section ---
st.markdown("---")
st.header("🤖 Fraud Detection Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input from user
if prompt := st.chat_input("Ask me anything about the system..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = ""
    
    if "fraud" in prompt.lower():
        response = "Fraud is detected based on unusual patterns like high amount, unknown UPI IDs, weekend usage, and night-time transactions."
    
    elif "upload" in prompt.lower():
        response = "To upload data, scroll to the '📤 Upload Transaction CSV' section and select your .csv file."

    elif "how it works" in prompt.lower() or "model" in prompt.lower():
        response = "We use an XGBoost ML model trained on transaction data to classify whether a transaction is likely to be fraudulent."

    elif "status" in prompt.lower():
        df_pred = st.session_state.get("prediction_df")
        if df_pred is None:
            response = "Please upload and analyze a transaction CSV first."
        else:
            found = False
            for _, row in df_pred.iterrows():
                if (
                    str(int(row["Amount (INR)"])) in prompt
                    and row["Sender UPI ID"].lower() in prompt.lower()
                ):
                    status = row["Predicted_Status"]
                    response = (
                        f"Transaction of ₹{row['Amount (INR)']} from **{row['Sender UPI ID']}** "
                        f"to **{row['Receiver UPI ID']}** is **{status}**."
                    )
                    found = True
                    break
            if not found:
                response = "Couldn't find a matching transaction. Try mentioning both amount and sender UPI ID."

    else:
        response = "I'm here to assist! Ask about uploading files, how fraud is detected, or the status of a transaction."

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
