import pandas as pd
import joblib
from datetime import datetime

# Load the saved model
model = joblib.load("fraud_model.pkl")

# Example new transaction
new_transaction = {
    "Sender UPI ID": "9876543210@okicici",
    "Receiver UPI ID": "1234567890@okhdfcbank",
    "Amount (INR)": 5200.00,
    "Timestamp": "2024-06-27 14:35:00"
}

# Convert to DataFrame
new_df = pd.DataFrame([new_transaction])

# Feature Engineering (same as training)
new_df['Timestamp'] = pd.to_datetime(new_df['Timestamp'])
new_df['Hour'] = new_df['Timestamp'].dt.hour
new_df['Day'] = new_df['Timestamp'].dt.day
new_df['Weekday'] = new_df['Timestamp'].dt.weekday
new_df['Month'] = new_df['Timestamp'].dt.month
new_df.drop(columns=['Timestamp'], inplace=True)

# Encode UPI IDs using the same strategy
new_df['Sender UPI ID'] = pd.factorize(new_df['Sender UPI ID'])[0]
new_df['Receiver UPI ID'] = pd.factorize(new_df['Receiver UPI ID'])[0]

# Predict
prediction = model.predict(new_df)[0]

# Output
result = "❌ FRAUD (FAILED)" if prediction == 1 else "✅ SUCCESSFUL"
print(f"Prediction result: {result}")
