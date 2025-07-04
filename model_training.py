import pandas as pd
import joblib
import hashlib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Helper function to hash UPI IDs
def hash_upi(upi):
    return int(hashlib.sha256(str(upi).encode('utf-8')).hexdigest(), 16) % (10 ** 8)

# Load dataset
df = pd.read_csv("transactions.csv")

# Feature engineering
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


# Encode target
df['Status'] = df['Status'].map({'SUCCESS': 0, 'FAILED': 1})

# Prepare input/output
X = df.drop(columns=[
    'Status', 'Timestamp', 'Transaction ID', 'Sender Name', 'Receiver Name',
    'Sender UPI ID', 'Receiver UPI ID'
], errors='ignore')
y = df['Status']


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Save model
joblib.dump(model, "xgboost_model.pkl")
print("✅ Model saved as xgboost_model.pkl")

# Save feature names
model_features = list(X_train.columns)
joblib.dump(model_features, "model_features.pkl")
print("✅ Feature list saved as model_features.pkl")
