import pandas as pd

# Load data
df = pd.read_csv("transactions.csv")

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract new time features
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Weekday'] = df['Timestamp'].dt.weekday
df['Month'] = df['Timestamp'].dt.month

# Drop original Timestamp, Transaction ID, and names (not useful for ML)
df.drop(columns=['Timestamp', 'Transaction ID', 'Sender Name', 'Receiver Name'], inplace=True)

# Encode categorical variables: UPI IDs & Status
df['Sender UPI ID'] = df['Sender UPI ID'].astype('category').cat.codes
df['Receiver UPI ID'] = df['Receiver UPI ID'].astype('category').cat.codes
df['Status'] = df['Status'].map({'SUCCESS': 0, 'FAILED': 1})

# Show the updated feature set
print("Processed dataset:\n", df.head())
print("\nColumn types:\n", df.dtypes)
