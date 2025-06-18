import pandas as pd

# Load the dataset
df = pd.read_csv("transactions.csv")  # Make sure this filename matches exactly

# Show the first 5 rows
print(df.head())


# Show shape of the dataset (rows, columns)
print("\nShape of dataset:", df.shape)

# Show column names
print("\nColumn names:")
print(df.columns.tolist())

# Show data types and non-null counts
print("\nData info:")
print(df.info())

# Check for missing values again
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check the distribution of the 'Status' column (SUCCESS vs FAILED)
print("\nClass distribution:")
print(df['Status'].value_counts())

# Optional: check percentage
print("\nClass distribution (in %):")
print(df['Status'].value_counts(normalize=True) * 100)

import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 1. Plot class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Status', palette='Set2')
plt.title('Transaction Status Distribution')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

# 2. Transaction amount distribution by status
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Amount (INR)', hue='Status', bins=30, kde=True, palette='Set1')
plt.title('Amount Distribution by Transaction Status')
plt.xlabel('Amount (INR)')
plt.ylabel('Frequency')
plt.show()

# 3. Transaction volume over time
df['Date'] = df['Timestamp'].dt.date
daily_counts = df.groupby(['Date', 'Status']).size().unstack()

daily_counts.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='coolwarm')
plt.title('Daily Transaction Volume by Status')
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


