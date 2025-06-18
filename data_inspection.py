import pandas as pd

# Load the dataset
df = pd.read_csv("transactions.csv")

print("🔍 Checking missing values:\n")
print(df.isnull().sum())

print("\n🧬 Unique values per column:\n")
print(df.nunique())

print("\n📊 Class distribution in 'Status':\n")
print(df['Status'].value_counts())

print("\n🎯 Class distribution (in %):")
print(df['Status'].value_counts(normalize=True) * 100)
