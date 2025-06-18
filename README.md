# UPI Fraud Detection System

An end-to-end machine learning project that detects potential fraudulent UPI (Unified Payments Interface) transactions using data analysis and an XGBoost classifier. This project includes a **Streamlit UI** that allows users to:

* Upload a CSV file of transactions for batch prediction
* Enter individual transaction details for instant manual prediction

---

## 🚀 Features

* **Model**: XGBoost classifier
* **Accuracy**: \~57% (can be further improved)
* **Interactive UI**: Built with Streamlit
* **Feature Engineering**: Extracts time-based patterns (hour, day, weekday, etc.) and UPI provider insights
* **Encodes UPI IDs securely** using hashing
* **One-hot encoding** of sender/receiver UPI providers
* **Robust feature alignment** for prediction compatibility

---

## 📂 Project Structure

```
upi_fraud_detection/
├── app.py                      # Streamlit UI app
├── model_training.py          # Script to train and save the model
├── feature_engineering.py     # Optional - utilities for preprocessing
├── fraud_model.pkl            # Old trained model (optional)
├── xgboost_model.pkl          # ✅ Final trained model
├── data_exploration.py        # Optional - exploratory analysis
├── transactions.csv           # Sample transaction data
├── dataset_upifraud.zip       # Full dataset (zipped)
├── venv/                      # Python virtual environment
```

---

## 📊 Dataset

- Source: [Kaggle - UPI Fraud Detection Dataset](https://www.kaggle.com/)
- The dataset includes columns like:
  - `Transaction ID`
  - `Sender Name`, `Receiver Name`
  - `Sender UPI ID`, `Receiver UPI ID`
  - `Amount (INR)`
  - `Timestamp`
  - `Status` (Success / Failed)

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```
streamlit
pandas
xgboost
scikit-learn
matplotlib
seaborn
```

---

## 💡 How to Run

### 1. Train the Model (Optional)

If you want to retrain the model:

```bash
python model_training.py
```

This generates `xgboost_model.pkl`.

### 2. Launch the Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 Usage

### 📤 CSV Upload:

* Upload a transaction CSV with at least these columns:

```
Amount (INR), Sender UPI ID, Receiver UPI ID, Timestamp
```

* The app will show predictions and summary charts.

### ✍️ Manual Input:

* Enter:

  * Amount (INR)
  * Sender and Receiver UPI IDs
  * Transaction Date and Time
* Get instant fraud prediction

---

## 🎯 Feature Engineering

* **Sender/Receiver Provider** extracted from UPI ID suffix (`@oksbi`, `@okhdfcbank`, etc.)
* **Sender/Receiver UPI ID** encoded via SHA-256 hashing (for privacy)
* **Time-Based Features**: Hour, Day, Weekday, Month, Time Segment
* **Weekend and High Amount Flags**

---

## 📈 Model Evaluation

Confusion Matrix:

```
[[63 40]
 [46 51]]
```

Accuracy: **57%**

> Can be improved using SMOTE, better features, or deep learning.

---


---

## 🤝 Contributing

Pull requests and suggestions are welcome. Let me know if you want to improve the model or the UI.

---

## 📜 License

This project is for educational purposes only. Use responsibly in real-world applications.

## ⭐️ If you find this project helpful, star it on GitHub!
