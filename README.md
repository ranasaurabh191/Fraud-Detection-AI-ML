# 💡 Financial Fraud Detection System 🚀


## 📌 **Project Overview**
The **Financial Fraud Detection System** is a machine learning-powered web application designed to detect fraudulent financial transactions.  
It uses multiple ML models to classify transactions as **"Fraudulent"** or **"Genuine"** based on features such as transaction amount, balance ratio, merchant reputation, etc.  

The project consists of:  
✅ A **beautiful web interface** with a stylish dark theme, dynamic form, and animated result display  
✅ A **Flask backend API** that processes transaction data, applies model predictions, and returns the results  
✅ Multiple **ML models** for accurate fraud detection  
✅ API integration to send and receive predictions  

---

## 💡 **Explanation of the Project**

### 🔥 **1️⃣ Goal of the Project**
The **main objective** of this project is to build a system capable of detecting financial fraud by analyzing transaction patterns.  
Fraudulent activities are identified based on multiple transaction attributes like **transaction amount, balance ratio, merchant reputation, user credibility score**, etc.

---

### 🛠️ **2️⃣ Datasets & Features**
The model is trained using a financial transactions dataset with multiple features:  

✅ **Time:** The timestamp of the transaction  
✅ **Amount:** The transaction value  
✅ **Balance Ratio:** The ratio between the balance before and after the transaction  
✅ **High-Risk Flag:** A binary flag indicating suspicious patterns (0 = Normal, 1 = Suspicious)  
✅ **Previous Transaction Amount:** The value of the last transaction  
✅ **Time Since Last Transaction:** Time difference (in hours) between two consecutive transactions  
✅ **Merchant Reputation:** Binary value indicating whether the merchant is trusted or not (1 = Trusted, 0 = Untrusted)  
✅ **User Credibility Score:** A normalized score between 0 and 1 indicating the user's financial credibility  
✅ **Location Anomaly:** Indicates whether the transaction was made from an unusual location (0 = Normal, 1 = Suspicious)  
✅ **Frequency of Recent Transactions:** Indicates the number of transactions made by the same user in a short time  

---

### ⚙️ **3️⃣ ML Models Used**
To make the predictions more robust, I implemented **six different machine learning models**:

1. **Logistic Regression:** A simple linear model that predicts the probability of fraud based on transaction features.  
2. **Random Forest:** An ensemble learning model that uses multiple decision trees to classify fraudulent and genuine transactions.  
3. **Gradient Boosting Machine (GBM):** A boosting algorithm that combines multiple weak models into a strong learner.  
4. **XGBoost:** An optimized implementation of gradient boosting, effective for large datasets.  
5. **Multi-Layer Perceptron (MLP):** A deep learning model with multiple hidden layers that captures complex patterns in the data.  
6. **Autoencoder:** An unsupervised neural network used for anomaly detection. It learns to compress and reconstruct genuine transactions and flags anomalies as potential fraud.

---

### 🔥 **4️⃣ Workflow Explanation**

✅ **Step 1: Frontend Form Submission**
- The user enters transaction details into the form  
- The form uses **JavaScript** to capture the input data and send it to the backend API  

✅ **Step 2: Flask API Processing**
- The Flask backend receives the transaction data in JSON format  
- The data is scaled using the **pre-trained scaler** to ensure consistent feature scaling  
- The transaction is passed through **six ML models**  
- Each model returns a prediction: either **"Fraudulent"** or **"Genuine"**  

✅ **Step 3: Displaying the Results**
- The frontend dynamically displays the predictions with:  
    - **Color-coded cards**:  
        - 🟥 **Red** → Fraudulent  
        - 🟩 **Green** → Genuine  
    - **Auto-scrolling animation** to smoothly display the results  

---

## ⚙️ **Tech Stack**
- 💻 **Frontend:** HTML, CSS, JavaScript  
- ⚙️ **Backend:** Flask (Python)  
- 🛠️ **Machine Learning Models:**  
    - Logistic Regression  
    - Random Forest  
    - Gradient Boosting Machine (GBM)  
    - XGBoost  
    - Multi-Layer Perceptron (MLP)  
    - Autoencoder  
- 📊 **Libraries Used:**  
    - `pandas`  
    - `sklearn`  
    - `xgboost`  
    - `matplotlib`  
    - `flask`  
- 🗃️ **Data Preprocessing:**  
    - Feature scaling using `StandardScaler`  

---

🛠️ API Usage
🔹 Endpoint: /predict
Method: POST
Content-Type: application/json

✅ Request Example:
json
Copy
Edit
{
  "data": [
    [1200.50, 1, 0.45, 1, 500.00, 12.5, 1, 0.8, 1, 0.75]
  ]
}
✅ Response Example:
json
Copy
Edit
{
  "Autoencoder_Prediction": "Fraudulent",
  "GBM_Prediction": "Fraudulent",
  "LogisticRegression_Prediction": "Fraudulent",
  "MLP_Prediction": "Fraudulent",
  "RandomForest_Prediction": "Genuine",
  "XGBoost_Prediction": "Genuine"
}






