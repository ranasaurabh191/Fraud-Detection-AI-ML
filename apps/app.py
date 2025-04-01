import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ✅ Load the scaler
scaler = joblib.load('./models/scaler.pkl')

# ✅ Register the loss function for the autoencoder
custom_objects = {'mse': MeanSquaredError()}

# ✅ Load models safely
autoencoder = load_model('./models/autoencoder_model.h5', custom_objects=custom_objects)
gbm = joblib.load('./models/gbm_model.pkl')
xgb_model = joblib.load('./models/xgboost_model.pkl')
rf_model = joblib.load('./models/random_forest_model.pkl')
logreg_model = joblib.load('./models/logistic_regression_model.pkl')
mlp_model = joblib.load('./models/mlp_model.pkl')

# ✅ Original feature names
original_feature_names = [
    "Transaction Amount",
    "High-Risk Flag (0/1)",
    "Balance Ratio (0-1)",
    "Verification (0/1)",
    "Previous Amount",
    "Time Since Last (hrs)",
    "Merchant Reputation (0/1)",
    "Credibility Score (0-1)",
    "Location Anomaly (0/1)",
    "Recent Frequency"
]

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Ensure JSON data
        if not request.is_json:
            return jsonify({"error": "Invalid request. Expected JSON data."}), 400

        request_data = request.get_json()
        data = request_data.get('data', [])

        # ✅ Validate input format
        if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
            return jsonify({"error": "Invalid input format. Expected a list of lists."}), 400
        if len(data) == 0 or len(data[0]) != 10:
            return jsonify({"error": "Invalid input dimensions. Expected 10 features."}), 400

        # ✅ Convert input to DataFrame
        data_df = pd.DataFrame(data, columns=original_feature_names)

        # ✅ Scale the data
        try:
            data_scaled = scaler.transform(data_df)
        except Exception as e:
            return jsonify({"error": f"Data scaling failed: {str(e)}"}), 400

        # ✅ Helper function for safe predictions
        def safe_predict(model, X):
            pred = model.predict(X)
            return float(pred.flatten()[0])  # Ensure scalar output

        # ✅ Model Predictions
        gbm_pred = safe_predict(gbm, data_scaled)
        xgb_pred = safe_predict(xgb_model, data_scaled)
        rf_pred = safe_predict(rf_model, data_scaled)
        logreg_pred = safe_predict(logreg_model, data_scaled)

        # ✅ MLP Prediction
        mlp_raw_pred = np.atleast_2d(mlp_model.predict(data_scaled))  # Ensure 2D
        mlp_pred = int((mlp_raw_pred[0, 0] > 0.5))  # Thresholding

        # ✅ Autoencoder reconstruction error
        reconstruction = autoencoder.predict(data_scaled)
        recon_error = np.mean(np.square(data_scaled - reconstruction), axis=1)  # Compute per sample error
        autoencoder_pred = int(recon_error[0] > 0.01)  # First prediction

        # ✅ Debugging Logs
        print("Predictions:")
        print(f"GBM: {gbm_pred}, XGBoost: {xgb_pred}, RF: {rf_pred}, LogReg: {logreg_pred}, MLP: {mlp_pred}, Autoencoder: {autoencoder_pred}")

        # ✅ Formatting predictions
        result = {
            "Gradient Boost Method": "Fraudulent" if gbm_pred == 1 else "Genuine",
            "XGBoost": "Fraudulent" if xgb_pred == 1 else "Genuine",
            "RandomForest": "Fraudulent" if rf_pred == 1 else "Genuine",
            "Logistic Regression": "Fraudulent" if logreg_pred == 1 else "Genuine",
            "Multilayer Perceptron": "Fraudulent" if mlp_pred == 1 else "Genuine",
            "Autoencoder": "Fraudulent" if autoencoder_pred == 1 else "Genuine",
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        print("🔥 Error:", traceback.format_exc())  # Debugging traceback
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
