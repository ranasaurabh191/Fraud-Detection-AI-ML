import tensorflow.keras.losses  # type: ignore # ✅ Register the loss function
from tensorflow.keras.models import load_model # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from flask import Flask, request, jsonify, render_template # type: ignore
from flask_cors import CORS # type: ignore

# ✅ Load the scaler
scaler = joblib.load('./models/scaler.pkl')

# ✅ Register the 'mse' loss function
custom_objects = {'mse': tensorflow.keras.losses.MeanSquaredError()}

# ✅ Load models with proper loss registration
autoencoder = load_model('./models/autoencoder_model.h5', custom_objects=custom_objects)
gbm = joblib.load('./models/gbm_model.pkl')
xgb_model = joblib.load('./models/xgboost_model.pkl')
rf_model = joblib.load('./models/random_forest_model.pkl')
logreg_model = joblib.load('./models/logistic_regression_model.pkl')
mlp_model = joblib.load('./models/mlp_model.pkl')

# ✅ Load the original feature names to ensure consistency
original_df = pd.read_csv('./data/processed_transactions.csv')
original_feature_names = original_df.columns[:-1][:10]  # Select the first 10 features used in training

app = Flask(__name__)
CORS(app)
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Parse and validate input data
        data = request.json.get('data', [])
        
        if not data or len(data[0]) != 10:
            return jsonify({"error": "Invalid input dimensions. Expected 10 features."}), 400

        # ✅ Preprocessing
        data = np.array(data)

        # ✅ Use original feature names
        data_df = pd.DataFrame(data, columns=original_feature_names)

        # ✅ Scale the data
        data_scaled = scaler.transform(data_df)

        # ✅ Helper function for safe predictions
        def safe_predict(model, X):
            """Ensures model predictions are consistently shaped"""
            pred = model.predict(X)
            return np.atleast_1d(pred).reshape(-1)[0]  # Ensure scalar output

        # ✅ Model Predictions with consistent shapes
        gbm_pred = safe_predict(gbm, data_scaled)
        xgb_pred = safe_predict(xgb_model, data_scaled)
        rf_pred = safe_predict(rf_model, data_scaled)
        logreg_pred = safe_predict(logreg_model, data_scaled)

        # ✅ MLP Prediction: Proper thresholding with safe output handling
        mlp_raw_pred = np.atleast_2d(mlp_model.predict(data_scaled))  # Ensure 2D
        mlp_pred = int((mlp_raw_pred[0, 0] > 0.5))  # Thresholding

        # ✅ Autoencoder reconstruction error
        reconstruction = autoencoder.predict(data_scaled)
        recon_error = np.mean(np.square(data_scaled - reconstruction))
        autoencoder_pred = 1 if recon_error > 0.01 else 0  # Thresholding

        # ✅ Debug: Print the predictions and shapes for verification
        print("GBM Prediction:", gbm_pred, type(gbm_pred))
        print("XGBoost Prediction:", xgb_pred, type(xgb_pred))
        print("RF Prediction:", rf_pred, type(rf_pred))
        print("Logistic Regression Prediction:", logreg_pred, type(logreg_pred))
        print("MLP Prediction:", mlp_pred, type(mlp_pred))
        print("Autoencoder Error:", recon_error, type(autoencoder_pred))

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
        print("🔥 Error:", traceback.format_exc())  # Print detailed traceback for debugging
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
