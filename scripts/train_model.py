import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ✅ Load preprocessed data
X_train = pd.read_csv('./data/X_train.csv')
X_test = pd.read_csv('./data/X_test.csv')
y_train = pd.read_csv('./data/y_train.csv').values.ravel()
y_test = pd.read_csv('./data/y_test.csv').values.ravel()

# ✅ Handle class imbalance by assigning higher weight to fraudulent class
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# ✅ Train GBM
print("🚀 Training GBM...")
gbm = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5)
gbm.fit(X_train, y_train)
joblib.dump(gbm, './models/gbm_model.pkl')

# ✅ Train XGBoost with class weights
print("🚀 Training XGBoost...")
xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, scale_pos_weight=class_weights[1] / class_weights[0])
xgb.fit(X_train, y_train)
joblib.dump(xgb, './models/xgboost_model.pkl')

# ✅ Train Random Forest with class weights
print("🚀 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=7)
rf.fit(X_train, y_train)
joblib.dump(rf, './models/random_forest_model.pkl')

# ✅ Train Logistic Regression with class weights
print("🚀 Training Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=200)
lr.fit(X_train, y_train)
joblib.dump(lr, './models/logistic_regression_model.pkl')

# ✅ Train MLP with dropout for better generalization
print("🚀 Training MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, alpha=0.01, random_state=42)
mlp.fit(X_train, y_train)
joblib.dump(mlp, './models/mlp_model.pkl')

# ✅ Train Autoencoder (anomaly detection)
print("🚀 Training Autoencoder...")
autoencoder = Sequential([
    Dense(32, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(X_train.shape[1], activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train Autoencoder
autoencoder.fit(X_train, X_train, epochs=30, batch_size=32, validation_data=(X_test, X_test))

# Save Autoencoder
autoencoder.save('./models/autoencoder_model.h5')

print("✅ All models trained successfully!")
