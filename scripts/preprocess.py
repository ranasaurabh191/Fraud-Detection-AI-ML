import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # ✅ Import joblib to save the scaler

# ✅ Load the dataset
df = pd.read_csv('./data/processed_transactions.csv')

# ✅ Select only the first 10 features + the target class
X = df.drop(['Class'], axis=1).iloc[:, :10]  # Only 10 features
y = df['Class']

# ✅ Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Save the scaler
joblib.dump(scaler, './models/scaler.pkl')  # ✅ Save the scaler to use in inference

# ✅ Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Save preprocessed data
pd.DataFrame(X_train).to_csv('./data/X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('./data/X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('./data/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('./data/y_test.csv', index=False)

print("✅ Preprocessing completed with only 10 features and scaler saved!")
