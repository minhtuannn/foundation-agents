import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Load data
train_df = pd.read_csv('processed_train.csv')
test_df = pd.read_csv('processed_test.csv')

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_df.drop('target', axis=1), train_df['target'].values, test_size=0.2, random_state=42)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_df_scaled = scaler.transform(test_df)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Print validation metrics for Random Forest
y_pred_rf = rf_model.predict(X_val_scaled)
print("Random Forest Validation Metrics:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_rf):.4f}")
print(classification_report(y_val, y_pred_rf))
print(confusion_matrix(y_val, y_pred_rf))

# Train XGBoost model
xgb_model = XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Print validation metrics for XGBoost
y_pred_xgb = xgb_model.predict(X_val_scaled)
print("XGBoost Validation Metrics:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_xgb):.4f}")
print(classification_report(y_val, y_pred_xgb))
print(confusion_matrix(y_val, y_pred_xgb))

# Print final metric (choose the best model based on validation metrics)
final_metric = accuracy_score(y_val, y_pred_rf)  # or use xgb_model's accuracy
print(f"FINAL_METRIC: {final_metric:.4f}")