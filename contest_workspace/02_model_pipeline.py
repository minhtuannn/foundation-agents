import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the processed CSV files
train_df = pd.read_csv('processed_train.csv')
test_df = pd.read_csv('processed_test.csv')

# Define the models and hyperparameters
models = [
    ('XGBoost', xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100)),
    ('LightGBM', LGBMClassifier(num_leaves=31, max_depth=-1, learning_rate=0.05)),
    ('RandomForest', RandomForestClassifier(n_estimators=100, max_depth=5))
]

# Define the evaluation metric
metric = f1_score

# Train and evaluate each model on internal validation set
best_model = None
best_score = 0
for name, model in models:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(train_df, train_df.target):
        X_train, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
        y_train, y_val = train_df.target.iloc[train_idx], train_df.target.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = metric(y_val, y_pred)
        scores.append(score)
    
    avg_score = sum(scores) / len(scores)
    if avg_score > best_score:
        best_model = model
        best_score = avg_score

# Generate submission file
submission_df = pd.DataFrame({'id': test_df.id, 'target': best_model.predict(test_df)})
submission_df.to_csv('submission.csv', index=False)

# Print the final score
print(f'FINAL_SCORE: {best_score:.4f}')