import os
import glob
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set root path to find the dataset folder
root_path = "/Users/minhtuan/Documents/Documents/Work/Hanoi/crawler/all_datasets"

# Find the folder matching the Contest Name
contest_folder = [folder for folder in os.listdir(root_path) if "titanic" in folder][0]
data_folder = os.path.join(root_path, contest_folder)

# Check file count using glob
files = glob.glob(os.path.join(data_folder, "*.csv"))
file_count = len(files)
if file_count == 1:
    # Load the single file as both train and test sets
    data_file = files[0]
    train_data = pd.read_csv(os.path.join(data_folder, data_file))
    test_data = train_data.copy()
elif file_count > 1:
    raise ValueError("Invalid number of files found")

# Define preprocessing steps
categorical_cols = ["Pclass", "Sex", "Embarked"]
numerical_cols = ["Age"]

# Preprocess categorical columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImporter(strategy='constant', fill_value='Unknown')),
    ('encoder', OrdinalEncoder(categories=[['1st', '2nd', '3rd'], ['male', 'female'], ['C', 'Q', 'S']]))
])

# Preprocess numerical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', categorical_transformer, categorical_cols),
        ('numerical', numerical_transformer, numerical_cols)
    ]
)

train_data_processed = preprocessor.fit_transform(train_data.drop("PassengerId", axis=1))
test_data_processed = preprocessor.transform(test_data.drop("PassengerId", axis=1))

# Verify preprocessing steps
assert train_data_processed.shape[0] == train_data.shape[0]
assert test_data_processed.shape[0] == test_data.shape[0]

# Save processed data
train_data_processed_df = pd.DataFrame(train_data_processed, columns=categorical_cols + numerical_cols)
test_data_processed_df = pd.DataFrame(test_data_processed, columns=categorical_cols + numerical_cols)

train_data_processed_df.to_csv('processed_train.csv', index=False)
test_data_processed_df.to_csv('processed_test.csv', index=False)

print("DATA_READY: processed_train.csv processed_test.csv")