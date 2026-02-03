
import os
import pandas as pd

# Set the root search path
root_search_path = '/Users/minhtuan/Documents/Documents/Work/Hanoi/crawler/all_datasets'

# Find the subfolder inside ROOT SEARCH PATH that matches the Contest Name
contest_name = 'Titanic'
contest_folder = [f for f in os.listdir(root_search_path) if contest_name in f][0]

# List all files in that folder
files_in_contest_folder = os.listdir(os.path.join(root_search_path, contest_folder))

# Decide strategy based on file names
if any('train.csv' in f and 'test.csv' in f for f in files_in_contest_folder):
    # CASE A: Load train and test files directly
    train_file = [f for f in files_in_contest_folder if 'train.csv' in f][0]
    test_file = [f for f in files_in_contest_folder if 'test.csv' in f][0]
    train_df = pd.read_csv(os.path.join(root_search_path, contest_folder, train_file))
    test_df = pd.read_csv(os.path.join(root_search_path, contest_folder, test_file))
elif any('train.csv' in f and 'val.csv' in f for f in files_in_contest_folder):
    # CASE B: Merge val into train or ignore it, then load train/test
    if 'val.csv' in [f for f in files_in_contest_folder]:
        val_file = [f for f in files_in_contest_folder if 'val.csv' in f][0]
        train_df = pd.concat([pd.read_csv(os.path.join(root_search_path, contest_folder, 'train.csv')), 
                              pd.read_csv(os.path.join(root_search_path, contest_folder, val_file))])
        test_df = pd.read_csv(os.path.join(root_search_path, contest_folder, 'test.csv'))
    else:
        train_file = [f for f in files_in_contest_folder if 'train.csv' in f][0]
        test_file = [f for f in files_in_contest_folder if 'test.csv' in f][0]
        train_df = pd.read_csv(os.path.join(root_search_path, contest_folder, train_file))
        test_df = pd.read_csv(os.path.join(root_search_path, contest_folder, test_file))
else:
    # CASE C: Load single csv file as data files
    data_file = [f for f in files_in_contest_folder if 'csv' in f][0]
    df = pd.read_csv(os.path.join(root_search_path, contest_folder, data_file))

# Prepare data
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip().title())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Cabin'] = df['Cabin'].str[0]
df['Embarked'] = df['Embarked'].fillna('Unknown')
embarked_dummies = pd.get_dummies(df['Embarked'], drop_first=True)
df = pd.concat([df, embarked_dummies], axis=1)

# Detect label and data
target_col = 'Survived'
df.rename(columns={target_col: target_col}, inplace=True)

# Output
train_df.to_csv('processed_train.csv', index=False)
test_df.to_csv('processed_test.csv', index=False)

print('DATA_PROCESSED: processed_train.csv processed_test.csv')
