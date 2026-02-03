import os
import glob
import pandas as pd

root_path = '/Users/minhtuan/Documents/Documents/Work/Hanoi/crawler/all_datasets'

contest_folder = [folder for folder in os.listdir(root_path) if folder == 'titanic'][0]

train_file = glob.glob(os.path.join(root_path, contest_folder, '*', 'train.csv'))[0]
test_file = glob.glob(os.path.join(root_path, contest_folder, '*', 'test.csv'))[0]

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Drop PassengerId column
train_df.drop('PassengerId', axis=1, inplace=True)
test_df.drop('PassengerId', axis=1, inplace=True)

# Extract Title from Name and create new feature 'Title'
train_df['Title'] = train_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip().title())
test_df['Title'] = test_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip().title())

# Map Sex to 0/1
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# Fill missing Age values by Title
age_median_by_title = train_df.groupby('Title')['Age'].median().reset_index(name='Age')
train_df = pd.merge(train_df, age_median_by_title, how='left', on='Title').fillna(train_df['Age'])
test_df = pd.merge(test_df, age_median_by_title, how='left', on='Title').fillna(test_df['Age'])

# Extract first letter of Cabin as 'Deck' and fill missing with 'Unknown'
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: x[0] if not pd.isna(x) else 'Unknown')
test_df['Cabin'] = test_df['Cabin'].apply(lambda x: x[0] if not pd.isna(x) else 'Unknown')

# One-Hot Encode Embarked
train_df = pd.get_dummies(train_df, columns=['Embarked'])
test_df = pd.get_dummies(test_df, columns=['Embarked'])

# Split data into training and validation sets (80/20)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Save processed data
train_df.to_csv('processed_train.csv', index=False)
val_df.to_csv('processed_val.csv', index=False)
test_df.to_csv('processed_test.csv', index=False)

print('DATA_READY: processed_train.csv processed_val.csv processed_test.csv')