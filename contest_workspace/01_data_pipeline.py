import os
import glob
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

root_path = '/Users/minhtuan/Documents/Documents/Work/Hanoi/crawler/all_datasets'
contest_folder = [folder for folder in os.listdir(root_path) if folder == 'Titanic'][0]
data_folder = os.path.join(root_path, contest_folder)

file_list = glob.glob(os.path.join(data_folder, '*.csv'))
if len(file_list) == 1:
    train_file, test_file = file_list[0], None
elif len(file_list) == 2:
    train_file, test_file = file_list

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file) if test_file else None

def extract_title(name):
    titles = ['Mr.', 'Mrs.', 'Miss', 'Master', 'Dr.', 'Rev.']
    for title in titles:
        if name.startswith(title):
            return title
    return 'Unknown'

train_df['Title'] = train_df['Name'].apply(extract_title)
if test_file:
    test_df['Title'] = test_df['Name'].apply(extract_title)

sex_encoder = LabelEncoder()
train_df['Sex'] = sex_encoder.fit_transform(train_df['Sex'])
test_sex = None
if test_file:
    test_sex = sex_encoder.transform(test_df['Sex'])

age_median_by_title = train_df.groupby('Title')['Age'].median().to_dict()
def fill_age_missing(row):
    title = row['Title']
    return age_median_by_title[title] if pd.isna(row['Age']) else row['Age']

train_df['Age'] = train_df.apply(fill_age_missing, axis=1)
if test_file:
    test_df['Age'] = test_df.apply(fill_age_missing, axis=1)

def extract_deck(cabin):
    return cabin[0].upper() if pd.notna(cabin) else 'Unknown'

train_df['Cabin'] = train_df['Cabin'].apply(extract_deck)
if test_file:
    test_df['Cabin'] = test_df['Cabin'].apply(extract_deck)

embarked_encoder = OneHotEncoder(handle_unknown='ignore')
embarked_encoded = embarked_encoder.fit_transform(train_df[['Embarked']])
train_df = pd.concat([train_df, pd.DataFrame(embarked_encoded.toarray(), columns=embarked_encoder.get_feature_names(['Embarked']))], axis=1)
if test_file:
    test_embarked = embarked_encoder.transform(test_df[['Embarked']])
    test_df = pd.concat([test_df, pd.DataFrame(test_embarked.toarray(), columns=embarked_encoder.get_feature_names(['Embarked']))], axis=1)

train_df.to_csv('processed_train.csv', index=False)
if test_file:
    test_df.to_csv('processed_test.csv', index=False)

print('DATA_READY: processed_train.csv processed_test.csv')