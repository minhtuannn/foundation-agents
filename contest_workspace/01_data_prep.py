import os
import glob
import pandas as pd

# 1. Find the folder matching the Contest Name
root_path = '/Users/minhtuan/Documents/Documents/Work/Hanoi/crawler/all_datasets'
contest_folder = [name for name in os.listdir(root_path) if name == 'Titanic'][0]
data_folder = os.path.join(root_path, contest_folder)

# 2. List CSV files
csv_files = glob.glob(os.path.join(data_folder, '*.csv'))

# Check if exactly one file found (train.csv and test.csv)
if len(csv_files) == 1:
    # Load the single file
    df = pd.read_csv(csv_files[0])
    
    # Split data into training and testing sets (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1), df['Survived'], test_size=0.2, random_state=42)
    
    # Save processed data
    X_train.to_csv('processed_train.csv', index=False)
    X_test.to_csv('processed_test.csv', index=False)
    print('DATA_PROCESSED: processed_train.csv processed_test.csv')
elif len(csv_files) == 2:
    # Identify 'train'/'test' by name
    train_file, test_file = [file for file in csv_files if 'train' in file], [file for file in csv_files if 'test' in file]
    
    # Load the files
    df_train = pd.read_csv(train_file[0])
    df_test = pd.read_csv(test_file[0])
    
    # Save processed data
    df_train.to_csv('processed_train.csv', index=False)
    df_test.to_csv('processed_test.csv', index=False)
    print('DATA_PROCESSED: processed_train.csv processed_test.csv')
else:
    raise ValueError("Invalid number of CSV files found")