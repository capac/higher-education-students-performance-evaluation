#!/usr/bin/env python
# coding: utf-8

import json
from pathlib import Path
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split


# Working directory and file paths
work_dir = Path.home() / (
    'Programming/Python/machine-learning-exercises/'
    'higher-education-students-performance-evaluation'
    )
data_file = work_dir / 'data/students-performance.csv'
attribute_names_json_file = work_dir / 'attribute_names.json'
preprocessor_file = work_dir / 'preprocessor.bin'
best_model_file = work_dir / 'xgb_cls.bin'
production_data_file = work_dir / 'monitoring/data/production.parquet'
reference_data_file = work_dir / 'monitoring/data/reference.parquet'


# Loading attribute data set
print('Loading attribute data set...')
with open(attribute_names_json_file, 'rt') as f_in:
    attribute_names_json = json.load(f_in)

labels_dict = {}
string_indexes = [str(id) for id in range(1, 33)]
for ind in string_indexes:
    label = attribute_names_json[ind]['name']
    labels_dict[ind] = label
labels_dict['0'] = 'STUDENT ID'

sp_df = pd.read_csv(data_file)
sp_df.rename(columns=labels_dict, inplace=True)

X = sp_df.drop(['STUDENT ID', 'GRADE'], axis=1)
y = sp_df['GRADE'].copy()


# Split into training and testing data sets
print('Split into training and testing data sets...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=33
    )

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# Training and testing predictions dataframes
print('Training and testing predictions dataframes...')
with open(preprocessor_file, 'rb') as f_in:
    preprocessor = pickle.load(f_in)

with open(best_model_file, 'rb') as f_in:
    model = pickle.load(f_in)

X_train_tr = preprocessor.transform(X_train)
X_test_tr = preprocessor.transform(X_test)
train_predictions = model.predict(X_train_tr)
test_predictions = model.predict(X_test_tr)


# Selected columns for training and testing dataframes
print('Selected columns for training and testing dataframes...')
selected_columns = [
    "Student Age", "Sex", "Graduated high-school type", "Scholarship type",
    "Parental status", "Mother's occupation", "Father's occupation",
    "Weekly study hours", "Attendance to classes", "Taking notes in classes"
    ]

prod_df = pd.DataFrame()
prod_df[selected_columns] = X_train[selected_columns]
prod_df['Output Grade'] = train_predictions

ref_df = pd.DataFrame()
ref_df[selected_columns] = X_test[selected_columns]
ref_df['Output Grade'] = test_predictions


# Saving data frames as Parquet files
print('Saving data frames as Parquet files...')
prod_df.to_parquet(production_data_file)
ref_df.to_parquet(reference_data_file)
print('Done!')
