#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd

from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.metrics import (
    ValueDrift, DriftedColumnsCount, MissingValueCount
    )


# Working directory and data file paths
work_dir = Path.home() / (
    'Programming/Python/machine-learning-exercises/'
    'higher-education-students-performance-evaluation'
    )
production_data_file = work_dir / 'data/production.parquet'
reference_data_file = work_dir / 'data/reference.parquet'
data_report_file = work_dir / 'monitoring/data_report.html'


# Selected columns for training and testing dataframes
selected_columns = [
    "Student Age", "Sex", "Graduated high-school type",
    "Scholarship type", "Parental status", "Mother's occupation",
    "Father's occupation", "Weekly study hours",
    "Attendance to classes", "Taking notes in classes",
    "Output Grade"]

prod_df = pd.read_parquet(production_data_file)
ref_df = pd.read_parquet(reference_data_file)


# Evidently data setup
schema = DataDefinition(categorical_columns=selected_columns)

prod_dataset = Dataset.from_pandas(
    pd.DataFrame(prod_df),
    data_definition=schema
    )

ref_dataset = Dataset.from_pandas(
    pd.DataFrame(ref_df),
    data_definition=schema
    )

# Evidently data drift, value drift and data summary presets
print('Evidently data drift, value drift and data summary presets...')
report = Report(
    [ValueDrift(column='Output Grade'),
     DriftedColumnsCount(),
     MissingValueCount(column='Output Grade'),
     ], include_tests='True')


snapshot = report.run(current_data=prod_dataset, reference_data=ref_dataset)

# Saving data report as HTML file
print('Saving data report to HTML file...')
snapshot.save_html(str(data_report_file))

result = snapshot.dict()

# Prediction drift
prediction_drift = result['metrics'][0]['value']
print(f'Prediction drift: {round(prediction_drift, 3)}')

# Number of drifted columns
number_drifted_columns = result['metrics'][1]['value']['count']
print(f'Number of drifted columns: {round(number_drifted_columns, 3)}')

# Share of missing values
share_missing_values = result['metrics'][2]['value']['count']
print(f'Share of missing values: {round(share_missing_values, 3)}')

print('Done!')
