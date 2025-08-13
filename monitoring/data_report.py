#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd

from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


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

# Split the data into two batches. Run a set of pre-built data quality
# tests to evaluate the quality of the current_data:
data_stability = TestSuite(tests=[
    DataStabilityTestPreset(),
])
data_stability.run(
    current_data=prod_df, reference_data=ref_df, column_mapping=None
    )

# Evidently data drift, value drift and data summary presets
print('Evidently data drift report...')

data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report.run(
    current_data=prod_df, reference_data=ref_df, column_mapping=None
    )

# Saving data report to HTML file
print('\nSaving data report to HTML file...')
data_drift_report.save_html(str(data_report_file))

result = data_drift_report.as_dict()

print('\nPrediction drift...')
for column in selected_columns:
    drift_by_columns = result['metrics'][1]['result']['drift_by_columns']
    drift_score = drift_by_columns[column]['drift_score']
    print(f'{column}: {round(drift_score, 6)}')

# Share of drifted columns
share_of_drift = result['metrics'][1]['result']['share_of_drifted_columns']
print(f'\nShare of drifted columns: {round(share_of_drift, 6)}')

print('\nDone!')
