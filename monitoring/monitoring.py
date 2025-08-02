#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd

from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset, DataSummaryPreset


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

eval_data_1 = Dataset.from_pandas(
    pd.DataFrame(prod_df),
    data_definition=schema
    )

eval_data_2 = Dataset.from_pandas(
    pd.DataFrame(ref_df),
    data_definition=schema
    )

# Evidently data drift, value drift and data summary presets
print('Evidently data drift, value drift and data summary presets...')
report = Report(
    [DataDriftPreset(),
     ValueDrift(column='Output Grade'),
     DataSummaryPreset(),
     ], include_tests='True')


snapshot = report.run(current_data=eval_data_1, reference_data=eval_data_2)
print('Saving data report to HTML file...')
snapshot.save_html('data_report.html')
print('Done!')
