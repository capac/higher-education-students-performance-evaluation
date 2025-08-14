from pathlib import Path
import logging
import pandas as pd
import psycopg

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
    )

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
    )

CONNECTION_STRING = "host=localhost port=5432 user=postgres password=example"
CONNECTION_STRING_DB = CONNECTION_STRING + " dbname=student_db"

create_table_statement = """
drop table if exists student_metrics;
create table student_metrics(
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float
)
"""

# Working directory and data file paths
work_dir = Path.home() / (
    'Programming/Python/machine-learning-exercises/'
    'higher-education-students-performance-evaluation'
    )
production_data_file = work_dir / 'data/production.parquet'
reference_data_file = work_dir / 'data/reference.parquet'

# Selected columns for training and testing dataframes
selected_columns = [
    "Student Age", "Sex", "Graduated high-school type",
    "Scholarship type", "Parental status", "Mother's occupation",
    "Father's occupation", "Weekly study hours",
    "Attendance to classes", "Taking notes in classes",
    "Output Grade"]

# Production and reference dataframes
prod_df = pd.read_parquet(production_data_file)
ref_df = pd.read_parquet(reference_data_file)

# Categorical features
cat_features = [
    "Student Age", "Sex", "Graduated high-school type",
    "Scholarship type", "Parental status", "Mother's occupation",
    "Father's occupation", "Weekly study hours",
    "Attendance to classes", "Taking notes in classes"
    ]

# Column mapping to categorical features
column_mapping = ColumnMapping(
    prediction='Output Grade',
    categorical_features=cat_features,
    target=None
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='Output Grade'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])


# Database generation
def prep_db():
    with psycopg.connect(CONNECTION_STRING, autocommit=True) as conn:
        res = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname='student_db'"
            )
        if len(res.fetchall()) == 0:
            conn.execute("create database student_db;")
        with psycopg.connect(CONNECTION_STRING_DB) as conn:
            conn.execute(create_table_statement)


# Defining Evidently metrics to database
def calculate_metrics_postgresql(curr):
    report.run(
        reference_data=ref_df, current_data=prod_df,
        column_mapping=column_mapping
        )

    result = report.as_dict()
    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = (
        result['metrics'][1]['result']['number_of_drifted_columns']
        )
    share_missing_values = (
        result['metrics'][2]['result']['current']['share_of_missing_values']
        )

    insert_query = ('INSERT INTO student_metrics VALUES (%s, %s, %s) ')
    curr.execute(
        insert_query,
        (prediction_drift, num_drifted_columns, share_missing_values)
        )


# Executing Evidently metrics to database
def batch_monitoring():
    prep_db()

    with psycopg.connect(CONNECTION_STRING_DB, autocommit=True) as conn:
        with conn.cursor() as curr:
            calculate_metrics_postgresql(curr)
            logging.info("Data sent")


if __name__ == '__main__':
    batch_monitoring()
