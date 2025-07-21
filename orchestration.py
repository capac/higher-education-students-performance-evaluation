#!/usr/bin/env python
# coding: utf-8

import json
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import scipy.sparse
from typing import Tuple

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import (
    LogisticRegression, SGDClassifier
    )
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
    )
from sklearn.svm import SVC

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import mlflow
from prefect import task, flow


work_dir = Path.home() / (
    'Programming/Python/machine-learning-exercises/'
    'higher-education-students-performance-evaluation'
    )
data_file = work_dir / 'data/students-performance.csv'
attribute_names_json_file = work_dir / 'attribute_names.json'
xgb_model_file = work_dir / 'models/xgb_cls.bin'


@task(retries=3, retry_delay_seconds=2)
def read_data(data_file: str) -> pd.DataFrame:
    """Read data file into data frame"""
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

    return sp_df


@task
def add_features(sp_df: pd.DataFrame) -> Tuple[
        scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix,
        np.ndarray,
        np.ndarray,
        ]:
    """Add features to the model"""
    X = sp_df.drop(['STUDENT ID', 'GRADE'], axis=1)
    y = sp_df['GRADE'].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=33
        )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Data preparation
    cat_attribs = sp_df.columns[1:-1]
    cat_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))
    preprocessing = ColumnTransformer([("cat", cat_pipeline, cat_attribs)])

    X_train_tr = preprocessing.fit_transform(X_train)
    X_test_tr = preprocessing.transform(X_test)

    return X_train_tr, X_test_tr, y_train, y_test


# Scikit-Learn Classifiers
@task(log_prints=True)
def train_best_sklearn_model(
    X_train_tr: scipy.sparse.csr_matrix,
    X_test_tr: scipy.sparse.csr_matrix,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """train Scikit-Learn models and save results to MLflow"""

    mlflow.sklearn.autolog()
    model_names = ['rf_cls', 'gbc_cls', 'etc_cls', 'sgdc_cls',
                   'svc_cls', 'dtc_cls', 'log_reg']
    model_classifiers = [RandomForestClassifier(),
                         GradientBoostingClassifier(),
                         ExtraTreesClassifier(),
                         SGDClassifier(loss='log_loss'),
                         SVC(probability=True),
                         DecisionTreeClassifier(),
                         LogisticRegression()]

    for model_name, model_class in zip(model_names, model_classifiers):
        with mlflow.start_run():
            mlflow.set_tag('model', model_class.__class__.__name__)

            model_class.fit(X_train_tr, y_train.to_numpy())
            y_pred = model_class.predict_proba(X_test_tr)

            auc = roc_auc_score(y_test, y_pred, multi_class='ovo')
            mlflow.log_metric("AUC", auc)

            Path("models").mkdir(exist_ok=True)
            model_file = f'models/{model_name}.bin'
            with open(model_file, 'wb') as f_out:
                pickle.dump(model_class, f_out)


# XGBoost Classifier with hyperparameter tuning
@task(log_prints=True)
def train_best_xgboost_model(
    X_train_tr: scipy.sparse.csr_matrix,
    X_test_tr: scipy.sparse.csr_matrix,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Train XGBoost models with best hyperparams and save results to MLflow"""

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag('model', 'XGBoost')
            mlflow.log_params(params)
            clf = xgb.XGBClassifier(
                **params,
                eval_metric='auc',
                early_stopping_rounds=50,
                n_jobs=-1,
            )
            clf.fit(
                X_train_tr,
                y_train,
                eval_set=[(X_test_tr, y_test)],
            )
            y_pred = clf.predict_proba(X_test_tr)
            auc = roc_auc_score(y_test, y_pred, multi_class='ovo')
            mlflow.log_metric("AUC", auc)

        return {'loss': auc, 'status': STATUS_OK}

    xgboost_search_space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 4, 200, 1)),
        'max_depth': scope.int(hp.quniform('max_depth', 4, 50, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'binary:logistic',
        'seed': 42
        }

    _ = fmin(
        fn=objective,
        space=xgboost_search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )

    # Saving XGBoost model with best parameters
    xgboost_best_params = {
        'learning_rate': 0.1822609570893024,
        'max_depth': 16,
        'min_child_weight': 0.4639113171017813,
        'n_estimators': 131,
        'objective': 'binary:logistic',
        'reg_alpha': 0.007860242176975434,
        'reg_lambda': 0.02768073078548693,
        'seed': 42,
    }

    with mlflow.start_run():
        mlflow.set_tag('model', 'XGBoost')
        mlflow.log_params(xgboost_best_params)
        xgboost_clf = xgb.XGBClassifier(
            **xgboost_best_params,
            eval_metric='auc',
            early_stopping_rounds=50,
            n_jobs=-1,
        )
        xgboost_clf.fit(
            X_train_tr,
            y_train,
            eval_set=[(X_test_tr, y_test)],
        )
        y_pred = xgboost_clf.predict_proba(X_test_tr)
        auc = roc_auc_score(y_test, y_pred, multi_class='ovo')
        mlflow.log_metric("AUC", auc)

        Path("models").mkdir(exist_ok=True)
        with open(xgb_model_file, 'wb') as f_out:
            pickle.dump(xgboost_clf, f_out)

        mlflow.log_artifact(xgb_model_file, artifact_path='best_models')

        mlflow.xgboost.log_model(xgboost_clf, artifact_path='artifacts')

    return None


@flow
def main_flow(
    data_file: Path = work_dir / 'data/students-performance.csv',
) -> None:
    """The main training pipeline"""

    # MLflow settings
    experiment_name = 'higher-education-students-performance-evaluation'
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name, artifact_location='artifacts'
            )

    # Load data from file
    sp_df = read_data(data_file)

    # Transform to traning and testing data
    X_train_tr, X_test_tr, y_train, y_test = add_features(sp_df)

    # Train best Scikit-Learn models
    train_best_sklearn_model(X_train_tr, X_test_tr, y_train, y_test)

    # Train best XGBoost model
    train_best_xgboost_model(X_train_tr, X_test_tr, y_train, y_test)


if __name__ == "__main__":
    main_flow()
