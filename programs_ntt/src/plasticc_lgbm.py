#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import Counter

import click
import lightgbm as gbm
import mlflow
import mlflow.sklearn
import numpy as np
from keras.utils import to_categorical
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from plasticc_simple1 import multi_weighted_log_loss
from plasticc_simple2 import make_training_data, draw_confusion_matrix

try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import matplotlib.pyplot as plt
    import seaborn as sns

__author__ = 'Yasuhiro Imoto'
__date__ = '20/12/2018'


def optimize(df, class_weight, class_map, parameters, random_state):
    oof_predictions = np.zeros((len(df), len(class_weight)))
    cv_index = np.empty(len(df), dtype=np.int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    split = skf.split(df['target'], df['target'])
    for i, (train_index, validation_index) in enumerate(split):
        train_df = df.iloc[train_index]
        validation_df = df.iloc[validation_index]

        train_counter = Counter(train_df['target'].values)
        validation_counter = Counter(validation_df['target'].values)

        ss = StandardScaler()
        # ロスの計算の都合で変換
        y_train = np.array([class_map[y] for y in train_df['target']])
        y_validation = np.array([
            class_map[y] for y in validation_df['target']
        ])
        weight_train = np.array([
            class_weight[y] / train_counter[y] for y in train_df['target']
        ])
        weight_validation = np.array([
            class_weight[y] / validation_counter[y]
            for y in validation_df['target']
        ])
        del train_df['target'], validation_df['target']

        x_train = ss.fit_transform(train_df)
        x_validation = ss.transform(validation_df)
        del train_df, validation_df

        lgb_train = gbm.Dataset(x_train, y_train, weight=weight_train)
        lgb_validation = gbm.Dataset(
            x_validation, y_validation, reference=lgb_train,
            weight=weight_validation
        )

        eval_result = {}
        model = gbm.train(
            parameters, lgb_train, valid_sets=[lgb_validation, lgb_train],
            valid_names=['validation', 'train'],
            evals_result=eval_result
        )

        for v in eval_result['train']['multi_logloss']:
            mlflow.log_metric('train_log_loss{}'.format(i), v)
        for v in eval_result['validation']['multi_logloss']:
            mlflow.log_metric('validation_log_loss{}'.format(i), v)

        oof_predictions[validation_index] = model.predict(x_validation)
        cv_index[validation_index] = i

        mlflow.sklearn.log_model(model, 'model{}'.format(i))
        path = (Path(mlflow.get_artifact_uri()) /
                'standard_scaler{}.pickle'.format(i))
        joblib.dump(ss, str(path))

    y = np.array([class_map[t] for t in df['target']])
    y = to_categorical(y, len(class_weight))
    log_loss = multi_weighted_log_loss(
        y_ohe=y, y_p=oof_predictions, class_weight=class_weight
    )
    mlflow.log_metric('loss', log_loss)

    draw_confusion_matrix(
        target=y, prediction=oof_predictions, class_map=class_map,
        path=str(Path(mlflow.get_artifact_uri()) / 'confusion_matrix.png')
    )


def evaluate(trial, n_estimators, learning_rate, subsample,
             early_stopping_round, num_iterations, data_path, seed,
             experiment_id, tracking_client, null_loss):
    if isinstance(trial, dict):
        parameters = trial
    else:
        parameters = {
            'min_samples_split': str(trial.suggest_int(
                'min_samples_split', 0, 5
            )),
            'min_samples_leaf': str(trial.suggest_int(
                'min_samples_leaf', 0, 5
            )),
            'max_depth':  str(trial.suggest_int(
                'max_depth', 2, 20
            )),
            'max_features': trial.suggest_categorical(
                'max_features', ['auto', 'log2', 'None']
            )
        }


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--data-path', type=click.Path(exists=True))
@click.option('--seed', type=int, default=42)
@click.option('--random-state', type=int, default=1)
def train(data_path, seed, random_state):
    df, classes, class_weight = make_training_data(
        data_path=data_path, only_sn=True
    )
    class_map = {c: i for i, c in enumerate(classes)}

    lgbm_params = {
        'objective': 'multiclass',
        'num_class': len(classes),
        'metric': 'multi_logloss',
        'seed': seed
    }
    with mlflow.start_run():
        optimize(
            df=df, class_weight=class_weight, class_map=class_map,
            parameters=lgbm_params, random_state=random_state
        )


@cmd.command()
@click.option('--max-runs', type=int, default=10)
@click.option('--db-dir', type=click.Path())
@click.option('--data-path', type=click.Path(exists=True))
@click.option('--seed', type=int, default=42)
@click.option('--training-experiment-id', type=int, default=-1)
@click.option('--study-name')
def search(max_runs, db_dir, data_path, seed,
           training_experiment_id, study_name):
    tracking_client = mlflow.tracking.MlflowClient()

    with mlflow.start_run() as run:
        if training_experiment_id == -1:
            experiment_id = run.info.experiment_id
        else:
            experiment_id = training_experiment_id

        pass


def main():
    cmd()


if __name__ == '__main__':
    main()
