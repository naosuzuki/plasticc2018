#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import math
from collections import Counter
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import mlflow
import mlflow.keras
import numpy as np
import optuna
import pandas as pd
import sklearn.utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.utils import to_categorical, Sequence
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from plasticc_simple1 import multi_weighted_log_loss, weighted_log_loss

try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import matplotlib.pyplot as plt
    import seaborn as sns

__author__ = 'Yasuhiro Imoto'
__date__ = '18/12/2018'


class TrainingDataset(Sequence):
    def __init__(self, df, standard_scaler, batch_size,
                 is_training, class_map):
        self.object_id = df.index
        self.y = to_categorical(
            np.array([class_map[y] for y in df['target']]),
            len(class_map)
        )

        del df['target']
        if is_training:
            self.x = standard_scaler.fit_transform(df)
        else:
            self.x = standard_scaler.transform(df)

        self.is_training = is_training
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.object_id) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        s = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_x = self.x[s]
        batch_y = self.y[s]

        return batch_x, batch_y

    def on_epoch_end(self):
        if not self.is_training:
            return

        self.x, self.y, self.object_id = sklearn.utils.shuffle(
            self.x, self.y, self.object_id
        )


def make_training_data(data_path, only_sn=False, only_extra=False,
                       only_intra=False):
    """
    特徴量に転換されてtargetなどの必要な情報も結合されたcsvが与えられるものとする
    特徴量への変換はここでは関知しない

    :param data_path: 学習データのcsv
    :param only_sn:
    :param only_extra:
    :param only_intra:
    :return:
    """
    # データ数が少ない順
    if only_sn:
        classes = (95, 52, 62, 42, 90)
    elif only_extra:
        classes = (95, 52, 67, 62, 15, 42, 90)
    elif only_intra:
        classes = (53, 6, 92, 16, 65)
    else:
        classes = (53, 64, 6, 95, 52, 67, 92, 88, 62, 15, 16, 65, 42, 90)

    df = pd.read_csv(data_path, index_col=0, header=0)
    if only_sn or only_extra or only_intra:
        # 目的のクラスだけにする
        flag = np.zeros(len(df), dtype=np.bool)
        for c in classes:
            flag = np.logical_or(flag, df['target'] == c)
        df = df[flag]

    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    class_weight = {
        c: 1 if c not in (64, 15) else 2 for c in classes
    }

    print('Unique classes : ', classes)
    return df, classes, class_weight


def create_model(parameters, weight_table, input_size):
    K.clear_session()

    model = Sequential()
    for i, (hidden_size, drop_rate) in enumerate(zip(
            parameters['hidden_size'], parameters['drop_rate'])):
        if i == 0:
            model.add(Dense(
                hidden_size, input_shape=(input_size,), activation='relu'
            ))
        else:
            model.add(Dense(hidden_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
    model.add(Dense(len(weight_table), activation='softmax'))

    model.compile(
        optimizer=parameters['optimizer'],
        loss=lambda y_true, y_pred: weighted_log_loss(
            y_true=y_true, y_pred=y_pred, weight_table=weight_table
        ),
        metrics=['accuracy']
    )

    return model


class MLflowCheckpoint(Callback):
    def __init__(self, cv_index):
        super().__init__()

        self.train_loss_name = 'train_log_loss{}'.format(cv_index)
        self.validation_loss_name = 'validation_log_loss{}'.format(cv_index)
        self._best_train_loss = math.inf
        self._best_validation_loss = math.inf

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        train_loss = logs["loss"]
        validation_loss = logs["val_loss"]
        mlflow.log_metric(self.train_loss_name, train_loss)
        mlflow.log_metric(self.validation_loss_name, validation_loss)

        if validation_loss < self._best_validation_loss:
            self._best_train_loss = train_loss
            self._best_validation_loss = validation_loss
        mlflow.log_metric(
            'best_{}'.format(self.train_loss_name), self._best_train_loss
        )
        mlflow.log_metric(
            'best_{}'.format(self.validation_loss_name),
            self._best_validation_loss
        )


def optimize(df, class_weight, class_map, batch_size, epochs, parameters):
    oof_predictions = np.zeros((len(df), len(class_weight)))
    cv_index = np.empty(len(df), dtype=np.int)

    if epochs == 0:
        oof_predictions[:] = 1.0 / len(class_map)

        y = np.array([class_map[t] for t in df['target']])
        y = to_categorical(y, len(class_weight))
        log_loss = multi_weighted_log_loss(
            y_ohe=y, y_p=oof_predictions, class_weight=class_weight
        )
        mlflow.log_metric('loss', log_loss)

        return

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    split = skf.split(df['target'], df['target'])
    for i, (train_index, validation_index) in enumerate(split):
        train_df = df.iloc[train_index]
        validation_df = df.iloc[validation_index]

        weight_table = np.zeros(len(class_weight), dtype=np.float32)
        count = Counter(train_df['target'].values)
        for j, (_, value) in enumerate(sorted(count.items(),
                                              key=lambda x: x[1])):
            weight_table[j] = value / len(train_index)

        ss = StandardScaler()

        train_dataset = TrainingDataset(
            df=train_df, standard_scaler=ss, batch_size=batch_size,
            is_training=True, class_map=class_map
        )
        validation_dataset = TrainingDataset(
            df=validation_df, standard_scaler=ss, batch_size=batch_size,
            is_training=False, class_map=class_map
        )

        model = create_model(
            parameters=parameters, weight_table=weight_table,
            input_size=train_dataset.x.shape[1]
        )

        with TemporaryDirectory() as tmp_dir:
            tmp_path = str(Path(tmp_dir) / 'keras.model')
            check_point = ModelCheckpoint(
                tmp_path, monitor='val_loss', mode='min',
                save_best_only=True, verbose=0
            )
            mlflow_logger = MLflowCheckpoint(cv_index=i)

            model.fit_generator(
                generator=train_dataset,
                validation_data=validation_dataset,
                epochs=epochs, shuffle=True, verbose=0,
                callbacks=[check_point, mlflow_logger]
            )

            model.load_weights(tmp_path)
            predictions = model.predict_generator(validation_dataset)

            oof_predictions[validation_index] = predictions
            cv_index[validation_index] = i

            # 一番validationが良かった時の値はMLflowCheckpointが記録しているので、
            # モデルだけ保存
            mlflow.keras.log_model(model, 'model{}'.format(i))

        # foldのインデックスに固有で、全ての探索で共通だが、
        # 入力の特徴量を管理が大変になるので、素直にそれぞれで保存する
        path = (Path(mlflow.get_artifact_uri()) /
                'standard_scaler{}.pickle'.format(i))
        if not path.exists():
            joblib.dump(ss, str(path))

    y = np.array([class_map[t] for t in df['target']])
    y = to_categorical(y, len(class_weight))
    log_loss = multi_weighted_log_loss(
        y_ohe=y, y_p=oof_predictions, class_weight=class_weight
    )
    mlflow.log_metric('loss', log_loss)

    np.savez_compressed(
        str(Path(mlflow.get_artifact_uri()) / 'predictions.npz'),
        prediction=oof_predictions, cv_index=cv_index,
        target=df['target'].values, class_map=class_map
    )
    draw_confusion_matrix(
        target=y, prediction=oof_predictions, class_map=class_map,
        path=str(Path(mlflow.get_artifact_uri()) / 'confusion_matrix.png')
    )


def draw_confusion_matrix(target, prediction, class_map, path):
    # class_mapはクラス名からインデックスを答えるので、逆変換のリストを作る
    class_list = np.empty(len(class_map), dtype=np.int)
    for key, value in class_map.items():
        class_list[value] = key

    y_true = np.argmax(target, axis=1)
    y_pred = np.argmax(prediction, axis=1)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    cm = cm / np.sum(cm, axis=1, keepdims=True)
    annotation = np.around(cm, 2)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(cm, xticklabels=class_list, yticklabels=class_list,
                cmap='Blues', annot=annotation, lw=0.5, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_aspect('equal')
    fig.tight_layout()

    fig.savefig(path, bbox_inches='tight')
    plt.close()


def evaluate(trial, epochs, batch_size, data_path, seed,
             experiment_id, tracking_client, null_loss):
    if isinstance(trial, dict):
        parameters = trial
    else:
        hidden_size, drop_rate = [], []
        n_hidden_layers = trial.suggest_int('n_hidden_layers', 2, 7)
        for i in range(n_hidden_layers):
            hidden_size.append(str(int(trial.suggest_discrete_uniform(
                'hidden_size{}'.format(i), 10, 500, 10
            ))))
            drop_rate.append(str(trial.suggest_uniform(
                'drop_rate{}'.format(i), 0.01, 0.99
            )))
        optimizer = trial.suggest_categorical(
            "optimizer", ["sgd", "adam", "rmsprop"]
        )

        parameters = dict(
            hidden_size=','.join(hidden_size),
            drop_rate=','.join(drop_rate), optimizer=optimizer
        )
    parameters.update(dict(
        epochs=str(epochs), batch_size=str(batch_size), data_path=data_path
    ))

    p = mlflow.projects.run(
        uri='.', entry_point='train', parameters=parameters,
        experiment_id=experiment_id
    )

    if p.wait():
        training_run = tracking_client.get_run(p.run_id)
        loss = min(
            null_loss,
            get_metric(training_run=training_run, metric_name='loss')
        )
    else:
        loss = null_loss

    mlflow.log_metric('loss', loss)
    return loss


def get_metric(training_run, metric_name):
    return [m.value for m in training_run.data.metrics if
            m.key == metric_name][0]


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--data-path', type=click.Path(exists=True))
@click.option('--epochs', type=int, default=1000)
@click.option('--batch-size', type=int, default=100)
@click.option('--hidden-size', '-h', type=str)
@click.option('--drop-rate', '-d', type=str)
@click.option('--optimizer', type=click.Choice(["sgd", "adam", "rmsprop"]))
def train(data_path, epochs, batch_size, hidden_size, drop_rate,
          optimizer):
    hidden_size = eval('[{}]'.format(hidden_size))
    drop_rate = eval('[{}]'.format(drop_rate))

    df, classes, class_weight = make_training_data(
        data_path=data_path, only_sn=True
    )
    class_map = {c: i for i, c in enumerate(classes)}

    parameters = {'hidden_size': hidden_size, 'drop_rate': drop_rate,
                  'optimizer': optimizer}
    with mlflow.start_run():
        optimize(
            df=df, class_weight=class_weight, class_map=class_map,
            batch_size=batch_size, epochs=epochs, parameters=parameters
        )


@cmd.command()
@click.option('--max-runs', type=int, default=10)
@click.option('--epochs', type=int, default=500)
@click.option('--batch-size', type=int, default=100)
@click.option('--db-dir', type=click.Path())
@click.option('--data-path', type=click.Path(exists=True))
@click.option('--seed', type=int, default=42)
@click.option('--training-experiment-id', type=int, default=-1)
@click.option('--study-name')
def search(max_runs, epochs, batch_size, db_dir, data_path, seed,
           training_experiment_id, study_name):
    tracking_client = mlflow.tracking.MlflowClient()

    with mlflow.start_run() as run:
        if training_experiment_id == -1:
            experiment_id = run.info.experiment_id
        else:
            experiment_id = training_experiment_id

        f = partial(
            evaluate, batch_size=batch_size,
            data_path=data_path, seed=seed, experiment_id=experiment_id,
            tracking_client=tracking_client
        )
        # Evaluate null model first.
        null_loss = f(
            trial={'hidden_size': '100', 'drop_rate': '0.5',
                   'optimizer': 'sgd'},
            epochs=0,
            null_loss=math.inf
        )

        storage = 'sqlite:///{}/example.db'.format(db_dir)
        db_dir = Path(db_dir)
        if not db_dir.exists():
            db_dir.mkdir(parents=True)

        if (db_dir / 'example.db').exists():
            study = optuna.Study(study_name=study_name, storage=storage)
        else:
            study = optuna.create_study(
                study_name=study_name, storage=storage
            )
        study.optimize(
            lambda trial: f(trial=trial, epochs=epochs, null_loss=null_loss),
            n_trials=max_runs
        )

        df = study.trials_dataframe()
        df.to_csv(db_dir / 'optimization.csv')
        print('best params: {}'.format(study.best_params))
        print('best value: {}'.format(study.best_value))
        print('best trial: {}'.format(study.best_trial))


def main():
    cmd()


if __name__ == '__main__':
    main()
