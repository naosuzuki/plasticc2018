#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import json
import math
import os
from collections import namedtuple
from datetime import datetime
from functools import partial

import bson
import click
import numpy as np
import tensorflow as tf
import xarray as xr

try:
    from tensorflow import keras
except ImportError:
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.keras.python import keras
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.pyll import scope
from sklearn.externals import joblib
# noinspection PyProtectedMember
from sklearn.utils import shuffle

__author__ = 'Yasuhiro Imoto'
__date__ = '13/2/2018'


Dataset = namedtuple('Dataset', ['flux', 'flux_err', 'label', 'redshift',
                                 'name', 'sn_epoch', 'min_elapsed_day'])


def load_data(file_path, band_data):
    ds = xr.open_dataset(file_path)  # type: xr.Dataset
    band_list = np.unique(ds.band.values)

    flux = []
    flux_err = []
    for band in band_list:
        # バンド名がband_dataに含まれているかを確認する
        if isinstance(band, bytes):
            band_name = band.decode()
        else:
            band_name = band
        if band_name not in band_data:
            # バンド名が含まれていないので、利用しない
            continue

        tmp = ds.where(ds.band == band, drop=True)  # type: xr.Dataset
        indices = tmp.index.values
        indices = np.sort(indices)
        for i in indices:
            flux.append(tmp.flux.where(tmp.index == i, drop=True).values)
            flux_err.append(
                tmp.flux_err.where(tmp.index == i, drop=True).values
            )

    flux = np.nan_to_num(np.hstack(flux))
    flux_err = np.nan_to_num(np.hstack(flux_err))

    dataset = Dataset(flux=flux, flux_err=flux_err, label=ds.label.values,
                      redshift=ds.redshift.values, name=ds.sample.values,
                      sn_epoch=ds.sn_epoch.values,
                      min_elapsed_day=np.min(ds.elapsed_day.values))
    return dataset


def compute_magnitude(flux, method='modified'):
    if method == 'modified':
        return compute_magnitude_modified(flux)
    else:
        return compute_magnitude_traditional(flux)


def compute_magnitude_modified(flux):
    a = 2.5 * np.log10(np.e)
    magnitude = -a * np.arcsinh(flux * 0.5)
    return magnitude


def compute_magnitude_traditional(flux):
    a = 2.5 * np.log10(np.e)
    tmp = np.where(flux < 0.1, 0.1, flux)
    magnitude = -a * np.log(tmp)
    return magnitude


def compute_noisy_magnitude(flux, flux_err, blackout_rate, outlier_rate,
                            method='modified'):
    shape = flux.shape

    noise = np.random.randn(*shape) * flux_err
    magnitude = compute_magnitude(flux + noise, method=method)

    # blackout_rateの確率でFalseを生成する
    blackout_mask = np.random.rand(*shape) > blackout_rate
    # Falseの部分を0にする
    magnitude = np.where(blackout_mask, magnitude, 0)

    # outlier_rateの確率でTrueを生成する
    outlier_mask = np.random.rand(*shape) < outlier_rate
    # Trueの部分のみノイズを加える
    outlier = np.where(outlier_mask, np.random.randn(*shape), 0)
    magnitude = magnitude + outlier

    return magnitude


class NovaTypeSequence(keras.utils.Sequence):
    def __init__(self, file_path, band_data, n_classes, batch_size,
                 use_redshift, train, method='modified', mean=None, std=None,
                 outlier_rate=None, blackout_rate=None, dropout_rate=None,
                 prediction=False):
        self.ds = load_data(file_path=file_path, band_data=band_data)
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.use_redshift = use_redshift
        self.train = train
        self.method = method
        self.outlier_rate = outlier_rate
        self.blackout_rate = blackout_rate
        self.dropout_rate = dropout_rate

        self.prediction = prediction

        if mean is None or std is None:
            magnitude = compute_magnitude(self.ds.flux, method=method)
            self.mean = np.mean(magnitude, axis=0)
            self.std = np.std(magnitude, axis=0)
        else:
            self.mean = mean
            self.std = std

    def __len__(self):
        return math.ceil(self.ds.flux.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        s = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        flux = self.ds.flux[s]
        flux_err = self.ds.flux_err[s]
        redshift = self.ds.redshift[s]
        label = self.ds.label[s]

        if self.train:
            magnitude = compute_noisy_magnitude(
                flux=flux, flux_err=flux_err,
                blackout_rate=self.blackout_rate,
                outlier_rate=self.outlier_rate, method=self.method
            )
        else:
            magnitude = compute_magnitude(flux=flux, method=self.method)
        magnitude = (magnitude - self.mean) / self.std

        if self.use_redshift:
            magnitude = np.concatenate(
                (magnitude, np.reshape(redshift, [-1, 1])), axis=1
            )

        if self.prediction:
            return magnitude

        label = tf.keras.utils.to_categorical(
            label, num_classes=self.n_classes
        )

        return magnitude, label

    def on_epoch_end(self):
        if not self.train:
            return

        flux, flux_err, redshift, label, name, sn_epoch = shuffle(
            self.ds.flux, self.ds.flux_err, self.ds.redshift, self.ds.label,
            self.ds.name, self.ds.sn_epoch
        )
        ds = Dataset(flux=flux, flux_err=flux_err, label=label,
                     redshift=redshift, name=name, sn_epoch=sn_epoch,
                     min_elapsed_day=self.ds.min_elapsed_day)
        self.ds = ds


def make_model(input_size, hidden_size, output_size, dropout_rate,
               n_highways, use_bn):
    x = keras.Input(shape=(input_size,))
    dense1 = keras.layers.Dense(units=hidden_size, name='my_dense1')(x)
    drop1 = keras.layers.Dropout(rate=dropout_rate, name='my_dropout1')(dense1)

    h = drop1
    for i in range(n_highways):
        transform = keras.layers.Dense(
            units=hidden_size, name='my_transform{}'.format(i + 1),
            activation=keras.activations.sigmoid
        )(h)
        gate = keras.layers.Dense(
            units=hidden_size, name='my_gate{}'.format(i + 1),
            activation=keras.activations.sigmoid
        )(h)
        h = keras.layers.Lambda(
            lambda v: v[0] * v[1] + (1 - v[0]) * v[2]
        )([gate, transform, h])
        if use_bn:
            h = keras.layers.BatchNormalization(name='my_bn{}'.format(i + 1))(
                h
            )

    y = keras.layers.Dense(units=output_size, name='my_dense2',
                           activation=keras.activations.softmax)(h)
    model = keras.models.Model(inputs=x, outputs=y)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def fit(input_size, output_size, epoch, batch_size, band_data, method,
        use_redshift, train_data_path, validation_data_path, dropout_rate,
        blackout_rate, outlier_rate, hidden_size, n_highways, use_bn):
    train_sequence = NovaTypeSequence(
        file_path=train_data_path, band_data=band_data, n_classes=output_size,
        batch_size=batch_size, use_redshift=use_redshift, train=True,
        method=method, outlier_rate=outlier_rate,
        blackout_rate=blackout_rate, dropout_rate=dropout_rate
    )
    validation_sequence = NovaTypeSequence(
        file_path=validation_data_path, band_data=band_data,
        n_classes=output_size, batch_size=batch_size,
        use_redshift=use_redshift, train=False,
        method=method, mean=train_sequence.mean, std=train_sequence.std
    )

    model = make_model(
        input_size=input_size, hidden_size=hidden_size,
        output_size=output_size, dropout_rate=dropout_rate,
        n_highways=n_highways, use_bn=use_bn
    )

    history = model.fit_generator(
        generator=train_sequence,
        steps_per_epoch=len(train_sequence),
        epochs=epoch,
        callbacks=[keras.callbacks.EarlyStopping(patience=100)],
        validation_data=validation_sequence,
        validation_steps=len(validation_sequence),
        verbose=1,
        shuffle=True
    )
    return model, history, train_sequence.mean, train_sequence.std


def objective(args, input_size, output_size, epoch, batch_size, band_data,
              method, use_redshift, train_data_path, validation_data_path,
              test_data_path):
    dropout_rate = args['dropout_rate']
    blackout_rate = args['blackout_rate']
    outlier_rate = args['outlier_rate']
    hidden_size = int(args['hidden_size'])
    n_highways = int(args['n_highways'])
    use_bn = args['use_bn']

    model, history, mean, std = fit(
        input_size=input_size, output_size=output_size, epoch=epoch,
        batch_size=batch_size, band_data=band_data, method=method,
        use_redshift=use_redshift, train_data_path=train_data_path,
        validation_data_path=validation_data_path, dropout_rate=dropout_rate,
        blackout_rate=blackout_rate, outlier_rate=outlier_rate,
        hidden_size=hidden_size, n_highways=n_highways, use_bn=use_bn
    )

    test_sequence = NovaTypeSequence(
        file_path=test_data_path, band_data=band_data, n_classes=output_size,
        batch_size=batch_size, use_redshift=use_redshift, train=False,
        method=method, mean=mean, std=std
    )

    tmp = model.evaluate_generator(generator=test_sequence,
                                   steps=len(test_sequence))
    metrics = {key: float(value)
               for key, value in zip(model.metrics_names, tmp)}

    cost = -float(history.history['val_categorical_accuracy'][0])

    return {'loss': cost, 'status': STATUS_OK, 'metrics': metrics}


def run_optimization(input_size, output_size, epoch, batch_size, band_data,
                     method, use_redshift, train_data_path,
                     validation_data_path, test_data_path,
                     dropout_rate, blackout_rate, outlier_rate, hidden_size,
                     n_highways, use_bn):
    model, history, mean, std = fit(
        input_size=input_size, output_size=output_size, epoch=epoch,
        batch_size=batch_size, band_data=band_data, method=method,
        use_redshift=use_redshift, train_data_path=train_data_path,
        validation_data_path=validation_data_path, dropout_rate=dropout_rate,
        blackout_rate=blackout_rate, outlier_rate=outlier_rate,
        hidden_size=hidden_size, n_highways=n_highways, use_bn=use_bn
    )

    metrics = {}
    tmp = (('train', train_data_path), ('validation', validation_data_path),
           ('test', test_data_path))
    for tag, path in tmp:
        data = NovaTypeSequence(
            file_path=path, band_data=band_data, n_classes=output_size,
            batch_size=batch_size, use_redshift=use_redshift, train=False,
            method=method, mean=mean, std=std
        )
        result = model.evaluate_generator(generator=data, steps=len(data))
        # データの要素がnumpyの型で、jsonで扱えないので、変換する
        result = {key: float(value)
                  for key, value in zip(model.metrics_names, result)}
        metrics[tag] = result

    return model, mean, std, metrics


def get_search_space():
    space = {
        'hidden_size': scope.int(hp.qloguniform(
            'hidden_size', np.log(50), np.log(1000), 50
        )),
        'dropout_rate': hp.quniform('dropout_rate', 0, 0.8, 0.05),
        'outlier_rate': hp.uniform('outlier_rate', 0, 0.1),
        'blackout_rate': hp.uniform('blackout_rate', 0, 0.1),
        'n_highways': scope.int(hp.quniform('n_highways', 2, 10, 2)),
        'use_bn': hp.choice('use_bn', [False, True])
    }
    return space


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--output_size', type=int, default=6)
@click.option('--epoch', type=int, default=5)
@click.option('--batch_size', type=int, default=100)
@click.option('--band_data', type=str)
@click.option('--method', type=click.Choice(['modified', 'traditional']),
              default='modified')
@click.option('--use_redshift', is_flag=True)
@click.option('--train_data_path',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--validation_data_path',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--test_data_path',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--output_dir', type=click.Path(dir_okay=True))
@click.option('--n_iterations', type=int, default=100)
def search(output_size, epoch, batch_size, band_data, method, use_redshift,
           train_data_path, validation_data_path, test_data_path, output_dir,
           n_iterations):
    band_data = parse_band_data(band_data=band_data)
    input_size = get_input_size(band_data=band_data, use_redshift=use_redshift)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f_objective = partial(
        objective, input_size=input_size, output_size=output_size, epoch=epoch,
        batch_size=batch_size, band_data=band_data, method=method,
        use_redshift=use_redshift, train_data_path=train_data_path,
        validation_data_path=validation_data_path,
        test_data_path=test_data_path
    )

    space = get_search_space()

    # 履歴の読み込み
    trial_file = os.path.join(output_dir, 'trials.pickle')
    if os.path.exists(trial_file):
        trials = joblib.load(trial_file)  # type: Trials
    else:
        trials = Trials()

    step = 10
    for i in range(len(trials.trials), n_iterations, step):
        n = min(i + step, n_iterations)
        best = fmin(f_objective, space=space, algo=tpe.suggest,
                    max_evals=n, trials=trials)

        # 履歴の保存
        joblib.dump(trials, trial_file)

        print('best score: ', best)
        print(trials.best_trial)
        with open(os.path.join(output_dir, 'best_parameter.json'),
                  'w') as f:
            json.dump(trials.best_trial, f, sort_keys=True, indent=4,
                      default=serialize_object)


@cmd.command()
@click.option('--output_size', type=int, default=6)
@click.option('--epoch', type=int, default=5)
@click.option('--batch_size', type=int, default=100)
@click.option('--band_data', type=str)
@click.option('--method', type=click.Choice(['modified', 'traditional']),
              default='modified')
@click.option('--use_redshift', is_flag=True)
@click.option('--train_data_path',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--validation_data_path',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--test_data_path',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--output_dir', type=click.Path(dir_okay=True))
@click.option('--parameter_path', type=click.Path(exists=True, dir_okay=False),
              default=None)
@click.option('--hidden_size', type=int, default=None)
@click.option('--dropout_rate', type=float, default=None)
@click.option('--outlier_rate', type=float, default=None)
@click.option('--blackout_rate', type=float, default=None)
@click.option('--n_highways', type=int, default=None)
@click.option('--use_bn/--unuse_bn', is_flag=True, default=False)
def optimize(output_size, epoch, batch_size, band_data, method, use_redshift,
             train_data_path, validation_data_path, test_data_path,
             output_dir, parameter_path, hidden_size, dropout_rate,
             blackout_rate, outlier_rate, n_highways, use_bn):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    band_data = parse_band_data(band_data=band_data)
    input_size = get_input_size(band_data=band_data, use_redshift=use_redshift)

    search_parameters = SearchParameters(
        hidden_size=hidden_size, dropout_rate=dropout_rate,
        blackout_rate=blackout_rate, outlier_rate=outlier_rate,
        n_highways=n_highways, use_bn=use_bn
    )
    search_parameters.update(parameter_path=parameter_path)

    model, mean, std, metrics = run_optimization(
        input_size=input_size, output_size=output_size, epoch=epoch,
        batch_size=batch_size, band_data=band_data, method=method,
        use_redshift=use_redshift, train_data_path=train_data_path,
        validation_data_path=validation_data_path,
        test_data_path=test_data_path,
        dropout_rate=search_parameters.dropout_rate,
        blackout_rate=search_parameters.blackout_rate,
        outlier_rate=search_parameters.outlier_rate,
        hidden_size=search_parameters.hidden_size,
        n_highways=search_parameters.n_highways,
        use_bn=search_parameters.use_bn
    )

    keras.models.save_model(model, os.path.join(output_dir, 'model.h5'))

    summary = {
        'output_size': output_size, 'band_data': band_data, 'method': method,
        'use_redshift': use_redshift, 'train_data_path': train_data_path,
        'validation_data_path': validation_data_path,
        'test_data_path': test_data_path,
        'hidden_size': hidden_size, 'dropout_rate': dropout_rate,
        'outlier_rate': outlier_rate, 'blackout_rate': blackout_rate,
        'n_highways': n_highways, 'use_bn': use_bn,
        'mean': mean.to_list(), 'std': std.to_list(), 'metrics': metrics
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4, sort_keys=True)


class SearchParameters(object):
    """
    ハイパーパラメータの値を設定するためのクラス

    実行時にオプションを指定する方法と
    探索結果のファイル(best_parameter.json)を読み込む方法がある
    両方を設定した場合は、実行時オプションが優先
    """
    def __init__(self, hidden_size, dropout_rate, blackout_rate, outlier_rate,
                 n_highways, use_bn):
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.blackout_rate = blackout_rate
        self.outlier_rate = outlier_rate
        self.n_highways = n_highways
        self.use_bn = use_bn

    def update(self, parameter_path):
        if parameter_path is None:
            return

        parameter = self._load_from_file(path=parameter_path)

        # 個別のオプションの値が優先
        self.hidden_size = self.hidden_size or parameter['hidden_size']
        self.dropout_rate = self.dropout_rate or parameter['dropout_rate']
        self.blackout_rate = self.blackout_rate or parameter['blackout_rate']
        self.outlier_rate = self.outlier_rate or parameter['outlier_rate']
        self.n_highways = self.n_highways or parameter['n_highways']
        self.use_bn = self.use_bn or parameter['use_bn']

    @staticmethod
    def _load_from_file(path):
        with open(path, 'r') as f:
            parameters = json.load(f)
        vals = parameters['misc']['vals']
        vals = {key: value[0] for key, value in vals.items()}

        space = get_search_space()
        parameter = space_eval(space=space, hp_assignment=vals)

        return parameter


def parse_band_data(band_data):
    # band_data は {"Y": 10, "g": 8} みたいな文字列が与えられる
    band_data = eval(band_data)
    assert isinstance(band_data, dict), "band_data is an unexpected value"
    return band_data


def get_input_size(band_data, use_redshift):
    input_size = sum(band_data.values())
    if use_redshift:
        input_size += 1
    return input_size


def serialize_object(o):
    if isinstance(o, datetime):
        return o.isoformat()
    elif isinstance(o, bson.ObjectId):
        return str(o)
    raise TypeError(repr(o) + ' is not JSON serializable')


def main():
    cmd()


if __name__ == '__main__':
    main()
