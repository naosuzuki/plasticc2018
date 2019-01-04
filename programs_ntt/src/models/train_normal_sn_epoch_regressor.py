#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import json
import math
import os
from functools import partial

import click
import numpy as np

try:
    from tensorflow import keras
except ImportError:
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.keras.python import keras
from hyperopt import fmin, tpe, STATUS_OK, Trials
from sklearn.externals import joblib
# noinspection PyProtectedMember
from sklearn.utils import shuffle

from train_normal_classifier_model import (
    Dataset, load_data, compute_magnitude, compute_noisy_magnitude,
    serialize_object, parse_band_data, get_input_size, SearchParameters
)
from train_normal_redshift_regressor import make_model, get_search_space

__author__ = 'Yasuhiro Imoto'
__date__ = '15/2/2018'


class SnEpochSequence(keras.utils.Sequence):
    def __init__(self, file_path, band_data, batch_size,
                 use_redshift, train, method='modified', mean=None, std=None,
                 outlier_rate=None, blackout_rate=None, dropout_rate=None,
                 prediction=False):
        self.ds = load_data(file_path=file_path, band_data=band_data)
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
        # 木村様の時と同じで、明るさのピーク日付と観測の基準日との日数差
        sn_epoch = (np.min(self.ds.sn_epoch[s], axis=1) -
                    self.ds.min_elapsed_day)

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

        # 正規化する
        #   標準偏差の値は木村様の時と同じ値のまま
        #   観測データに依存して決定されているようだが、算出方法が分からない
        # 配列をN×1にする
        sn_epoch = np.reshape(sn_epoch / 30.0, [-1, 1])

        return magnitude, sn_epoch

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


def fit(input_size, epoch, batch_size, band_data, method,
        use_redshift, train_data_path, validation_data_path, dropout_rate,
        blackout_rate, outlier_rate, hidden_size, n_highways, use_bn):
    train_sequence = SnEpochSequence(
        file_path=train_data_path, band_data=band_data,
        batch_size=batch_size, use_redshift=use_redshift, train=True,
        method=method, outlier_rate=outlier_rate,
        blackout_rate=blackout_rate, dropout_rate=dropout_rate
    )
    validation_sequence = SnEpochSequence(
        file_path=validation_data_path, band_data=band_data,
        batch_size=batch_size, use_redshift=use_redshift, train=False,
        method=method, mean=train_sequence.mean, std=train_sequence.std
    )

    model = make_model(input_size=input_size, hidden_size=hidden_size,
                       dropout_rate=dropout_rate, n_highways=n_highways,
                       use_bn=use_bn)

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


def objective(args, input_size, epoch, batch_size, band_data,
              method, use_redshift, train_data_path, validation_data_path,
              test_data_path):
    dropout_rate = args['dropout_rate']
    blackout_rate = args['blackout_rate']
    outlier_rate = args['outlier_rate']
    hidden_size = int(args['hidden_size'])
    n_highways = int(args['n_highways'])
    use_bn = args['use_bn']

    model, history, mean, std = fit(
        input_size=input_size, epoch=epoch,
        batch_size=batch_size, band_data=band_data, method=method,
        use_redshift=use_redshift, train_data_path=train_data_path,
        validation_data_path=validation_data_path, dropout_rate=dropout_rate,
        blackout_rate=blackout_rate, outlier_rate=outlier_rate,
        hidden_size=hidden_size, n_highways=n_highways, use_bn=use_bn
    )

    test_sequence = SnEpochSequence(
        file_path=test_data_path, band_data=band_data,
        batch_size=batch_size, use_redshift=use_redshift, train=False,
        method=method, mean=mean, std=std
    )

    tmp = model.evaluate_generator(generator=test_sequence,
                                   steps=len(test_sequence))
    metrics = {key: float(value)
               for key, value in zip(model.metrics_names, tmp)}

    cost = -float(history.history['val_r2_score'][0])

    return {'loss': cost, 'status': STATUS_OK, 'metrics': metrics}


def run_optimization(input_size, epoch, batch_size, band_data,
                     method, use_redshift, train_data_path,
                     validation_data_path, test_data_path,
                     dropout_rate, blackout_rate, outlier_rate, hidden_size,
                     n_highways, use_bn):
    model, history, mean, std = fit(
        input_size=input_size, epoch=epoch,
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
        data = SnEpochSequence(
            file_path=path, band_data=band_data,
            batch_size=batch_size, use_redshift=use_redshift, train=False,
            method=method, mean=mean, std=std
        )
        result = model.evaluate_generator(generator=data, steps=len(data))
        # データの要素がnumpyの型で、jsonで扱えないので、変換する
        result = {key: float(value)
                  for key, value in zip(model.metrics_names, result)}
        metrics[tag] = result

    return model, mean, std, metrics


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
             output_dir, parameter_path,
             hidden_size, dropout_rate, blackout_rate, outlier_rate,
             n_highways, use_bn):
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
        input_size=input_size, epoch=epoch,
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


def main():
    cmd()


if __name__ == '__main__':
    main()
