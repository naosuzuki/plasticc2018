#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import os
import json
import functools

import numpy as np
import pandas as pd
import click
from sklearn.model_selection import train_test_split

from pauc import train_pauc_relaxed, train_pauc_exact
from random_forest import train_random_forest

__author__ = 'Yasuhiro Imoto'
__date__ = '14/12/2017'


@click.group()
def train_model():
    pass


def common_train_parameters(func):
    @click.option('--data-path', type=click.Path(exists=True))
    @click.option('--output-dir', type=click.Path(file_okay=False))
    @click.option('--seed', type=int, default=0)
    @click.option('--split-ratio', type=float, nargs=3,
                  default=(0.8, 0.1, 0.1))
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@train_model.group()
def pauc():
    pass

# ----------
# pauc の種類
# ----------


def validate_epoch_range(ctx, param, value):
    # --max-epochの内部名はepoch
    if 0 < value < ctx.params['epoch']:
        return value
    else:
        raise click.BadParameter('min-epoch must be smaller than max-epoch.')


def common_pauc_parameters(func):
    # ソースコード内部の処理との変数名の一貫性のために内部名を設定する
    @click.option('--max-epoch', 'epoch', type=int, default=1000)
    @click.option('--min-epoch', 'patience', type=int, default=100)
    # なぜかコマンドプロンプトの方ではepochを取得できないので
    # チェックできない場合があり、廃止
    # @click.option('--min-epoch', 'patience', type=int, default=100,
    #               callback=validate_epoch_range)
    @click.option('--resume', is_flag=True)
    @click.option('--batch-size', type=int, default=1000)
    @click.option('--validation-frequency', type=int, default=50)
    @click.option('--increasing-ratio', 'patience_increase', type=float,
                  default=2.0)
    @click.option('--improvement-threshold', type=float, default=1.0)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def pauc_relaxed_parameters(func):
    @click.option('--gamma', type=float, default=0.1)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def pauc_exact_parameters(func):
    @click.option('--beta', type=float, default=0.1)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@pauc.group()
def relaxed():
    pass


@pauc.group()
def exact():
    pass

# ----------
# DNN
# ----------


def dnn_parameters(func):
    @click.option('--hidden-size', type=int, default=100)
    @click.option('--regularizer-weights', nargs=2, type=float,
                  default=(0.0, 0.0))
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@relaxed.command(name='dnn')
@common_train_parameters
@common_pauc_parameters
@pauc_relaxed_parameters
@dnn_parameters
def relaxed_dnn(*args, **kwargs):
    train_pauc_model(model_type='dnn', mode='relaxed', *args, **kwargs)


@exact.command(name='dnn')
@common_train_parameters
@common_pauc_parameters
@pauc_exact_parameters
@dnn_parameters
def exact_dnn(*args, **kwargs):
    train_pauc_model(model_type='dnn', mode='exact', *args, **kwargs)

# ----------
# GMM
# ----------


def gmm_parameters(func):
    @click.option('--positive-mixture', type=int, default=1)
    @click.option('--negative-mixture', type=int, default=1)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@relaxed.command(name='gmm')
@common_train_parameters
@common_pauc_parameters
@pauc_relaxed_parameters
@gmm_parameters
@click.argument('variance_type', type=click.Choice(['full', 'diagonal']))
def relaxed_gmm(variance_type, *args, **kwargs):
    train_pauc_model(model_type='gmm-{}'.format(variance_type), mode='relaxed',
                     *args, **kwargs)


@exact.command(name='gmm')
@common_train_parameters
@common_pauc_parameters
@pauc_exact_parameters
@gmm_parameters
@click.argument('variance_type', type=click.Choice(['full', 'diagonal']))
def exact_gmm(variance_type, *args, **kwargs):
    train_pauc_model(model_type='gmm-{}'.format(variance_type), mode='exact',
                     *args, **kwargs)


def convert_data(data_path):
    ds = pd.read_pickle(data_path)
    ds = ds.fillna(10)
    # 変換
    tmp = 1.0857 / ds.loc[:, 'magerr'].values
    ds.loc[:, 'magerr'] = tmp

    # 必要な列を抜き出す
    # 順序の固定を行う
    columns = ['magerr', 'elongation.norm', 'fwhm.norm',
               'significance.abs', 'residual', 'psffit.sigma.ratio',
               'psffit.peak.ratio', 'frac.det', 'density',
               'density.good', 'baPsf', 'sigmaPsf']
    data = ds[columns].values

    label = ds['real/bogus']

    return data, label


def split_data(ds, label, seed, split_ratio):
    train_data, tmp_data, train_label, tmp_label = train_test_split(
        ds, label, test_size=np.sum(split_ratio[1:]) / np.sum(split_ratio),
        stratify=label, random_state=seed
    )

    tmp = train_test_split(
        tmp_data, tmp_label,
        test_size=split_ratio[2] / np.sum(split_ratio[1:]),
        stratify=tmp_label, random_state=seed + 1
    )
    validation_data, test_data, validation_label, test_label = tmp

    dataset = {'train': {'x': train_data, 'y': train_label},
               'validation': {'x': validation_data, 'y': validation_label},
               'test': {'x': test_data, 'y': test_label}}
    return dataset


def train_pauc_model(model_type, data_path, epoch, resume, output_dir,
                     validation_frequency, seed, split_ratio, mode,
                     batch_size, patience, patience_increase,
                     improvement_threshold, **kwargs):
    """
    pauc_relaxedとpauc_exactで処理の大半が共通しているので、
    それらを関数に集約する

    :param model_type:
    :param data_path:
    :param epoch:
    :param resume:
    :param output_dir:
    :param validation_frequency:
    :param seed:
    :param split_ratio:
    :param mode:
    :param batch_size:
    :param patience:
    :param patience_increase:
    :param improvement_threshold:
    :param kwargs:
    :return:
    """
    data, label = convert_data(data_path)

    dataset = split_data(data, label, seed, split_ratio)
    train_data = dataset['train']['x']
    validation_data = dataset['validation']['x']

    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    if mode == 'relaxed':
        parameter_name = 'gamma'
    else:
        parameter_name = 'beta'
    parameters = {
        'data_path': data_path, 'model_type': model_type,
        parameter_name: kwargs[parameter_name], 'batch_size': batch_size,
        'seed': seed, 'split_ratio': split_ratio,
        'mean': mean.tolist(), 'std': std.tolist()
    }
    if model_type == 'dnn':
        parameters['hidden_size'] = kwargs['hidden_size']
        parameters['regularizer_weights'] = kwargs['regularizer_weights']
    else:
        parameters['positive_components'] = kwargs['positive_mixture']
        parameters['negative_components'] = kwargs['negative_mixture']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        json.dump(parameters, f, sort_keys=True)

    train = {'positive': train_data[dataset['train']['y'] != 0],
             'negative': train_data[dataset['train']['y'] == 0]}
    train = {key: (value - mean) / std for key, value in train.items()}

    validation_label = dataset['validation']['y']
    validation = {'positive': validation_data[validation_label != 0],
                  'negative': validation_data[validation_label == 0]}
    validation = {key: (value - mean) / std
                  for key, value in validation.items()}

    d = dict(epoch=epoch, resume=resume, output_dir=output_dir,
             model_type=model_type, validation_frequency=validation_frequency,
             batch_size=batch_size, patience=patience,
             patience_increase=patience_increase,
             improvement_threshold=improvement_threshold)
    d.update({parameter_name: kwargs[parameter_name]})
    if model_type == 'dnn':
        d['hidden_size'] = kwargs['hidden_size']
        d['lambda_constraints'] = kwargs['regularizer_weights']
    else:
        d['positive_components'] = kwargs['positive_mixture']
        d['negative_components'] = kwargs['negative_mixture']
        d['lambda_constraints'] = None
    if mode == 'relaxed':
        train_pauc_relaxed(train_data=train, validation_data=validation, **d)
    else:
        train_pauc_exact(train_data=train, validation_data=validation, **d)


def train_rf_model(data_path, resume, output_dir, n_estimators, max_depth,
                   n_jobs, seed, split_ratio, mode, **kwargs):
    data, label = convert_data(data_path)

    train_data, tmp_data, train_label, tmp_label = train_test_split(
        data, label, test_size=np.sum(split_ratio[1:]) / np.sum(split_ratio),
        stratify=label, random_state=seed
    )

    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    parameters = {
        'data_path': data_path, 'n_estimators': n_estimators,
        'n_jobs': n_jobs, 'seed': seed,
        'split_ratio': split_ratio, 'mean': mean.tolist(), 'std': std.tolist()
    }
    if mode == 'random_forest':
        parameters['criterion'] = kwargs['criterion']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        json.dump(parameters, f, sort_keys=True)

    train_random_forest({'x': train_data, 'y': train_label},
                        n_estimators=n_estimators, max_depth=max_depth,
                        n_jobs=n_jobs, resume=resume, output_dir=output_dir,
                        mode=mode, random_state=seed, **kwargs)


def validate_patience_range(ctx, param, value):
    if 0 < value < ctx.params['epoch']:
        return value
    else:
        raise click.BadParameter('patience should be smaller than epoch.')


@train_model.command()
@click.option('--model-type', default='dnn',
              type=click.Choice(['dnn', 'gmm-full', 'gmm-diagonal']))
@click.option('--model_type', default='dnn',
              type=click.Choice(['dnn', 'gmm-full', 'gmm-diagonal']))
@click.option('--data-path', type=str)
@click.option('--data_path', type=str)
@click.option('--epoch', type=int, default=1000)
@click.option('--resume', is_flag=True)
@click.option('--output-dir', type=str)
@click.option('--output_dir', type=str)
@click.option('--hidden-size', type=int, default=100)
@click.option('--hidden_size', type=int, default=100)
@click.option('--positive-mixture', type=int, default=1)
@click.option('--positive_mixture', type=int, default=1)
@click.option('--negative-mixture', type=int, default=1)
@click.option('--negative_mixture', type=int, default=1)
@click.option('--batch-size', type=int, default=1000)
@click.option('--batch_size', type=int, default=1000)
@click.option('--regularizer-weights', nargs=2, type=float, default=(0.0, 0.0))
@click.option('--regularizer_weights', nargs=2, type=float, default=(0.0, 0.0))
@click.option('--validation-frequency', type=int, default=50)
@click.option('--validation_frequency', type=int, default=50)
@click.option('--gamma', type=float, default=0.1)
@click.option('--seed', type=int, default=0)
@click.option('--split-ratio', type=float, nargs=3, default=(0.8, 0.1, 0.1))
@click.option('--split_ratio', type=float, nargs=3, default=(0.8, 0.1, 0.1))
@click.option('--min-epoch', 'patience', type=int, default=500)
@click.option('--min_epoch', 'patience', type=int, default=500)
@click.option('--increasing-ratio', 'patience_increase', type=float,
              default=2.0)
@click.option('--increasing_ratio', 'patience_increase', type=float,
              default=2.0)
@click.option('--improvement-threshold', type=float, default=1.0)
@click.option('--improvement_threshold', type=float, default=1.0)
def pauc_relaxed(model_type, data_path, epoch, resume, output_dir, hidden_size,
                 batch_size, positive_mixture, negative_mixture,
                 regularizer_weights, validation_frequency, gamma,
                 seed, split_ratio, patience, patience_increase,
                 improvement_threshold):
    train_pauc_model(model_type=model_type, data_path=data_path, epoch=epoch,
                     resume=resume,
                     output_dir=output_dir, hidden_size=hidden_size,
                     positive_mixture=positive_mixture,
                     negative_mixture=negative_mixture,
                     regularizer_weights=regularizer_weights,
                     validation_frequency=validation_frequency, seed=seed,
                     split_ratio=split_ratio, mode='relaxed', gamma=gamma,
                     batch_size=batch_size, patience=patience,
                     patience_increase=patience_increase,
                     improvement_threshold=improvement_threshold)


@train_model.command()
@click.option('--model-type', default='dnn',
              type=click.Choice(['dnn', 'gmm-full', 'gmm-diagonal']))
@click.option('--model_type', default='dnn',
              type=click.Choice(['dnn', 'gmm-full', 'gmm-diagonal']))
@click.option('--data-path', type=str)
@click.option('--data_path', type=str)
@click.option('--epoch', type=int, default=1000)
@click.option('--resume', is_flag=True)
@click.option('--output-dir', type=str)
@click.option('--output_dir', type=str)
@click.option('--hidden-size', type=int, default=100)
@click.option('--hidden_size', type=int, default=100)
@click.option('--positive-mixture', type=int, default=1)
@click.option('--positive_mixture', type=int, default=1)
@click.option('--negative-mixture', type=int, default=1)
@click.option('--negative_mixture', type=int, default=1)
@click.option('--batch-size', type=int, default=1000)
@click.option('--batch_size', type=int, default=1000)
@click.option('--regularizer-weights', nargs=2, type=float, default=(0.0, 0.0))
@click.option('--regularizer_weights', nargs=2, type=float, default=(0.0, 0.0))
@click.option('--validation-frequency', type=int, default=50)
@click.option('--validation_frequency', type=int, default=50)
@click.option('--beta', type=float, default=0.1)
@click.option('--seed', type=int, default=0)
@click.option('--split-ratio', type=float, nargs=3, default=(0.8, 0.1, 0.1))
@click.option('--split_ratio', type=float, nargs=3, default=(0.8, 0.1, 0.1))
@click.option('--min-epoch', 'patience', type=int, default=500)
@click.option('--min_epoch', 'patience', type=int, default=500)
@click.option('--increasing-ratio', 'patience_increase', type=float,
              default=2.0)
@click.option('--increasing_ratio', 'patience_increase', type=float,
              default=2.0)
@click.option('--improvement-threshold', type=float, default=1.0)
@click.option('--improvement_threshold', type=float, default=1.0)
def pauc_exact(model_type, data_path, epoch, resume, output_dir, hidden_size,
               batch_size, positive_mixture, negative_mixture,
               regularizer_weights, validation_frequency, beta,
               seed, split_ratio, patience, patience_increase,
               improvement_threshold):
    train_pauc_model(model_type=model_type, data_path=data_path, epoch=epoch,
                     resume=resume,
                     output_dir=output_dir, hidden_size=hidden_size,
                     positive_mixture=positive_mixture,
                     negative_mixture=negative_mixture,
                     regularizer_weights=regularizer_weights,
                     validation_frequency=validation_frequency, seed=seed,
                     split_ratio=split_ratio, mode='exact', beta=beta,
                     batch_size=batch_size, patience=patience,
                     patience_increase=patience_increase,
                     improvement_threshold=improvement_threshold)


@train_model.command()
@click.option('--data_path', type=str)
@click.option('--resume', is_flag=True)
@click.option('--output_dir', type=str)
@click.option('--n_estimators', type=int, default=100)
@click.option('--max_depth', type=int, default=None)
@click.option('--criterion', type=click.Choice(['gini', 'entropy']),
              default='gini')
@click.option('--n_jobs', type=int, default=1)
@click.option('--seed', type=int, default=0)
@click.option('--split_ratio', type=float, nargs=3, default=(0.8, 0.1, 0.1))
def random_forest(data_path, resume, output_dir, n_estimators, max_depth,
                  criterion, n_jobs, seed, split_ratio):
    train_rf_model(data_path=data_path, resume=resume, output_dir=output_dir,
                   n_estimators=n_estimators, max_depth=max_depth,
                   n_jobs=n_jobs, seed=seed, split_ratio=split_ratio,
                   mode='random_forest', criterion=criterion)


@train_model.command()
@click.option('--data_path', type=str)
@click.option('--resume', is_flag=True)
@click.option('--output_dir', type=str)
@click.option('--n_estimators', type=int, default=100)
@click.option('--max_depth', type=int, default=None)
@click.option('--n_jobs', type=int, default=1)
@click.option('--seed', type=int, default=0)
@click.option('--split_ratio', type=float, nargs=3, default=(0.8, 0.1, 0.1))
def xgboost(data_path, resume, output_dir, n_estimators, max_depth,
            n_jobs, seed, split_ratio):
    train_rf_model(data_path=data_path, resume=resume, output_dir=output_dir,
                   n_estimators=n_estimators, max_depth=max_depth,
                   n_jobs=n_jobs, seed=seed, split_ratio=split_ratio,
                   mode='xgboost')


def main():
    train_model()


if __name__ == '__main__':
    main()
