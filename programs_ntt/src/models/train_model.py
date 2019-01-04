#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import json
import os
import re
from functools import partial, wraps

import click
from hyperopt import fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials
from sklearn.externals import joblib

try:
    from tensorflow import keras
except ImportError:
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.keras.python import keras

import train_normal_classifier_model
import train_normal_redshift_regressor
import train_normal_sn_epoch_regressor
from train_normal_classifier_model import serialize_object, SearchParameters

__author__ = 'Yasuhiro Imoto'
__date__ = '25/8/2017'


@click.group()
def cmd():
    pass


# ---------------------------------------------------------------------------
# sub task 1
# ---------------------------------------------------------------------------

@cmd.group()
def sn_class():
    pass


@cmd.group()
def redshift():
    pass


@cmd.group()
def sn_epoch():
    pass


# ---------------------------------------------------------------------------
# parameter
# ---------------------------------------------------------------------------

def common_parameter(func):
    @click.option('--epoch', type=int, default=5)
    @click.option('--batch_size', type=int, default=100)
    @click.option('--band_data', type=str)
    @click.option('--method', type=click.Choice(['modified', 'traditional']),
                  default='modified')
    @click.option('--train_data_path',
                  type=click.Path(exists=True, dir_okay=False))
    @click.option('--validation_data_path',
                  type=click.Path(exists=True, dir_okay=False))
    @click.option('--test_data_path',
                  type=click.Path(exists=True, dir_okay=False))
    @click.option('--output_dir', type=click.Path(dir_okay=True))
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def use_redshift_parameter(func):
    @click.option('--use_redshift', is_flag=True)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def sn_class_parameter(func):
    @click.option('--output_size', type=int, default=None)
    @use_redshift_parameter
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def redshift_parameter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(output_size=1, use_redshift=False, *args, **kwargs)
    return wrapper


def sn_epoch_parameter(func):
    @use_redshift_parameter
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(output_size=1, *args, **kwargs)
    return wrapper


def search_parameter(func):
    @click.option('--n_iterations', type=int, default=100)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def search_parallel_parameter(func):
    @search_parameter
    @click.option('--hostname', type=str, default='localhost')
    @click.option('--port', type=int, default=1234)
    @click.option('--db_name', type=str)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def optimization_parameter(func):
    @click.option('--parameter_path',
                  type=click.Path(exists=True, dir_okay=False),
                  default=None)
    @click.option('--hidden_size', type=int, default=None)
    @click.option('--dropout_rate', type=float, default=None)
    @click.option('--outlier_rate', type=float, default=None)
    @click.option('--blackout_rate', type=float, default=None)
    @click.option('--n_highways', type=int, default=None)
    @click.option('--use_bn/--unuse_bn', is_flag=True, default=False)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# sub task 2 (sn_class)
# ---------------------------------------------------------------------------

@sn_class.command(name='search_parallel')
@common_parameter
@search_parallel_parameter
@sn_class_parameter
def sn_class_search_parallel(*args, **kwargs):
    search_parallel(task_type='sn_class', *args, **kwargs)


@sn_class.command(name='search')
@common_parameter
@search_parameter
@sn_class_parameter
def sn_class_search(*args, **kwargs):
    search(task_type='sn_class', *args, **kwargs)


@sn_class.command(name='optimize')
@common_parameter
@optimization_parameter
@sn_class_parameter
def sn_class_optimize(*args, **kwargs):
    optimize(task_type='sn_class', *args, **kwargs)


# ---------------------------------------------------------------------------
# sub task 2 (redshift)
# ---------------------------------------------------------------------------

@redshift.command(name='search_parallel')
@common_parameter
@search_parallel_parameter
@redshift_parameter
def redshift_search_parallel(*args, **kwargs):
    search_parallel(task_type='redshift', *args, **kwargs)


@redshift.command(name='search')
@common_parameter
@search_parameter
@redshift_parameter
def redshift_search(*args, **kwargs):
    search(task_type='redshift', *args, **kwargs)


@redshift.command(name='optimize')
@common_parameter
@optimization_parameter
@redshift_parameter
def redshift_optimize(*args, **kwargs):
    optimize(task_type='redshift', *args, **kwargs)


# ---------------------------------------------------------------------------
# sub task 2 (sn_epoch)
# ---------------------------------------------------------------------------

@sn_epoch.command(name='search_parallel')
@common_parameter
@search_parallel_parameter
@sn_epoch_parameter
def sn_epoch_search_parallel(*args, **kwargs):
    search_parallel(task_type='sn_epoch', *args, **kwargs)


@sn_epoch.command(name='search')
@common_parameter
@search_parameter
@sn_epoch_parameter
def sn_epoch_search(*args, **kwargs):
    search(task_type='sn_epoch', *args, **kwargs)


@sn_epoch.command(name='optimize')
@common_parameter
@optimization_parameter
@sn_epoch_parameter
def sn_epoch_optimize(*args, **kwargs):
    optimize(task_type='sn_epoch', *args, **kwargs)


def get_objective_function(task_type, output_size, epoch, batch_size,
                           band_data, method, use_redshift, train_data_path,
                           validation_data_path, test_data_path):
    band_data = parse_band_data(band_data=band_data)
    input_size = get_input_size(band_data=band_data, use_redshift=use_redshift)

    if task_type == 'sn_class':
        f = train_normal_classifier_model.objective
    elif task_type == 'redshift':
        f = train_normal_redshift_regressor.objective
    elif task_type == 'sn_epoch':
        f = train_normal_sn_epoch_regressor.objective
    else:
        raise ValueError(task_type)

    d = set_parameter(
        task_type=task_type, input_size=input_size, epoch=epoch,
        batch_size=batch_size, band_data=band_data, method=method,
        train_data_path=train_data_path,
        validation_data_path=validation_data_path,
        test_data_path=test_data_path, output_size=output_size,
        use_redshift=use_redshift
    )

    f_objective = partial(f, **d)
    return f_objective


def get_search_space(task_type):
    if task_type == 'sn_class':
        space = train_normal_classifier_model.get_search_space()
    elif task_type in ('redshift', 'sn_epoch'):
        space = train_normal_redshift_regressor.get_search_space()
    else:
        raise ValueError(task_type)

    return space


def search_parallel(task_type, output_size, epoch, batch_size, band_data,
                    method, use_redshift, train_data_path,
                    validation_data_path, test_data_path, output_dir,
                    n_iterations, hostname, port, db_name):
    """
    オプションの例(sn_classの場合)

    ROOT_DIR=/data/antares/imoto/crest_auto
    DATA_DIR=$ROOT_DIR/data/processed/180420/dataset_selected/train
    OUT_DIR=$ROOT_DIR/models/tmp180706

    sn_class
    search_parallel
    --epoch=3
    --band_data={\"r\":1,\"i\":6,\"z\":7}
    --method=modified
    --train_data_path=$DATA_DIR/dataset.tr-2classes.nc
    --validation_data_path=$DATA_DIR/dataset.va-2classes.nc
    --test_data_path=$DATA_DIR/dataset.te-2classes.nc
    --output_dir=$OUT_DIR/sn_class
    --n_iterations=10
    --hostname=localhost
    --port=1234
    --db_name=ex20180706

    :param task_type:
    :param output_size:
    :param epoch:
    :param batch_size:
    :param band_data:
    :param method:
    :param use_redshift:
    :param train_data_path:
    :param validation_data_path:
    :param test_data_path:
    :param output_dir:
    :param n_iterations:
    :param hostname:
    :param port:
    :param db_name:
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f_objective = get_objective_function(
        task_type=task_type, output_size=output_size, epoch=epoch,
        batch_size=batch_size, band_data=band_data, method=method,
        use_redshift=use_redshift, train_data_path=train_data_path,
        validation_data_path=validation_data_path,
        test_data_path=test_data_path
    )

    space = get_search_space(task_type=task_type)

    r = re.compile(r'(dataset_(?:all|selected))')
    m = r.search(train_data_path)
    if m is None:
        raise ValueError('data path does not contain dataset_all or '
                         'dataset_selected: {}'.format(train_data_path))

    if output_size is None:
        output_size = get_output_size(data_path=train_data_path)
    exp_key = '_'.join([task_type, m.group(1), method, str(output_size),
                        str(use_redshift)])

    trial_path = 'mongo://{hostname}:{port}/{db_name}/jobs'.format(
        hostname=hostname, port=port, db_name=db_name
    )
    trials = MongoTrials(trial_path, exp_key=exp_key)

    step = 10
    result_path = os.path.join(output_dir, 'best_parameter.json')
    for i in range(len(trials.trials), n_iterations, step):
        n = min(i + step, n_iterations)
        best = fmin(f_objective, space=space, algo=tpe.suggest,
                    max_evals=n, trials=trials)

        print('best score: ', best)
        print(trials.best_trial)
        with open(result_path, 'w') as f:
            json.dump(trials.best_trial, f, sort_keys=True, indent=4,
                      default=serialize_object)


def search(task_type, output_size, epoch, batch_size, band_data, method,
           use_redshift, train_data_path, validation_data_path, test_data_path,
           output_dir, n_iterations):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f_objective = get_objective_function(
        task_type=task_type, output_size=output_size, epoch=epoch,
        batch_size=batch_size, band_data=band_data, method=method,
        use_redshift=use_redshift, train_data_path=train_data_path,
        validation_data_path=validation_data_path,
        test_data_path=test_data_path
    )

    space = get_search_space(task_type=task_type)

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


def get_optimization_function(task_type, input_size, output_size, epoch,
                              batch_size, band_data, method, use_redshift,
                              train_data_path, validation_data_path,
                              test_data_path, hidden_size, dropout_rate,
                              blackout_rate, outlier_rate, n_highways, use_bn):
    if task_type == 'sn_class':
        f = train_normal_classifier_model.run_optimization
    elif task_type == 'redshift':
        f = train_normal_redshift_regressor.run_optimization
    elif task_type == 'sn_epoch':
        f = train_normal_sn_epoch_regressor.run_optimization
    else:
        raise ValueError(task_type)

    d = set_parameter(
        task_type=task_type, input_size=input_size, epoch=epoch,
        batch_size=batch_size, band_data=band_data, method=method,
        train_data_path=train_data_path,
        validation_data_path=validation_data_path,
        test_data_path=test_data_path, output_size=output_size,
        use_redshift=use_redshift
    )
    d.update(dict(
        dropout_rate=dropout_rate, blackout_rate=blackout_rate,
        outlier_rate=outlier_rate, hidden_size=hidden_size,
        n_highways=n_highways, use_bn=use_bn
    ))

    model, mean, std, metrics = f(**d)

    return model, mean, std, metrics


def optimize(task_type, output_size, epoch, batch_size, band_data, method,
             use_redshift, train_data_path, validation_data_path,
             test_data_path, output_dir, parameter_path,
             hidden_size, dropout_rate, blackout_rate, outlier_rate,
             n_highways, use_bn):
    """
    オプションの例(redshiftの場合)

    ROOT_DIR=/data/antares/imoto/crest_auto
    DATA_DIR=$ROOT_DIR/data/processed/180420/dataset_selected/train
    OUT_DIR=$ROOT_DIR/models/tmp180706

    redshift
    optimize
    --epoch=3
    --band_data={\"r\":1,\"i\":6,\"z\":7}
    --method=modified
    --train_data_path=$DATA_DIR/dataset.tr-2classes.nc
    --validation_data_path=$DATA_DIR/dataset.va-2classes.nc
    --test_data_path=$DATA_DIR/dataset.te-2classes.nc
    --output_dir=$OUT_DIR/redshift/optimization
    --parameter_path=$OUT_DIR/redshift/best_parameter.json

    :param task_type:
    :param output_size:
    :param epoch:
    :param batch_size:
    :param band_data:
    :param method:
    :param use_redshift:
    :param train_data_path:
    :param validation_data_path:
    :param test_data_path:
    :param output_dir:
    :param parameter_path:
    :param hidden_size:
    :param dropout_rate:
    :param blackout_rate:
    :param outlier_rate:
    :param n_highways:
    :param use_bn:
    :return:
    """
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

    model, mean, std, metrics = get_optimization_function(
        task_type=task_type, input_size=input_size, output_size=output_size,
        epoch=epoch, batch_size=batch_size, band_data=band_data, method=method,
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
        'band_data': band_data, 'method': method,
        'train_data_path': train_data_path,
        'validation_data_path': validation_data_path,
        'test_data_path': test_data_path,
        'hidden_size': hidden_size, 'dropout_rate': dropout_rate,
        'outlier_rate': outlier_rate, 'blackout_rate': blackout_rate,
        'n_highways': n_highways, 'use_bn': use_bn,
        'mean': mean.tolist(), 'std': std.tolist(), 'metrics': metrics
    }
    if task_type == 'sn_class':
        if output_size is None:
            output_size = get_output_size(data_path=train_data_path)
        summary['output_size'] = output_size
    if task_type != 'redshift':
        summary['use_redshift'] = use_redshift
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4, sort_keys=True)


def set_parameter(task_type, input_size, epoch, batch_size, band_data, method,
                  train_data_path, validation_data_path, test_data_path,
                  output_size, use_redshift):
    d = dict(input_size=input_size, epoch=epoch,
             batch_size=batch_size, band_data=band_data, method=method,
             train_data_path=train_data_path,
             validation_data_path=validation_data_path,
             test_data_path=test_data_path)
    if task_type == 'sn_class':
        if output_size is None:
            # 明示的に値が設定されなかった場合はデータの名前から取得
            output_size = get_output_size(data_path=train_data_path)
        d['output_size'] = output_size
    if task_type != 'redshift':
        d['use_redshift'] = use_redshift

    return d


def get_output_size(data_path):
    r = re.compile(r'(\d)classes')
    m = r.search(data_path)
    if m is None:
        raise ValueError(r'data path does not contain "classes"'
                         r' (e.g. 2classes or 6classes)')
    output_size = int(m.group(1))
    return output_size


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


def main():
    cmd()


if __name__ == '__main__':
    main()
