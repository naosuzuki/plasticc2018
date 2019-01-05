#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import json
import os
import re
from datetime import datetime
from functools import partial

import click
import numpy as np
import tensorflow as tf
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.externals import joblib
from tensorflow import keras

from sn_model.dataset import compute_moments_from_file, make_dataset
from sn_model.normal import convert_data, make_model

__author__ = 'Yasuhiro Imoto'
__date__ = '12/1/2018'


def get_data(sess, dataset):
    iterator = dataset.make_one_shot_iterator()
    element = iterator.get_next()
    x, y, name = [], [], []
    while True:
        try:
            tmp = sess.run(element)
            x.append(tmp['x'])
            y.append(tmp['y'])
            name.append(tmp['name'])
        except tf.errors.OutOfRangeError:
            break
    x = np.vstack(x)
    y = np.squeeze(np.hstack(y))
    name = np.hstack(name)
    data = {'x': x, 'y': y, 'name': name}
    return data


class MyGenerator(object):
    def __init__(self, dataset, num_classes):
        sess = keras.backend.get_session()
        iterator = dataset.make_one_shot_iterator()
        element = iterator.get_next()
        self.sess = sess
        self.element = element

        self.num_classes = num_classes

    def __iter__(self):
        return self

    def __next__(self):
        try:
            tmp = self.sess.run(self.element)
            y = keras.utils.to_categorical(
                tmp['y'], num_classes=self.num_classes
            )
            return tmp['x'], y
        except tf.errors.OutOfRangeError:
            pass
        #
        # raise StopIteration()


def objective(args, input_size, output_size, epoch, batch_size, band_data,
              method, use_redshift, train_data_path, validation_data_path,
              test_data_path):
    dropout_rate = args['dropout_rate']
    blackout_rate = args['blackout_rate']
    outlier_rate = args['outlier_rate']
    hidden_size = int(args['hidden_size'])

    mean, std = compute_moments_from_file(train_data_path, method=method)
    train_dataset, train_size = make_dataset(
        train_data_path, epochs=epoch, shuffle=True, return_length=True
    )
    validation_dataset, validation_size = make_dataset(
        validation_data_path, epochs=epoch, shuffle=False, return_length=True
    )

    train_dataset = train_dataset.map(
        lambda v: convert_data(
            v, mean=mean, std=std, band_data=band_data, method=method,
            blackout_rate=blackout_rate, outlier_rate=outlier_rate,
            train=True, use_redshift=use_redshift
        )
    ).batch(batch_size)
    validation_dataset = validation_dataset.map(
        lambda v: convert_data(
            v, mean=mean, std=std, band_data=band_data, method=method,
            blackout_rate=blackout_rate, outlier_rate=outlier_rate,
            train=False, use_redshift=use_redshift
        )
    ).batch(batch_size)

    # config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    # sess = tf.Session(config=config)
    # keras.backend.set_session(sess)

    # train_data = get_data(sess, train_dataset)
    # validation_data = get_data(sess, validation_dataset)

    model = make_model(input_size=input_size,
                       hidden_size=hidden_size, output_size=output_size,
                       dropout_rate=dropout_rate)

    history = model.fit_generator(
        generator=MyGenerator(train_dataset, num_classes=output_size),
        steps_per_epoch=train_size // batch_size,
        epochs=epoch,
        callbacks=[keras.callbacks.EarlyStopping(patience=100)],
        validation_data=MyGenerator(validation_dataset,
                                    num_classes=output_size),
        validation_steps=validation_size // batch_size,
        verbose=1
    )

    test_dataset, test_size = make_dataset(test_data_path, epochs=1,
                                           shuffle=False, return_length=True)
    test_dataset = test_dataset.map(
        lambda v: convert_data(
            v, mean=mean, std=std, band_data=band_data, method=method,
            blackout_rate=blackout_rate, outlier_rate=outlier_rate,
            train=False, use_redshift=use_redshift
        )
    ).batch(batch_size)
    # test_data = get_data(sess, test_dataset)
    tmp = model.evaluate_generator(
        generator=MyGenerator(test_dataset, num_classes=output_size),
        steps=(test_size + batch_size - 1) // batch_size
    )
    metrics = {key: float(value)
               for key, value in zip(model.metrics_names, tmp)}

    cost = -history.history['val_categorical_accuracy'][0]

    return {'loss': cost, 'status': STATUS_OK, 'attachments': metrics}


def get_search_space():
    space = {
        'hidden_size': hp.qloguniform('hidden_size',
                                      np.log(10), np.log(1000), 10),
        'dropout_rate': hp.uniform('dropout_rate', 0, 0.99),
        'outlier_rate': hp.uniform('outlier_rate', 0, 0.99),
        'blackout_rate': hp.uniform('blackout_rate', 0, 0.99)
    }
    return space


@click.command()
@click.option('--output_size', type=int, default=6)
@click.option('--epoch', type=int, default=5)
@click.option('--batch_size', type=int, default=100)
@click.option('--band_data', type=str)
@click.option('--method', type=click.Choice(['modified', 'traditional']),
              default='modified')
@click.option('--use_redshift', is_flag=True)
@click.option('--train_data_path', type=str)
@click.option('--validation_data_path', type=str, default=None)
@click.option('--test_data_path', type=str, default=None)
@click.option('--output_dir', type=str)
@click.option('--n_iterations', type=int, default=100)
def main(output_size, epoch, batch_size, band_data, method, use_redshift,
         train_data_path, validation_data_path, test_data_path, output_dir,
         n_iterations):
    # band_data は {"Y": 10, "g": 8} みたいな文字列が与えられる
    band_data = eval(band_data)
    assert isinstance(band_data, dict), "band_data is an unexpected value"
    input_size = sum(band_data.values())
    if use_redshift:
        input_size += 1

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if validation_data_path is None:
        # (?<= ) は肯定後読み
        # (?= ) は肯定先読み
        # trの部分のみが検出される
        r = re.compile(r'(?<=dataset\.)tr(?=(?:-.*)?\.nc)')
        validation_data_path = r.sub('va', train_data_path)
    if test_data_path is None:
        # (?<= ) は肯定後読み
        # (?= ) は肯定先読み
        # trの部分のみが検出される
        r = re.compile(r'(?<=dataset\.)tr(?=(?:-.*)?\.nc)')
        test_data_path = r.sub('te', train_data_path)

    f_objective = partial(
        objective, input_size=input_size, output_size=output_size, epoch=epoch,
        batch_size=batch_size,
        method=method, use_redshift=use_redshift, band_data=band_data,
        train_data_path=train_data_path,
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

    for i in range(len(trials.trials), n_iterations, 10):
        n = min(i + 10, n_iterations)
        best = fmin(f_objective, space=space, algo=tpe.suggest,
                    max_evals=n, trials=trials)

        # 履歴の保存
        joblib.dump(trials, trial_file)

        print('best score: ', best)
        print(trials.best_trial)
        with open(os.path.join(output_dir, 'best_parameter.json'), 'w') as f:
            json.dump(trials.best_trial, f, sort_keys=True, indent=4,
                      default=serialize_datetime)


def serialize_datetime(o):
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(repr(o) + ' is not JSON serializable')


if __name__ == '__main__':
    main()
