#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
Autoregressive Convolutional Neural Networks for Asynchronous Time Series
(ICML 2018)
を参考に入力を与える
"""

import json
import os
from pathlib import Path
from collections import namedtuple

import click
import numpy as np
import pandas as pd
import xarray as xr
import sonnet as snt
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm, trange

from plasticc_a1_classifier1 import (convert_data, Dataset,
                                     draw_confusion_matrix)

__author__ = 'Yasuhiro Imoto'
__date__ = '28/11/2018'


def setup_dataset(values, use_hostgal, band, time, repeat, shuffle,
                  batch_size):
    if use_hostgal:
        additional_inputs = {
            key: getattr(values, key)
            for key in ('specz', 'photoz', 'photoz_err')
        }
    else:
        additional_inputs = None

    iterator, next_element = make_dataset(
        flux=values.flux, flux_err=values.flux_err, target=values.target,
        repeat=repeat, shuffle=shuffle,
        additional_inputs=additional_inputs, batch_size=batch_size,
        object_id=values.object_id, band=band, time=time
    )

    _, count = np.unique(values.target, return_counts=True)

    return iterator, next_element, count


def make_dataset(flux, flux_err, target, band, time, object_id, shuffle,
                 repeat, batch_size, additional_inputs=None):
    inputs = {'flux': flux, 'flux_err': flux_err, 'target': target,
              'object_id': object_id}
    if additional_inputs is not None:
        inputs.update(additional_inputs)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.repeat(repeat)
    if shuffle:
        global_step = tf.train.get_or_create_global_step()
        dataset = dataset.shuffle(1000, seed=global_step)
    dataset = dataset.map(
        lambda data: map_func(
            data, use_meta_data=additional_inputs is not None,
            band=band, time=time
        )
    )
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def map_func(data, use_meta_data, band, time):
    flux = data['flux']
    flux_err = data['flux_err']
    mag = tf.asinh(
        0.5 * (flux + flux_err * tf.random_normal(shape=tf.shape(flux)))
    )

    # 縦:観測、横:バンドの種類
    band_size = 6
    band = tf.one_hot(band, band_size)

    x = tf.concat(
        [tf.reshape(mag, [-1, 1]), band, tf.reshape(time, [-1, 1])], axis=1
    )
    # 畳み込み用にチャネルを追加
    x = tf.reshape(x, [1, -1, band_size + 2])

    y = data['target']
    object_id = data['object_id']

    new_data = {'magnitude': x, 'target': y, 'object_id': object_id}
    if not use_meta_data:
        return new_data

    if 'specz' in data:
        # 時々データをNaN相当(0)にする
        flag = tf.to_float(tf.squeeze(
            tf.multinomial([[tf.log(0.99), tf.log(0.01)]], num_samples=1)
        ))
        data['specz'] = flag * data['specz']
    if 'photoz' in data and 'photoz_err' in data:
        data['photoz'] = (data['photoz'] +
                          tf.random_normal(shape=[]) * data['photoz_err'])

    names = ('flux', 'flux_err', 'target', 'photoz_err', 'object_id')
    keys = [key for key in data.keys() if key not in names]
    # 順序を固定
    keys.sort()

    meta = [tf.reshape(data[key], [-1]) for key in keys]
    meta = tf.concat(meta, axis=0)

    new_data['meta_data'] = meta
    return new_data


class Classifier(snt.AbstractModule):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes

    def _build(self, inputs, meta, is_training):
        channels = 4
        h1 = snt.Conv2D(
            output_channels=channels, kernel_shape=(3, 8), stride=(1, 8),
            data_format='NCHW',
            initializers={"w": tf.keras.initializers.he_normal()}
        )(inputs)
        h2 = snt.Conv2D(
            output_channels=channels, kernel_shape=(5, 8), stride=(1, 8),
            data_format='NCHW',
            initializers={"w": tf.keras.initializers.he_normal()}
        )(inputs)
        h3 = snt.Conv2D(
            output_channels=channels, kernel_shape=(7, 8), stride=(1, 8),
            data_format='NCHW',
            initializers={"w": tf.keras.initializers.he_normal()}
        )(inputs)
        h4 = snt.Conv2D(
            output_channels=channels, kernel_shape=(9, 8), stride=(1, 8),
            data_format='NCHW',
            initializers={"w": tf.keras.initializers.he_normal()}
        )(inputs)

        h = tf.concat([h1, h2, h3, h4], axis=-1)
        h = tf.nn.relu(h)

        # [batch, 64, time, 1]
        h = snt.Conv2D(
            output_channels=channels, kernel_shape=(3, 4), stride=(1, 4),
            data_format='NCHW',
            initializers={"w": tf.keras.initializers.he_normal()}
        )(h)
        # [batch, 1, 64, time]
        h = tf.reshape(h, [-1, 1, channels, h.get_shape()[2].value])

        h = snt.BatchNormV2(
            data_format='NCHW', update_ops_collection=tf.GraphKeys.UPDATE_OPS
        )(h, is_training=is_training)
        h = snt.Sequential([
            tf.nn.relu,
            snt.Conv2D(
                output_channels=channels, kernel_shape=(3, 3),
                data_format='NCHW',
                initializers={"w": tf.keras.initializers.he_normal()}
            )
        ])(h)

        g = snt.BatchNormV2(
            data_format='NCHW', update_ops_collection=tf.GraphKeys.UPDATE_OPS
        )(h, is_training=is_training)
        # g = h
        g = tf.nn.relu(g)
        g = snt.Conv2D(
            output_channels=channels, kernel_shape=(3, 3), data_format='NCHW',
            use_bias=False,
            initializers={"w": tf.keras.initializers.he_normal()}
        )(g)
        g = snt.BatchNormV2(
            data_format='NCHW', update_ops_collection=tf.GraphKeys.UPDATE_OPS
        )(g, is_training=is_training)
        g = tf.nn.relu(g)
        g = snt.Conv2D(
            output_channels=channels, kernel_shape=(3, 3), data_format='NCHW',
            initializers={"w": tf.keras.initializers.he_normal()}
        )(g)

        h = h + g

        h = tf.keras.layers.GlobalAvgPool2D(data_format='channels_first')(h)

        if meta is not None:
            h = tf.concat([h, meta], axis=-1)

        outputs = snt.Sequential([
            tf.nn.relu,
            snt.Linear(
                output_size=64,
                initializers={"w": tf.keras.initializers.he_normal()}
            ),
            tf.nn.relu,
            snt.Linear(
                output_size=self.classes,
                initializers={"w": tf.keras.initializers.he_normal()}
            )
        ])(h)

        return outputs


def build_train(model, iterator, next_element, count, mixup, n_classes):
    weight = 1.0 / count
    # weight = np.sqrt(weight)
    print(weight)

    logits = model(
        next_element['magnitude'], next_element.get('meta_data'),
        is_training=True
    )
    labels = next_element['target']

    if mixup:
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(labels), logits=logits
        )

        labels = tf.argmax(labels, axis=-1)
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
    w = tf.constant(weight, dtype=tf.float32)
    loss = loss * tf.gather(w, labels) / tf.reduce_sum(w)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=1e-3, momentum=0.9, use_nesterov=True
    )
    updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(updates):
        opt_op = optimizer.minimize(tf.reduce_sum(loss))

    summary_op, update_op, reset_op = make_metrics(
        logits=logits, loss=loss, labels=labels,
        n_classes=n_classes, name='train'
    )

    TrainOperators = namedtuple(
        'TrainOperators',
        ['optimize', 'update', 'summary', 'reset', 'initialize']
    )

    ops = TrainOperators(
        optimize=opt_op, update=update_op, summary=summary_op,
        reset=reset_op, initialize=iterator.initializer
    )
    return ops, weight


def build_validation(model, iterator, next_element, validation_count, count,
                     n_classes):
    logits = model(
        next_element['magnitude'], next_element.get('meta_data'),
        is_training=False
    )
    labels = next_element['target']
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits
    )
    # final scoreでクラスごとの重みがほぼ均一になるようにとのことなので、
    # final scoreを計算するときのデータの個数で平均するのだと思う
    # 計算式の方にNがあるので、それを打ち消して実際の個数での平均にする値が
    # wの値と予想
    weight = count / validation_count
    w = tf.constant(weight, dtype=tf.float32)
    m = tf.gather(w / count, labels)
    n = tf.reduce_sum(w)
    loss = loss * m / n

    summary_op, update_op, reset_op = make_metrics(
        logits=logits, loss=loss,
        labels=labels, n_classes=n_classes,
        name='validation'
    )

    ValidationOperators = namedtuple(
        'ValidationOperators',
        ['update', 'summary', 'reset', 'initialize']
    )

    ops = ValidationOperators(
        update=update_op, summary=summary_op,
        reset=reset_op, initialize=iterator.initializer
    )
    return ops, weight


def make_metrics(logits, loss, labels, n_classes, name):
    with tf.variable_scope(name) as vs:
        total_loss = tf.get_local_variable(
            'total_loss', shape=[], dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )
        update_total_loss = tf.assign_add(total_loss, tf.reduce_sum(loss))

        predictions = tf.argmax(logits, axis=-1)
        mean_accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predictions, name='my_accuracy'
        )

        cm = tf.confusion_matrix(
            labels=labels, predictions=predictions, num_classes=n_classes,
            name='batch_cm'
        )
        confusion_matrix = tf.get_local_variable(
            'cm', shape=[n_classes, n_classes], dtype=tf.int32,
            initializer=tf.zeros_initializer()
        )

        local_variables = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES
        )
        reset_op = tf.variables_initializer(local_variables)

    update_cm = tf.assign_add(confusion_matrix, cm)
    cm_image = draw_confusion_matrix(confusion_matrix)

    summary_op = tf.summary.merge([
        tf.summary.scalar('{}/loss'.format(name), total_loss),
        tf.summary.scalar('{}/accuracy'.format(name), mean_accuracy[0]),
        tf.summary.image('{}/confusion_matrix'.format(name), cm_image)
    ])

    update_op = tf.group(update_total_loss, mean_accuracy[1], update_cm)

    return summary_op, update_op, reset_op


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--data-dir', type=click.Path(exists=True),
              default='../data/processed/PLAsTiCC_A1_181116')
@click.option('--model-dir', type=click.Path())
@click.option('--epochs', type=int, default=10)
@click.option('--batch-size', type=int, default=200)
@click.option('--seed', type=int, default=0)
@click.option('--hostgal', is_flag=True)
@click.option('--cv', type=int, default=0)
@click.option('--mixup', is_flag=True)
def train(data_dir, model_dir, epochs, batch_size, seed, hostgal, cv, mixup):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    data_dir = Path(data_dir)
    data_path = data_dir / 'train.nc'

    parameters = {
        'data': {'path': str(data_path), 'hostgal': hostgal},
        'seed': seed, 'cv': cv, 'mixup': mixup
    }
    with (model_dir / 'parameters.json').open('w') as f:
        json.dump(parameters, f, sort_keys=True, indent=4)

    dataset, (label, count) = convert_data(data_path=data_path)

    print(label)
    print(count)
    # noinspection PyTypeChecker
    count = count[label >= 0]
    n_classes = len(count)

    tmp = train_test_split(
        *list(dataset), test_size=0.2, random_state=seed,
        stratify=dataset.target
    )

    ds = xr.open_dataset(data_path)
    band = ds.passband.values
    mjd = ds.y.values
    min_mjd, max_mjd = np.min(mjd), np.max(mjd)
    # [-1, 1]に変換
    mjd = (mjd - min_mjd) / (max_mjd - min_mjd) * 2 - 1
    mjd = mjd.astype(np.float32)

    # noinspection PyProtectedMember
    train_values = Dataset._make(tmp[0::2])
    # noinspection PyProtectedMember
    validation_values = Dataset._make(tmp[1::2])

    with tf.Graph().as_default() as graph:
        train_iterator, train_element, train_count = setup_dataset(
            values=train_values, use_hostgal=hostgal, shuffle=True,
            batch_size=batch_size, repeat=1, band=band, time=mjd
        )
        (validation_iterator, validation_element,
         validation_count) = setup_dataset(
            values=validation_values, use_hostgal=hostgal, shuffle=False,
            batch_size=batch_size, repeat=1, band=band, time=mjd
        )

        model = Classifier(classes=n_classes)

        train_ops, train_wights = build_train(
            model=model, iterator=train_iterator, next_element=train_element,
            count=train_count, n_classes=n_classes, mixup=False
        )
        validation_ops, validation_weights = build_validation(
            model=model, iterator=validation_iterator,
            next_element=validation_element, validation_count=validation_count,
            count=train_count + validation_count, n_classes=n_classes
        )

        global_step = tf.train.get_or_create_global_step()
        count_op = tf.assign_add(global_step, 1)

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(str(model_dir))

        train_wights = [float(v) for v in train_wights]
        validation_weights = [float(v) for v in validation_weights]
        with (model_dir / 'weights.json').open('w') as f:
            json.dump(
                {'train': train_wights, 'validation': validation_weights},
                f, sort_keys=True, indent=4
            )

        train_logits = model(
            train_element['magnitude'], train_element.get('meta_data'),
            is_training=False
        )
        with tf.variable_scope('train_metric2'):
            train_accuracy = tf.metrics.accuracy(
                labels=train_element['target'],
                predictions=tf.argmax(train_logits, axis=1)
            )

        validation_logits = model(
            validation_element['magnitude'],
            validation_element.get('meta_data'), is_training=False
        )
        with tf.variable_scope('validation_metric2'):
            validation_accuracy = tf.metrics.accuracy(
                labels=validation_element['target'],
                predictions=tf.argmax(validation_logits, axis=1)
            )

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=config, graph=graph) as sess:
            checkpoint = tf.train.get_checkpoint_state(model_dir)
            if checkpoint:
                path = checkpoint.model_checkpoint_path
                saver.restore(sess=sess, save_path=path)

                sess.run(tf.local_variables_initializer())
            else:
                sess.run(tf.group(tf.global_variables_initializer(),
                                  tf.local_variables_initializer()))

            step = sess.run(global_step)

            for _ in trange(step, epochs):
                sess.run(train_ops.initialize)
                while True:
                    try:
                        sess.run([train_ops.optimize, train_ops.update])
                    except tf.errors.OutOfRangeError:
                        break
                summary, step = sess.run([train_ops.summary, count_op])
                writer.add_summary(summary=summary, global_step=step)
                sess.run(train_ops.reset)

                sess.run(validation_ops.initialize)
                while True:
                    try:
                        sess.run(validation_ops.update)
                    except tf.errors.OutOfRangeError:
                        break
                summary = sess.run(validation_ops.summary)
                writer.add_summary(summary=summary, global_step=step)
                sess.run(validation_ops.reset)

                saver.save(sess=sess, save_path=str(model_dir / 'model'),
                           global_step=global_step, write_meta_graph=False)

            # sess.run(train_iterator.initializer)
            sess.run(train_ops.initialize)
            while True:
                try:
                    sess.run(train_accuracy[1])
                except tf.errors.OutOfRangeError:
                    break

            sess.run(validation_ops.initialize)
            while True:
                try:
                    sess.run(validation_accuracy[1])
                except tf.errors.OutOfRangeError:
                    break

            train_score = float(sess.run(train_accuracy[0]))
            validation_score = float(sess.run(validation_accuracy[0]))

    with (model_dir / 'score.json').open('w') as f:
        json.dump({'train': train_score, 'test': validation_score}, f,
                  sort_keys=True, indent=4)


def main():
    cmd()


if __name__ == '__main__':
    main()
