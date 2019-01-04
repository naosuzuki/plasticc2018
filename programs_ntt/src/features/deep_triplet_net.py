#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import json
import math
import os
import re

import click
import numpy as np
import sonnet as snt
import tensorflow as tf
import xarray as xr
from sklearn import utils
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = 'Yasuhiro Imoto'
__date__ = '27/6/2018'


class FeatureBlock(snt.AbstractModule):
    def __init__(self, output_size, drop_rate):
        super().__init__()
        self.output_size = output_size
        self.rate = drop_rate

        with self._enter_variable_scope():
            self.prelu = tf.keras.layers.PReLU()

    def _build(self, inputs, training, bn_training, test_local_stats):
        h = snt.Linear(output_size=self.output_size, use_bias=False)(inputs)
        # h = tf.layers.dropout(h, rate=self.rate, training=training)
        h = snt.BatchNorm(axis=[0])(
            h, is_training=tf.logical_and(training, bn_training),
            test_local_stats=test_local_stats
        )
        h = self.prelu(h)

        return h


class FeatureNetwork(snt.AbstractModule):
    def __init__(self):
        super().__init__()

        with self._enter_variable_scope():
            self.feature_block1 = FeatureBlock(output_size=100, drop_rate=0.5)
            self.feature_block2 = FeatureBlock(output_size=50, drop_rate=0.5)
            self.feature_block3 = FeatureBlock(output_size=25, drop_rate=0.5)

    def _build(self, inputs, training, bn_training, test_local_stats):
        h = self.feature_block1(inputs, training, bn_training,
                                test_local_stats)
        h = self.feature_block2(h, training, bn_training, test_local_stats)
        h = self.feature_block3(h, training, bn_training, test_local_stats)
        outputs = snt.Linear(output_size=10)(h)
        return outputs


class DistanceLayer(snt.AbstractModule):
    def __init__(self):
        super().__init__()

    def _build(self, x1, x2):
        distance = tf.reduce_sum(tf.squared_difference(x1, x2), axis=-1,
                                 keepdims=True)

        return distance


class TripletNetwork(snt.AbstractModule):
    def __init__(self):
        super().__init__()

    def _build(self, x, x_plus, x_minus, training):
        feature = FeatureNetwork()

        z = feature(x, training, bn_training=True)
        z_plus = feature(x_plus, training, bn_training=False)
        z_minus = feature(x_minus, training, bn_training=False)

        d_plus = DistanceLayer()(z, z_plus)
        d_minus = DistanceLayer()(z, z_minus)
        d = tf.concat([d_plus, d_minus], axis=-1)
        softmax = tf.nn.softmax(d, axis=-1)
        return softmax


def load_data(file_path):
    ds = xr.open_dataset(file_path)
    flux = ds.flux.values
    label = ds.label.values

    flux = np.nan_to_num(flux)

    magnitude = np.arcsinh(flux * 0.5)
    return magnitude, label


def load_train_data(file_path):
    x, y = load_data(file_path=file_path)
    # 訓練データなので、平均と分散を求めて正規化する
    transformer = StandardScaler(copy=False)    # データを書き換えてもいいと思う
    x = transformer.fit_transform(x)

    x = x.astype(np.float32)
    y = y.astype(np.int32)

    return x, y, transformer


def load_validation_data(file_path, transformer):
    r = re.compile(r'\.tr-')
    validation_path = r.sub(r'.va-', file_path)

    x, y = load_data(file_path=validation_path)
    x = transformer.transform(x)

    x = x.astype(np.float32)
    y = y.astype(np.int32)

    return x, y


def load_test_data(file_path, transformer):
    r = re.compile(r'\.tr-')
    validation_path = r.sub(r'.te-', file_path)

    x, y = load_data(file_path=validation_path)
    x = transformer.transform(x)

    x = x.astype(np.float32)
    y = y.astype(np.int32)

    return x, y


def make_input_fn(x, y, batch_size):
    n_dims = x.shape[1]

    def input_fn():
        t = {'x': tf.float32, 'x_plus': tf.float32, 'x_minus': tf.float32}
        s = {'x': tf.TensorShape([None, n_dims]),
             'x_plus': tf.TensorShape([None, n_dims]),
             'x_minus': tf.TensorShape([None, n_dims])}
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: generate_data(x, y, batch_size),
            output_types=(t, tf.int32),
            output_shapes=(s, tf.TensorShape([None]))
        )
        return dataset
    return input_fn


def generate_data(x, y, batch_size):
    same_class = [x[y == i] for i in np.unique(y)]
    different_class = [x[y != i] for i in np.unique(y)]

    n = (len(x) + batch_size - 1) // batch_size
    while True:
        tmp_x, tmp_y = utils.shuffle(x, y)
        for i in range(n):
            batch_x = tmp_x[i * batch_size:(i + 1) * batch_size]
            batch_y = tmp_y[i * batch_size:(i + 1) * batch_size]

            batch_x_plus = []
            batch_x_minus = []
            for c in batch_y:
                plus_index = np.random.choice(len(same_class[c]))
                minus_index = np.random.choice(len(different_class[c]))

                batch_x_plus.append(same_class[c][plus_index])
                batch_x_minus.append(different_class[c][minus_index])

            feature = {'x': batch_x, 'x_plus': np.asarray(batch_x_plus),
                       'x_minus': np.asarray(batch_x_minus)}
            yield feature, batch_y


def make_input_fn_prediction(data, batch_size):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
        dataset = dataset.repeat(1).batch(batch_size=batch_size)
        return dataset
    return input_fn


# noinspection PyUnusedLocal
def model_fn(features, labels, mode, params, config):
    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope('triplet_network'):
            # モデルの読み込みのためにスコープを合わせる
            feature_network = FeatureNetwork()
        predictions = feature_network(features, False, False)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    d = TripletNetwork()(
        features['x'], features['x_plus'], features['x_minus'],
        mode == tf.estimator.ModeKeys.TRAIN
    )

    loss = tf.square(d[:, 0])

    mean_accuracy, update_mean_accuracy = tf.metrics.accuracy(
        labels=tf.ones_like(labels), predictions=tf.argmax(d, axis=-1)
    )
    # 半教師ありの方に名前を合わせる
    metrics = {'labeled/accuracy': (mean_accuracy, update_mean_accuracy)}
    tf.summary.scalar('labeled/accuracy', update_mean_accuracy)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=tf.reduce_mean(loss), eval_metric_ops=metrics
        )

    global_step = tf.train.get_global_step()
    optimizer = tf.train.AdamOptimizer()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(tf.reduce_mean(loss),
                                      global_step=global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=tf.reduce_mean(loss), train_op=train_op
    )


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--epochs', type=int, default=500)
@click.option('--output-dir', type=str, default='')
@click.option('--data-path', type=str,
              default='../../data/processed/180420/dataset_selected/train/'
                      'dataset.tr-2classes.nc')
def train(epochs, output_dir, data_path):
    train_x, train_y, transformer = load_train_data(file_path=data_path)
    validation_x, validation_y = load_validation_data(file_path=data_path,
                                                      transformer=transformer)
    test_x, test_y = load_test_data(file_path=data_path,
                                    transformer=transformer)

    config = tf.estimator.RunConfig(
        session_config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True)
        ),
        save_summary_steps=1000
    )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=output_dir, config=config, params=()
    )

    batch_size = 500
    estimator.train(
        input_fn=make_input_fn(x=train_x, y=train_y, batch_size=batch_size),
        steps=int(math.ceil(len(train_x) * epochs / batch_size))
    )
    evaluation_train = estimator.evaluate(
        input_fn=make_input_fn(x=train_x, y=train_y, batch_size=batch_size),
        steps=int(math.ceil(len(train_x) / batch_size)),
        name='train'
    )
    evaluation_test = estimator.evaluate(
        input_fn=make_input_fn(x=test_x, y=test_y, batch_size=batch_size),
        steps=int(math.ceil(len(test_x) / batch_size)),
        name='test'
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(evaluation_train, f, sort_keys=True, indent=4,
                  cls=NumpyEncoder)
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(evaluation_test, f, sort_keys=True, indent=4,
                  cls=NumpyEncoder)


@cmd.command()
@click.option('--output-dir', type=str, default='')
@click.option('--data-path', type=str,
              default='../../data/processed/180420/dataset_selected/train/'
                      'dataset.tr-2classes.nc')
def predict(output_dir, data_path):
    train_x, train_y, transformer = load_train_data(file_path=data_path)
    validation_x, validation_y = load_validation_data(file_path=data_path,
                                                      transformer=transformer)
    test_x, test_y = load_test_data(file_path=data_path,
                                    transformer=transformer)

    config = tf.estimator.RunConfig(
        session_config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True)
        ),
        save_summary_steps=10
    )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=output_dir, config=config
    )

    p = estimator.predict(
        input_fn=make_input_fn_prediction((train_x, train_y), batch_size=1000),
        yield_single_examples=False
    )
    train_prediction = np.vstack([v for v in p])
    compressed_feature = UMAP().fit_transform(train_prediction, train_y)
    plot_feature2d(os.path.join(output_dir, 'train_feature.png'),
                   compressed_feature, train_y)

    p = estimator.predict(
        input_fn=make_input_fn_prediction((test_x, test_y), batch_size=1000),
        yield_single_examples=False
    )
    test_prediction = np.vstack([v for v in p])
    compressed_feature = UMAP().fit_transform(test_prediction, test_y)
    plot_feature2d(os.path.join(output_dir, 'test_feature.png'),
                   compressed_feature, test_y)


# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def plot_feature2d(output_path, x, y=None):
    fig, ax = plt.subplots(figsize=(16, 12))
    if y is None:
        ax.scatter(x[:, 0], x[:, 1])
    else:
        for i in np.unique(y):
            tmp = x[y == i]
            ax.scatter(tmp[:, 0], tmp[:, 1], label=i)
        ax.legend(loc='best')
    ax.grid()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close()


def main():
    cmd()


if __name__ == '__main__':
    main()
