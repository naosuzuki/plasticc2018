#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
https://arxiv.org/pdf/1611.01449.pdf
特徴量空間での距離を考慮した半教師あり学習

入力データのバッチサイズが異なるのでkerasでは実行できない
諦めてtensorflowで実装する
"""
import json
import math
import os
import re
from collections import namedtuple

import click
import numpy as np
import sonnet as snt
import tensorflow as tf
import xarray as xr
from sklearn import utils
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.manifold import TSNE
# from bhtsne import tsne
from umap import UMAP
try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import matplotlib.pyplot as plt
    import seaborn as sns

from deep_triplet_net import NumpyEncoder

__author__ = 'Yasuhiro Imoto'
__date__ = '21/6/2018'


class FeatureBlock(snt.AbstractModule):
    def __init__(self, output_size, drop_rate):
        super().__init__()
        self.output_size = output_size
        self.rate = drop_rate

    def _build(self, inputs, training, bn_training):
        h = snt.Linear(output_size=self.output_size, use_bias=False)(inputs)
        # h = tf.layers.dropout(h, rate=self.rate, training=training)
        h = snt.BatchNorm(axis=[0])(
            h, is_training=tf.logical_and(training, bn_training),
            test_local_stats=False
        )
        h = tf.keras.layers.PReLU()(h)

        return h


class FeatureNetwork(snt.AbstractModule):
    def __init__(self):
        super().__init__()

    def _build(self, inputs, training, bn_training):
        h = FeatureBlock(output_size=100, drop_rate=0.5)(
            inputs, training, bn_training
        )
        h = FeatureBlock(output_size=50, drop_rate=0.5)(
            h, training, bn_training
        )
        h = FeatureBlock(output_size=25, drop_rate=0.5)(
            h, training, bn_training
        )
        outputs = snt.Linear(output_size=10)(h)
        return outputs


class DistanceLayer(snt.AbstractModule):
    def __init__(self):
        super().__init__()

    def _build(self, samples, references):
        """

        :param samples: 特徴空間に変換済みのデータ
        :param references: 特徴空間に変換済みの基準点
        :return:
        """
        # broadcastがいい感じになる様に軸を調整
        samples = tf.expand_dims(samples, axis=1)

        # L2 distance
        l2 = tf.reduce_sum(tf.squared_difference(samples, references), axis=2)
        logits = -l2
        return logits


def load_blobs():
    return make_blobs(n_samples=100000, centers=2, n_features=2,
                      random_state=0)


def load_data(file_path):
    ds = xr.open_dataset(file_path)
    flux = ds.flux.values
    label = ds.label.values

    flux = np.nan_to_num(flux)

    magnitude = np.arcsinh(flux * 0.5)
    return magnitude, label


Dataset = namedtuple('Dataset', ['x', 'y', 'labeled_x', 'labeled_y',
                                 'unlabeled_x', 'unlabeled_y'])


def load_train_data(file_path, label_ratio):
    x, y = load_data(file_path=file_path)

    # ラベルあり、無しの両方をまとめて正規化していいと思う
    transformer = StandardScaler(copy=False)
    x = transformer.fit_transform(x)

    x1, x2, y1, y2 = train_test_split(
        x, y, stratify=y, test_size=1.0 - label_ratio, random_state=1
    )

    x1 = x1.astype(np.float32)
    y1 = y1.astype(np.int32)
    x2 = x2.astype(np.float32)
    y2 = y2.astype(np.int32)
    x = x.astype(np.float32)
    y = y.astype(np.int32)

    dataset = Dataset(x=x, y=y, labeled_x=x1, labeled_y=y1,
                      unlabeled_x=x2, unlabeled_y=y2)

    return dataset, transformer


def load_validation_data(file_path, transformer, label_ratio):
    r = re.compile(r'\.tr-')
    validation_path = r.sub(r'.va-', file_path)
    x, y = load_data(file_path=validation_path)

    x = transformer.transform(x)

    x1, x2, y1, y2 = train_test_split(
        x, y, stratify=y, test_size=1.0 - label_ratio, random_state=1
    )

    x1 = x1.astype(np.float32)
    y1 = y1.astype(np.int32)
    x2 = x2.astype(np.float32)
    y2 = y2.astype(np.int32)
    x = x.astype(np.float32)
    y = y.astype(np.int32)

    dataset = Dataset(x=x, y=y, labeled_x=x1, labeled_y=y1,
                      unlabeled_x=x2, unlabeled_y=y2)

    return dataset


def load_test_data(file_path, transformer, label_ratio):
    r = re.compile(r'\.tr-')
    validation_path = r.sub(r'.te-', file_path)
    x, y = load_data(file_path=validation_path)

    x = transformer.transform(x)

    x1, x2, y1, y2 = train_test_split(
        x, y, stratify=y, test_size=1.0 - label_ratio, random_state=1
    )

    x1 = x1.astype(np.float32)
    y1 = y1.astype(np.int32)
    x2 = x2.astype(np.float32)
    y2 = y2.astype(np.int32)
    x = x.astype(np.float32)
    y = y.astype(np.int32)

    dataset = Dataset(x=x, y=y, labeled_x=x1, labeled_y=y1,
                      unlabeled_x=x2, unlabeled_y=y2)

    return dataset


def predict_knn(x1, y1, x2, k):
    """
    ラベル付きのデータを対象にk nearest neighborで近傍のクラスを求めて
    個数が最も多かったクラスを予測結果とする
    :param x1:
    :param y1:
    :param x2:
    :param k:
    :return:
    """
    def f(x):
        _, index = tf.nn.top_k(
            -tf.reduce_sum(tf.squared_difference(x1, x), axis=1),
            k=k,
            sorted=False
        )
        return index

    def g(x):
        y, _, count = tf.unique_with_counts(x)
        c = tf.gather(y, tf.argmax(count))
        return c

    with tf.device('/cpu:0'):
        indices = tf.map_fn(f, elems=x2, dtype=tf.int32, back_prop=False)
        # [batch, k]を一次元に並び替える
        flat_indices = tf.reshape(indices, [-1])
        labels = tf.gather(y1, flat_indices)
        # shapeを元に戻す
        labels = tf.reshape(labels, tf.shape(indices))

        # map_fnの対象をunique_with_countsだけにしたかったが、なぜかわからないが、
        # 戻り値の型の指定が上手くいかなかったので、別の書き方にする
        best_class = tf.map_fn(g, elems=labels, dtype=tf.int32,
                               back_prop=False)
    return best_class


def make_input_fn(data, batch_size):
    n_dims = data.x.shape[1]
    n_classes = len(np.unique(data.y))

    def input_fn():
        type_x = {'labeled': tf.float32, 'unlabeled': tf.float32,
                  'reference': tf.float32}
        type_y = {'labeled': tf.int32, 'unlabeled': tf.int32}

        shape_x = {'labeled': tf.TensorShape([None, n_dims]),
                   'unlabeled': tf.TensorShape([None, n_dims]),
                   'reference': tf.TensorShape([n_classes, n_dims])}
        shape_y = {'labeled': tf.TensorShape([None]),
                   'unlabeled': tf.TensorShape([None])}

        dataset = tf.data.Dataset.from_generator(
            generator=lambda: generate_data(data, batch_size),
            output_shapes=(shape_x, shape_y),
            output_types=(type_x, type_y)
        )
        return dataset
    return input_fn


def generate_data(dataset, batch_size):
    for (batch_x, batch_y), (batch_u, batch_v), reference in zip(
            generate_labeled_data(x=dataset.labeled_x, y=dataset.labeled_y,
                                  batch_size=batch_size),
            generate_unlabeled_data(
                x=dataset.unlabeled_x, y=dataset.unlabeled_y,
                batch_size=batch_size
            ),
            generate_reference_data(x=dataset.labeled_x, y=dataset.labeled_y)):
        d = {'labeled': batch_x, 'unlabeled': batch_u,
             'reference': reference}
        e = {'labeled': batch_y, 'unlabeled': batch_v}
        yield d, e


def generate_labeled_data(x, y, batch_size):
    n = (len(x) + batch_size - 1) // batch_size
    while True:
        tmp_x, tmp_y = utils.shuffle(x, y)
        for i in range(n):
            batch_x = tmp_x[i * batch_size:(i + 1) * batch_size]
            batch_y = tmp_y[i * batch_size:(i + 1) * batch_size]
            yield batch_x, batch_y


def generate_unlabeled_data(x, y, batch_size):
    """
    学習制度の評価のためにunlabeled dataの正解ラベルも一緒に返す
    :param x:
    :param y:
    :param batch_size:
    :return:
    """
    n = (len(x) + batch_size - 1) // batch_size
    if y is None:
        y = np.empty(len(x), dtype=np.int32)
    while True:
        tmp_x, tmp_y = utils.shuffle(x, y)
        for i in range(n):
            batch_x = tmp_x[i * batch_size:(i + 1) * batch_size]
            batch_y = tmp_y[i * batch_size:(i + 1) * batch_size]
            yield batch_x, batch_y


def generate_reference_data(x, y):
    class_data = [x[y == i] for i in np.unique(y)]
    while True:
        index = [np.random.choice(len(d)) for d in class_data]
        data = np.vstack([d[i] for d, i in zip(class_data, index)])
        yield data


def sample_reference_data(class_data):
    while True:
        index = [np.random.choice(len(d)) for d in class_data]
        data = np.vstack([d[i] for d, i in zip(class_data, index)])
        yield data


def make_input_fn_prediction(data, batch_size):
    """
    predictionの場合のデータの入力
    ラベルありとかなしとかがややこしいことになっているので、入力形式を変更する
    :param data:
    :param batch_size:
    :return:
    """
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((data.x, data.y))
        dataset = dataset.repeat(1).batch(batch_size=batch_size)
        return dataset
    return input_fn


# noinspection PyUnusedLocal
def model_fn(features, labels, mode, params, config):
    feature_network = FeatureNetwork()

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = feature_network(features, False, False)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    training = mode == tf.estimator.ModeKeys.TRAIN
    labeled_feature = feature_network(
        features['labeled'], training, bn_training=True
    )
    unlabeled_feature = feature_network(
        features['unlabeled'], training, bn_training=False
    )
    reference_feature = feature_network(
        features['reference'], training, bn_training=False
    )

    distance_layer = DistanceLayer()
    labeled_logits = distance_layer(labeled_feature, reference_feature)
    unlabeled_logits = distance_layer(unlabeled_feature, reference_feature)

    tmp = tf.nn.softmax(labeled_logits)
    shape = tf.shape(labeled_logits)
    batch_index = tf.range(shape[0])
    index = tf.stack([batch_index, labels['labeled']], axis=1)
    value = tf.gather_nd(tmp, index)
    labeled_cost = -tf.square(value) + 1
    # labeled_cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     labels=labels['labeled'], logits=labeled_logits
    # )
    # labelとlogitsの両方で勾配を求めるのが正しいと思うので、_v2
    unlabeled_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tf.nn.softmax(unlabeled_logits),
        logits=unlabeled_logits
    )
    cost = (params['lambda_l'] * tf.reduce_mean(labeled_cost) +
            params['lambda_u'] * tf.reduce_mean(unlabeled_cost))

    accuracy = tf.metrics.accuracy(
        labels=labels['labeled'], predictions=tf.argmax(labeled_logits, axis=1)
    )
    labeled_loss = tf.metrics.mean(labeled_cost)
    unlabeled_loss = tf.metrics.mean(unlabeled_cost)
    metrics = {'labeled/accuracy': accuracy, 'labeled/loss': labeled_loss,
               'unlabeled/loss': unlabeled_loss}
    tf.summary.scalar('labeled/accuracy', accuracy[1])
    tf.summary.scalar('labeled/loss', labeled_loss[1])
    tf.summary.scalar('unlabeled/loss', unlabeled_loss[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        # 計算が遅いので、評価の場合のみ
        for k in (1, 3, 5):
            knn_accuracy = tf.metrics.accuracy(
                labels=labels['unlabeled'],
                predictions=predict_knn(
                    features['labeled'], labels['labeled'],
                    features['unlabeled'], k=k
                )
            )
            name = 'unlabeled/accuracy_knn{}'.format(k)
            metrics[name] = knn_accuracy
            tf.summary.scalar(name, knn_accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=cost, eval_metric_ops=metrics
        )

    global_step = tf.train.get_global_step()
    optimizer = tf.train.AdamOptimizer()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(cost, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode=mode, loss=cost, train_op=train_op)


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--lambda_l', type=float, default=1.0)
@click.option('--lambda_u', type=float, default=1.0)
@click.option('--epoch', type=int, default=100)
@click.option('--output-dir', type=str, default='')
@click.option('--data-path', type=str,
              default='../../data/processed/180420/dataset_selected/train/'
                      'dataset.tr-2classes.nc')
@click.option('--label-ratio', type=float, default=0.5)
def train(lambda_l, lambda_u, epoch, output_dir, data_path, label_ratio):
    print(data_path)
    train_dataset, transformer = load_train_data(file_path=data_path,
                                                 label_ratio=label_ratio)
    test_dataset = load_test_data(
        file_path=data_path, transformer=transformer, label_ratio=label_ratio
    )

    config = tf.estimator.RunConfig(
        session_config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True)
        ),
        save_summary_steps=1000
    )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=output_dir, config=config,
        params={'lambda_l': lambda_l, 'lambda_u': lambda_u}
    )

    batch_size = 500
    train_size = max(len(train_dataset.labeled_x),
                     len(train_dataset.unlabeled_x))
    test_size = max(len(test_dataset.labeled_x), len(test_dataset.unlabeled_x))
    estimator.train(
        input_fn=make_input_fn(data=train_dataset, batch_size=batch_size),
        steps=int(math.ceil(train_size * epoch / batch_size))
    )

    evaluation_train = estimator.evaluate(
        input_fn=make_input_fn(data=train_dataset, batch_size=batch_size),
        steps=int(math.ceil(train_size / batch_size)),
        name='train'
    )
    evaluation_test = estimator.evaluate(
        input_fn=make_input_fn(data=test_dataset, batch_size=batch_size),
        steps=int(math.ceil(test_size / batch_size)),
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

    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        json.dump({'lambda_l': lambda_l, 'lambda_u': lambda_u,
                   'label_ratio': label_ratio},
                  f, sort_keys=True, indent=4, cls=NumpyEncoder)


@cmd.command()
@click.option('--lambda_l', type=float, default=1.0)
@click.option('--lambda_u', type=float, default=1.0)
@click.option('--output-dir', type=str, default='')
@click.option('--data-path', type=str,
              default='../../data/processed/180420/dataset_selected/train/'
                      'dataset.tr-2classes.nc')
@click.option('--label-ratio', type=float, default=0.5)
def predict(lambda_l, lambda_u, output_dir, data_path, label_ratio):
    print(data_path)
    train_dataset, transformer = load_train_data(file_path=data_path,
                                                 label_ratio=label_ratio)
    test_dataset = load_test_data(
        file_path=data_path, transformer=transformer, label_ratio=label_ratio
    )

    config = tf.estimator.RunConfig(
        session_config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True)
        )
    )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=output_dir, config=config,
        params={'lambda_l': lambda_l, 'lambda_u': lambda_u}
    )

    p = estimator.predict(
        input_fn=make_input_fn_prediction(train_dataset, batch_size=1000),
        yield_single_examples=False
    )
    train_prediction = np.vstack([v for v in p]).astype(np.float64)
    compressed_feature = UMAP().fit_transform(
        train_prediction, y=train_dataset.y
    )
    plot_feature2d(
        os.path.join(output_dir, 'umap_train_feature.png'),
        compressed_feature, train_dataset.y
    )

    p = estimator.predict(
        input_fn=make_input_fn_prediction(test_dataset, batch_size=1000),
        yield_single_examples=False
    )
    test_prediction = np.vstack([v for v in p]).astype(np.float64)
    compressed_feature = UMAP().fit_transform(test_prediction)
    plot_feature2d(
        os.path.join(output_dir, 'umap_test_feature.png'),
        compressed_feature, test_dataset.y
    )


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
