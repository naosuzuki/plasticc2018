#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
https://arxiv.org/pdf/1611.01449.pdf
SEMI-SUPERVISED DEEP LEARNING BY METRIC EMBEDDING
特徴量空間での距離を考慮した半教師あり学習
の方法は、ラベルありの部分のコストがあまり下がらない

https://arxiv.org/pdf/1412.6622.pdf
DEEP METRIC LEARNING USING TRIPLET NETWORK
はコストが良く下がる

2クラス分類において(ラベルありの部分の)二つの解は同じになるはずだが、
コストの下がり方に差がある
ラベルなしの部分の重みを0にして計算しても差がある

コスト関数の差が学習結果に違いを産んでいると思うので、コスト関数を
tripletの方に近づけてコスト関数の形が原因かを確かめる
"""

import json
import math
import os
import re

import click
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from sklearn import utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# from umap import UMAP
try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import matplotlib.pyplot as plt
    import seaborn as sns

from deep_triplet_net import NumpyEncoder, FeatureNetwork
from semi_sup_contrast import (load_train_data, load_validation_data,
                               load_test_data, predict_knn)

try:
    import sys
    sys.path.append('../visualization/parametric_tsne')
finally:
    # noinspection PyUnresolvedReferences
    from parametric_tSNE import Parametric_tSNE

__author__ = 'Yasuhiro Imoto'
__date__ = '09/7/2018'


def make_input_fn(data, batch_size):
    output_types_x = {'labeled': tf.float32, 'labeled_plus': tf.float32,
                      'labeled_minus': tf.float32, 'unlabeled': tf.float32}
    output_types_x.update({'unlabeled{}'.format(i): tf.float32
                           for i in np.unique(data.labeled_y)})
    output_types_y = {'labeled': tf.int32, 'unlabeled': tf.int32}
    output_types = (output_types_x, output_types_y)

    n_features = data.labeled_x.shape[1]
    output_shapes_x = {'labeled': tf.TensorShape([None, n_features]),
                       'labeled_plus': tf.TensorShape([None, n_features]),
                       'labeled_minus': tf.TensorShape([None, n_features]),
                       'unlabeled': tf.TensorShape([None, n_features])}
    output_shapes_x.update({
        'unlabeled{}'.format(i): tf.TensorShape([None, n_features])
        for i in np.unique(data.labeled_y)
    })
    output_shapes_y = {'labeled': tf.TensorShape([None]),
                       'unlabeled': tf.TensorShape([None])}
    output_shapes = (output_shapes_x, output_shapes_y)

    def input_fn():
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: make_generator(
                (data.labeled_x, data.labeled_y),
                (data.unlabeled_x, data.unlabeled_y),
                batch_size
            ),
            output_types=output_types,
            output_shapes=output_shapes
        )
        return dataset

    return input_fn


def make_generator(labeled, unlabeled, batch_size):
    for tmp1, tmp2 in zip(
            generator_labeled(labeled[0], labeled[1], batch_size),
            generator_unlabeled(labeled, unlabeled, batch_size)):
        d = {'labeled': tmp1[0], 'labeled_plus': tmp1[1],
             'labeled_minus': tmp1[2], 'unlabeled': tmp2[0]}
        # リストのままは良くない気がするので、展開する
        d.update({'unlabeled{}'.format(i): data
                  for i, data in enumerate(tmp2[1])})

        e = {'labeled': tmp1[3], 'unlabeled': tmp2[2]}

        yield d, e


def generator_labeled(x, y, batch_size):
    same_data = [x[y == i] for i in np.unique(y)]
    different_data = [x[y != i] for i in np.unique(y)]

    n = (len(x) + batch_size - 1) // batch_size
    while True:
        tmp_x, tmp_y = utils.shuffle(x, y)
        for i in range(n):
            batch_x = tmp_x[i * batch_size:(i + 1) * batch_size]
            batch_y = tmp_y[i * batch_size:(i + 1) * batch_size]

            batch_x_plus, batch_x_minus = [], []
            for label in batch_y:
                same = same_data[label]
                different = different_data[label]

                plus_index = np.random.choice(len(same))
                batch_x_plus.append(same[plus_index])

                minus_index = np.random.choice(len(different))
                batch_x_minus.append(different[minus_index])
            yield batch_x, batch_x_plus, batch_x_minus, batch_y


def generator_unlabeled(labeled, unlabeled, batch_size):
    class_data = [labeled[0][labeled[1] == i] for i in np.unique(labeled[1])]

    n = (len(unlabeled[0]) + batch_size - 1) // batch_size
    while True:
        tmp_x, tmp_y = utils.shuffle(unlabeled[0], unlabeled[1])
        for i in range(n):
            batch_x = tmp_x[i * batch_size:(i + 1) * batch_size]
            batch_y = tmp_y[i * batch_size:(i + 1) * batch_size]

            # ラベルありの方のgeneratorと異なり、
            # ラベルごとにまとめてデータをサンプリング
            reference_data = []
            for d in class_data:
                index = np.random.choice(len(d), size=len(batch_x))
                reference_data.append(d[index])
            yield batch_x, reference_data, batch_y


def make_input_fn_mix(artificial_data, observed_data, batch_size):
    """
    input_fnを作る
    データの内訳は、
    1. 観測データに人工データを交ぜる
    2. 距離を計算するための基準に人工データを使って、あとは観測データ
    3. 全て観測データ
    の3パターンがある

    ここでは、パターン2を採用
    :param artificial_data:
    :param observed_data:
    :param batch_size:
    :return:
    """
    def input_fn():
        types_x = {'labeled': tf.float32, 'labeled_plus': tf.float32,
                   'labeled_minus': tf.float32, 'unlabeled': tf.float32}
        types_x.update({'unlabeled{}'.format(i): tf.float32
                        for i in np.unique(artificial_data[2])})
        types_y = {'labeled': tf.int32, 'unlabeled': tf.int32}

        n_features = artificial_data[0].shape[1]
        shapes_x = {'labeled': tf.TensorShape([None, n_features]),
                    'labeled_plus': tf.TensorShape([None, n_features]),
                    'labeled_minus': tf.TensorShape([None, n_features]),
                    'unlabeled': tf.TensorShape([None, n_features])}
        shapes_x.update({
            'unlabeled{}'.format(i): tf.TensorShape([None, n_features])
            for i in np.unique(artificial_data[2])
        })
        shapes_y = {'labeled': tf.TensorShape([None]),
                    'unlabeled': tf.TensorShape([None])}

        dataset = tf.data.Dataset.from_generator(
            generator=lambda: generator_mix(
                artificial_data=artificial_data, observed_data=observed_data,
                batch_size=batch_size
            ),
            output_types=(types_x, types_y),
            output_shapes=(shapes_x, shapes_y)
        )
        return dataset
    return input_fn


def generator_mix(artificial_data, observed_data, batch_size):
    generator1 = generator_labeled_mix(
        artificial_data=artificial_data, observed_data=observed_data,
        batch_size=batch_size
    )
    generator2 = generator_unlabeled_mix(
        artificial_data=artificial_data, observed_data=observed_data,
        batch_size=batch_size
    )
    for labeled, unlabeled in zip(generator1, generator2):
        d = {'labeled': labeled[0], 'labeled_plus': labeled[1],
             'labeled_minus': labeled[2], 'unlabeled': unlabeled[0]}
        d.update({'unlabeled{}'.format(i): v
                  for i, v in enumerate(unlabeled[1])})
        # unlabeledのラベルはすべて-1で役に立たないと思うが、入れておく
        e = {'labeled': labeled[3], 'unlabeled': unlabeled[2]}
        yield d, e


def compute_magnitude(flux, flux_err):
    return np.arcsinh((flux + np.random.randn(*flux.shape) * flux_err) * 0.5)


def generator_labeled_mix(artificial_data, observed_data, batch_size):
    """
    ラベルのある観測データをランダムに選択(anchor)
    anchorと同じクラスと異なるクラスのデータを人工データから選択
    :param xr.Dataset artificial_data:
    :param xr.Dataset observed_data:
    :param batch_size:
    :return:
    """
    # ラベルなしは-1
    tmp = observed_data.label >= 0
    # ラベルありの部分を抽出
    ds_observed = observed_data.isel(index=np.where(tmp)[0])
    indices = list(range(ds_observed.flux.shape[0]))

    # クラスごとに分割
    # インデックスを拾う
    artificial_same_class = [np.where(artificial_data.label == i)[0]
                             for i in np.unique(artificial_data.label)]
    artificial_different_class = [np.where(artificial_data.label != i)[0]
                                  for i in np.unique(artificial_data.label)]

    n = (len(ds_observed.label.values) + batch_size - 1) // batch_size
    while True:
        indices = utils.shuffle(indices)
        for i in range(n):
            s = indices[i * batch_size:(i + 1) * batch_size]
            batch_x = compute_magnitude(flux=ds_observed.flux[s],
                                        flux_err=ds_observed.flux_err[s])
            batch_y = ds_observed.label.values[s]

            positive_index = [np.random.choice(artificial_same_class[j])
                              for j in batch_y]
            negative_index = [np.random.choice(artificial_different_class[j])
                              for j in batch_y]
            positive_data = compute_magnitude(
                flux=artificial_data.flux[positive_index],
                flux_err=artificial_data.flux_err[positive_index]
            )
            negative_data = compute_magnitude(
                flux=artificial_data.flux[negative_index],
                flux_err=artificial_data.flux_err[negative_index]
            )

            yield batch_x, positive_data, negative_data, batch_y


def generator_unlabeled_mix(artificial_data, observed_data, batch_size):
    """
    ラベルなしの観測データをランダムに選択
    各クラスの基準点を人工データからランダムに選択
    :param xr.Dataset artificial_data:
    :param xr.Dataset observed_data:
    :param batch_size:
    :return:
    """
    tmp = observed_data[2] == -1
    ds_observed = observed_data.isel(index=np.where(tmp)[0])
    indices = list(range(ds_observed.flux.shape[0]))

    class_index = [np.where(artificial_data.label == i)[0]
                   for i in np.unique(artificial_data.label)]

    n = (len(ds_observed.label) + batch_size - 1) // batch_size
    while True:
        indices = utils.shuffle(indices)
        for i in range(n):
            s = indices[i * batch_size:(i + 1) * batch_size]
            batch_x = compute_magnitude(flux=ds_observed.flux[s],
                                        flux_err=ds_observed.flux_err[s])
            batch_y = ds_observed.label.values[s]

            reference_data = []
            for index_list in class_index:
                index = np.random.choice(index_list, size=len(batch_x))
                data = compute_magnitude(
                    flux=artificial_data.flux[index],
                    flux_err=artificial_data.flux_err[index]
                )
                reference_data.append(data)

            yield batch_x, reference_data, batch_y


def load_observed_data(data_path, label_path):
    df_label = pd.read_csv(label_path, header=None, delim_whitespace=True)
    # 検索しやすいようにdictに変換
    name_dict = {name: 0 if c == 'Ia' else 1
                 for name, c in zip(df_label[0], df_label[1])}

    ds_data = xr.open_dataset(data_path, engine='scipy')
    flux = ds_data.flux.values
    flux = np.nan_to_num(flux)

    # クラスを番号に変換する
    # クラスの情報がない場合は-1
    label = np.empty(ds_data.flux.shape[0], dtype=np.int32)
    for i, name in enumerate(ds_data.name.values):
        if isinstance(name, bytes):
            name = name.decode()
        # 先頭の数字を取り除く
        name = name[2:]

        label[i] = name_dict.get(name, -1)

    ds = xr.Dataset({'flux': (['index', 'feature'], flux),
                     'flux_err': (['index', 'feature'], ds_data.flux_err),
                     'label': (['index'], label)},
                    coords={'index': range(flux.shape[0]),
                            'feature': range(flux.shape[1])})

    return ds


def load_data(file_path):
    # データを作ったときのオプションが少し違うので読み込みもそれに合わせる
    ds = xr.open_dataset(file_path, engine='scipy')
    flux = ds.flux.values
    flux_err = ds.flux_err.values
    label = ds.target.values

    flux = np.nan_to_num(flux)

    ds = xr.Dataset({'flux': (['index', 'feature'], flux),
                     'flux_err': (['index', 'feature'], flux_err),
                     'label': (['index'], label)},
                    coords={'index': range(flux.shape[0]),
                            'feature': range(flux.shape[1])})
    return flux, flux_err, label


def load_artificial_train_data(data_path):
    ds = load_data(file_path=data_path)
    return ds


def load_artificial_validation_data(file_path):
    r = re.compile(r'train(?=\.nc)')
    validation_path = r.sub(r'val', file_path)

    ds = load_data(file_path=validation_path)
    return ds


def load_artificial_test_data(file_path):
    r = re.compile(r'train(?=\.nc)')
    validation_path = r.sub(r'test', file_path)

    ds = load_data(file_path=validation_path)
    return ds


# noinspection PyUnusedLocal
def model_fn(features, labels, mode, params, config):
    feature_network = FeatureNetwork()

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = feature_network(features, False, False, False)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    training = mode == tf.estimator.ModeKeys.TRAIN

    # 実際の観測データの統計量を学習するために
    # ラベルありとなしの両方を合わせて処理する
    # ミニバッチでデータを取り出しているせいで実際のデータの分布と
    # 合わない気がするが気にしないでおく
    size_labeled = tf.shape(features['labeled'])[0]
    size_unlabeled = tf.shape(features['unlabeled'])[0]
    tmp = tf.concat([features['labeled'], features['unlabeled']], axis=0)
    tmp = feature_network(tmp, training, True, False)
    labeled_x, unlabeled_x = tf.split(tmp, [size_labeled, size_unlabeled])

    # ラベルあり
    # labeled_x = feature_network(features['labeled'], training, True, False)
    labeled_x_plus = feature_network(features['labeled_plus'],
                                     training, False, True)
    labeled_x_minus = feature_network(features['labeled_minus'],
                                      training, False, True)
    d_labeled_plus = tf.reduce_sum(
        tf.squared_difference(labeled_x, labeled_x_plus),
        axis=1, keepdims=True
    )
    d_labeled_minus = tf.reduce_sum(
        tf.squared_difference(labeled_x, labeled_x_minus),
        axis=1, keepdims=True
    )
    d_labeled = tf.concat([d_labeled_plus, d_labeled_minus], axis=1)
    d_labeled = tf.nn.softmax(d_labeled)
    labeled_cost = tf.square(d_labeled[:, 0])

    # ラベルなし
    # unlabeled_x = feature_network(features['unlabeled'], training,
    #                               False, False)
    unlabeled_reference = [
        feature_network(features['unlabeled{}'.format(i)], training,
                        False, True)
        for i in range(params['n_classes'])
    ]
    d_unlabeled = [
        tf.reduce_sum(tf.squared_difference(unlabeled_x, f),
                      axis=1, keepdims=True)
        for f in unlabeled_reference
    ]
    d_unlabeled = tf.concat(d_unlabeled, axis=1)
    unlabeled_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tf.nn.softmax(d_unlabeled),
        logits=d_unlabeled
    )

    cost = (params['lambda_l'] * tf.reduce_mean(labeled_cost) +
            params['lambda_u'] * tf.reduce_mean(unlabeled_cost))

    accuracy = tf.metrics.accuracy(
        labels=tf.zeros_like(labeled_cost, dtype=tf.int64),
        predictions=tf.argmin(d_labeled, axis=1)
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
        params={'lambda_l': lambda_l, 'lambda_u': lambda_u,
                'n_classes': len(np.unique(train_dataset.labeled_y))}
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
@click.option('--epoch', type=int, default=100)
@click.option('--output-dir', type=str, default='')
@click.option('--artificial-path', type=str,
              default='$HOME/CREST3/datalist/observation/train/'
                      '170810_modified/dataset.classify.train.nc')
@click.option('--observed-path', type=str,
              default='$HOME/CREST3/datalist/observation/test/'
                      '170810_modified/dataset.classify.all.nc')
@click.option('--label-path', type=str,
              default='../../data/external/label_list.dat')
def train_observation(lambda_l, lambda_u, epoch, output_dir, artificial_path,
                      observed_path, label_path):
    artificial_path = os.path.expanduser(os.path.expandvars(artificial_path))
    observed_path = os.path.expanduser(os.path.expandvars(observed_path))
    label_path = os.path.expanduser(os.path.expandvars(label_path))

    print(artificial_path)
    train_ds = load_artificial_train_data(data_path=artificial_path)
    test_ds = load_artificial_test_data(file_path=artificial_path)

    observed_ds = load_observed_data(data_path=observed_path,
                                     label_path=label_path)

    seed = 0
    observed_train_index, observed_test_index = train_test_split(
        list(range(observed_ds.flux.shape[0])),
        random_state=seed, stratify=observed_ds.label, test_size=0.1
    )
    observed_train_ds = xr.Dataset(
        {'flux': (['index', 'feature'],
                  observed_ds.flux[observed_train_index].values),
         'flux_err': (['index', 'feature'],
                      observed_ds.flux_err[observed_train_index].values),
         'label': (['index'], observed_ds.label[observed_train_index].values)},
        coords={'index': range(len(observed_train_index)),
                'feature': range(observed_ds.flux.shape[1])}
    )
    observed_test_ds = xr.Dataset(
        {'flux': (['index', 'feature'],
                  observed_ds.flux[observed_test_index].values),
         'flux_err': (['index', 'feature'],
                      observed_ds.flux_err[observed_test_index].values),
         'label': (['index'], observed_ds.label[observed_test_index].values)},
        coords={'index': range(len(observed_test_index)),
                'feature': range(observed_ds.flux.shape[1])}
    )

    n_train_observed = len(observed_train_index)
    n_class0 = np.count_nonzero(observed_train_ds.label == 0)
    n_class1 = np.count_nonzero(observed_train_ds.label == 1)
    n_others = n_train_observed - n_class0 - n_class1
    print('class 0:', n_class0, n_class0 / n_train_observed * 100)
    print('class 1:', n_class1, n_class1 / n_train_observed * 100)
    print('unlabeled:', n_others, n_others / n_train_observed * 100)

    config = tf.estimator.RunConfig(
        session_config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True)
        ),
        save_summary_steps=1000
    )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=output_dir, config=config,
        params={'lambda_l': lambda_l, 'lambda_u': lambda_u,
                'n_classes': len(np.unique(observed_train_ds.label)) - 1}
    )

    batch_size = 500
    train_size = max(n_class0 + n_class1, n_others)
    test_size = max(np.count_nonzero(observed_test_ds.label >= 0),
                    np.count_nonzero(observed_test_ds.label == -1))

    n = int(math.ceil(train_size * epoch / batch_size / 5000))
    tmp = observed_train_ds.label != -1
    labeled_observed_train_ds = xr.Dataset(
        {'flux': (['index', 'feature'], observed_train_ds.flux[tmp].values),
         'flux_err': (['index', 'feature'],
                      observed_train_ds.flux_err[tmp].values),
         'label': (['index'], observed_train_ds.label[tmp].values)},
        coords={'index': range(len(tmp.values)),
                'feature': range(observed_ds.flux.shape[1])}
    )
    train_input = make_input_fn_mix(
        artificial_data=labeled_observed_train_ds,
        observed_data=observed_train_ds,
        batch_size=batch_size
    )
    evaluation_train, evaluation_test = None, None
    for _ in range(n):
        estimator.train(input_fn=train_input, steps=5000)

        evaluation_train = estimator.evaluate(
            input_fn=train_input,
            steps=int(math.ceil(train_size / batch_size)),
            name='train'
        )
        evaluation_test = estimator.evaluate(
            input_fn=make_input_fn_mix(
                artificial_data=test_ds,
                observed_data=observed_test_ds,
                batch_size=batch_size
            ),
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
                   'seed': seed},
                  f, sort_keys=True, indent=4, cls=NumpyEncoder)


@cmd.command()
@click.option('--lambda_l', type=float, default=1.0)
@click.option('--lambda_u', type=float, default=1.0)
@click.option('--output-dir', type=str, default='')
@click.option('--artificial-path', type=str,
              default='$HOME/CREST3/datalist/observation/train/'
                      '170810_modified/dataset.classify.train.nc')
@click.option('--observed-path', type=str,
              default='$HOME/CREST3/datalist/observation/test/'
                      '170810_modified/dataset.classify.all.nc')
@click.option('--label-path', type=str,
              default='../../data/external/label_list.dat')
def predict_observation(lambda_l, lambda_u, output_dir, artificial_path,
                        observed_path, label_path):
    artificial_path = os.path.expanduser(os.path.expandvars(artificial_path))
    observed_path = os.path.expanduser(os.path.expandvars(observed_path))
    label_path = os.path.expanduser(os.path.expandvars(label_path))

    train_ds = load_artificial_train_data(data_path=artificial_path)
    test_ds = load_artificial_test_data(file_path=artificial_path)

    observed_ds = load_observed_data(
        data_path=observed_path, label_path=label_path
    )

    seed = 0
    observed_train_index, observed_test_index = train_test_split(
        list(range(observed_ds.flux.shape[0])),
        random_state=seed, stratify=observed_ds.label, test_size=0.1
    )
    observed_train_ds = xr.Dataset(
        {'flux': (['index', 'feature'],
                  observed_ds.flux[observed_train_index].values),
         'flux_err': (['index', 'feature'],
                      observed_ds.flux_err[observed_train_index].values),
         'label': (['index'], observed_ds.label[observed_train_index].values)},
        coords={'index': range(len(observed_train_index)),
                'feature': range(observed_ds.flux.shape[1])}
    )
    observed_test_ds = xr.Dataset(
        {'flux': (['index', 'feature'],
                  observed_ds.flux[observed_test_index].values),
         'flux_err': (['index', 'feature'],
                      observed_ds.flux_err[observed_test_index].values),
         'label': (['index'], observed_ds.label[observed_test_index].values)},
        coords={'index': range(len(observed_test_index)),
                'feature': range(observed_ds.flux.shape[1])}
    )

    config = tf.estimator.RunConfig(
        session_config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True)
        ),
        save_summary_steps=1000
    )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=output_dir, config=config,
        params={'lambda_l': lambda_l, 'lambda_u': lambda_u,
                'n_classes': len(np.unique(observed_train_ds.label)) - 1}
    )

    tmp = estimator.predict(
        input_fn=tf.estimator.inputs.numpy_input_fn(
            x=(observed_train_ds.flux.values,
               observed_train_ds.flux_err.values),
            y=observed_train_ds.label.values, shuffle=False
        ),
        yield_single_examples=False
    )
    prediction_train_observed = np.vstack([v for v in tmp])

    tmp = estimator.predict(
        input_fn=tf.estimator.inputs.numpy_input_fn(
            x=(observed_test_ds.flux.values, observed_test_ds.flux_err.values),
            y=observed_test_ds.label.values, shuffle=False
        ),
        yield_single_examples=False
    )
    prediction_test_observed = np.vstack([v for v in tmp])

    d = {'train': (prediction_train_observed, observed_train_ds.label.values),
         'test': (prediction_test_observed, observed_test_ds.label.values)}
    joblib.dump(d, os.path.join(output_dir, 'prediction.pickle'))


@cmd.command()
@click.option('--output-dir', type=str, default='')
@click.option('--perplexity', type=int, default=30)
def visualize(output_dir, perplexity):
    data = joblib.load(os.path.join(output_dir, 'prediction.pickle'))
    input_dims = data['train'][0].shape[1]

    output_dims = 2
    pt_sne = Parametric_tSNE(input_dims, output_dims, perplexity)

    transformed_train = pt_sne.transform(data['train'][0])
    transformed_test = pt_sne.transform(data['test'][0])

    fig, axes = plt.subplots(nrows=2, sharex='all', sharey='all',
                             figsize=(16, 12))
    for ax, x, y in zip(axes, [transformed_train, transformed_test],
                        [data['train'][1], data['test'][1]]):
        for i in np.unique(y):
            tmp = x[y == i]
            ax.scatter(tmp[:, 0], tmp[:, 1], label=i)
        ax.legend(loc='best')
        ax.grid()
    fig.savefig(os.path.join(output_dir, 'observed{}.png'.format(perplexity)),
                bbox_inches='tight')


def main():
    cmd()


if __name__ == '__main__':
    main()
