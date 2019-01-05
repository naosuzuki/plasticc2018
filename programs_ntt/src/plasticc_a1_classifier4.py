#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
入力を統計量に変換したものを使う
https://github.com/LSSTDESC/plasticc-kit/blob/master/plasticc_classification_demo.ipynb
の特徴量
"""

import json
import warnings
from collections import OrderedDict, namedtuple, Counter
from operator import itemgetter
from pathlib import Path

import cesium.featurize as featurize
import click
import numpy as np
import pandas as pd
import sonnet as snt
import tensorflow as tf
import xarray as xr
from astropy.table import Table
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from tqdm import trange

try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import tfmpl
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import seaborn as sns

__author__ = 'Yasuhiro Imoto'
__date__ = '30/11/2018'


Data = namedtuple(
    'Data', ['x', 'y', 'original_x', 'original_y', 'weight', 'object_id']
)
Data2 = namedtuple(
    'Data2',
    ['flux', 'flux_err', 'specz', 'photoz', 'photoz_err', 'target', 'weight',
     'y', 'object_id']
)


def load_data2(data_dir, binary):
    ds = xr.open_dataset(str(data_dir / 'train.nc'))

    flux = np.hstack(
        [ds['flux{}'.format(i)].values for i in range(6)]
    ).astype(np.float32)
    flux_err = np.hstack(
        [ds['flux_err{}'.format(i)].values for i in range(6)]
    ).astype(np.float32)
    specz = ds['hostgal_specz'].values.astype(np.float32)
    photoz = ds['hostgal_photoz'].values.astype(np.float32)
    photoz_err = ds['hostgal_photoz_err'].values.astype(np.float32)
    target = ds['target'].values

    # 超新星のクラスのみを選ぶ
    sn_classes = (42, 52, 62, 90, 95)
    # スライディングウィンドウでデータを作っているので、object_idで分割する
    train_index = np.zeros_like(target, dtype=np.bool)
    test_index = np.zeros_like(train_index)
    for c in sn_classes:
        # 個数は気にしない
        object_id = np.unique(ds.object_id[target == c])
        train_id, test_id = train_test_split(
            object_id, test_size=0.3, random_state=c
        )

        for i in train_id:
            train_index = np.logical_or(train_index, ds.object_id == i)
        for i in test_id:
            test_index = np.logical_or(test_index, ds.object_id == i)

    counts = Counter(target[np.logical_or(train_index, test_index)])
    labels, values = zip(*sorted(counts.items(), key=itemgetter(1)))

    original_target = target.astype(np.int32)
    target = np.zeros_like(original_target)
    if binary:
        t = 90
        labels = np.array([-1, t], dtype=np.int32)
        target[original_target == t] = 1
    else:
        for i, t in enumerate(labels):
            target[original_target == t] = i

    y_train = target[train_index]
    _, train_count = np.unique(y_train, return_counts=True)
    w_train = np.empty_like(y_train, dtype=np.float32)
    for i, c in enumerate(train_count):
        w_train[y_train == i] = 1.0 / c

    y_test = target[test_index]
    _, test_count = np.unique(y_test, return_counts=True)
    w_test = np.empty_like(y_test, dtype=np.float32)
    for i, c in enumerate(test_count):
        w_test[y_test == i] = 1.0 / c

    object_id = ds['object_id'].values.astype(np.int32)

    train_data = Data2(
        flux=flux[train_index], flux_err=flux_err[train_index],
        specz=specz[train_index], photoz=photoz[train_index],
        photoz_err=photoz_err[train_index], target=y_train, weight=w_train,
        y=y_train, object_id=object_id[train_index]
    )
    test_data = Data2(
        flux=flux[test_index], flux_err=flux_err[test_index],
        specz=specz[test_index], photoz=photoz[test_index],
        photoz_err=photoz_err[test_index], target=y_test, weight=w_test,
        y=y_test, object_id=object_id[test_index]
    )

    print(train_count)
    print(test_count)

    return train_data, test_data, labels


def load_data(data_dir):
    pb_map = OrderedDict(
        [(0, 'u'), (1, 'g'), (2, 'r'), (3, 'i'), (4, 'z'), (5, 'y')]
    )

    feature_table, _ = featurize.load_featureset(
        str(data_dir / 'plasticc_featuretable.npz')
    )

    old_names = feature_table.columns.values
    new_names = ['{}_{}'.format(x, pb_map.get(y, 'meta'))
                 for x, y in old_names]
    cols = [feature_table[col] for col in old_names]
    all_features = Table(cols, names=new_names)

    meta_data = Table.read(str(data_dir / 'training_set_metadata.csv'),
                           format='csv')

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.3, random_state=42
    )
    splits = list(splitter.split(all_features, meta_data['target']))[0]
    train_index, test_index = splits

    counts = Counter(meta_data['target'])
    labels, values = zip(*sorted(counts.items(), key=itemgetter(1)))

    original_target = np.asarray(meta_data['target'].tolist())
    # 0-based index
    target = np.zeros_like(original_target)
    for i, t in enumerate(labels):
        target[original_target == t] = i

    x_train = np.asarray(all_features[train_index].as_array().tolist())
    y_train = target[train_index]
    _, train_count = np.unique(y_train, return_counts=True)
    w_train = np.empty_like(y_train, dtype=np.float32)
    for i, c in enumerate(train_count):
        w_train[y_train == i] = 1.0 / c

    x_test = np.asarray(all_features[test_index].as_array().tolist())
    y_test = target[test_index]
    _, test_count = np.unique(y_test, return_counts=True)
    w_test = np.empty_like(y_test, dtype=np.float32)
    for i, c in enumerate(train_count):
        w_test[y_test == i] = 1.0 / c

    n_columns = len(new_names)
    n_pca_features = (n_columns - 3) // len(pb_map) + 3

    pca = PCA(
        n_components=n_pca_features, whiten=True, svd_solver='full',
        random_state=42
    )

    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    object_id = meta_data.indices

    train_data = Data(
        x=x_train_pca, y=y_train, original_x=x_train,
        original_y=original_target[train_index], weight=w_train,
        object_id=object_id[train_index]
    )
    test_data = Data(
        x=x_test_pca, y=y_test, original_x=x_test,
        original_y=original_target[test_index], weight=w_test,
        object_id=object_id[test_index]
    )

    return train_data, test_data, labels


OutputType = namedtuple('OutputType', ['x', 'y', 'weight', 'object_id'])


def make_dataset2(data, count, shuffle, batch_size, is_training):
    dataset = tf.data.Dataset.from_tensor_slices(data).repeat(count)
    if shuffle:
        global_step = tf.train.get_or_create_global_step()
        dataset = dataset.shuffle(1000, seed=global_step)
    dataset = dataset.map(
        lambda v: map_func(data=v, is_training=is_training)
    ).batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def map_func(data, is_training):
    (flux, flux_err, specz, photoz, photoz_err, target, weight,
     _, object_id) = data

    if is_training:
        tmp_flux = flux + flux_err * tf.random_normal(shape=tf.shape(flux))
    else:
        tmp_flux = flux
    magnitude = tf.asinh(tmp_flux * 0.5)

    if is_training:
        tmp_specz = tf.to_float(
            tf.multinomial(logits=[[0.0, 0.0]], num_samples=1)
        ) * specz
        tmp_photoz = photoz + photoz_err * tf.random_normal(shape=[])
    else:
        tmp_specz = specz
        tmp_photoz = photoz

    x = tf.concat(
        [magnitude, tf.reshape(tmp_specz, [1]), tf.reshape(tmp_photoz, [1])],
        axis=0
    )
    return OutputType(x=x, y=target, weight=weight, object_id=object_id)


def map_func_mixup2(data1, data2, n_classes, is_training):
    alpha = 2.0
    d = tf.distributions.Beta(alpha, alpha)
    r = d.sample()

    v1 = map_func(data=data1, is_training=is_training)
    v2 = map_func(data=data2, is_training=is_training)

    x = r * v1.x + (1 - r) * v2.x
    y = tf.squeeze(r * tf.one_hot(v1.y, n_classes) +
                   (1 - r) * tf.one_hot(v2.y, n_classes))
    weight = r * v1.weight + (1 - r) * v2.weight

    return OutputType(x=x, y=y, weight=weight,
                      object_id=(v1.object_id, v2.object_id))


def make_dataset_mixup2(data1, data2, count, batch_size, n_classes,
                        is_training):
    global_step = tf.train.get_or_create_global_step()
    ds1 = tf.data.Dataset.from_tensor_slices(data1).repeat(count)
    ds1 = ds1.shuffle(1000, seed=global_step)

    ds2 = tf.data.Dataset.from_tensor_slices(data2).repeat(-1)
    ds2 = ds2.shuffle(1000, seed=global_step + 1)

    dataset = tf.data.Dataset.zip((ds1, ds2))
    dataset = dataset.map(
        lambda v1, v2: map_func_mixup2(
            v1, v2, n_classes=n_classes, is_training=is_training
        )
    ).batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def make_dataset(data, count, shuffle, batch_size):
    tmp = OutputType(x=data.x.astype(np.float32), y=data.y,
                     weight=data.weight.astype(np.float32),
                     object_id=data.object_id.astype(np.float32))

    dataset = tf.data.Dataset.from_tensor_slices(tmp)
    dataset = dataset.repeat(count=count)
    if shuffle:
        global_step = tf.train.get_or_create_global_step()
        dataset = dataset.shuffle(1000, seed=global_step)
    dataset = dataset.batch(batch_size=batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def make_dataset_mixup(data1, data2, count, batch_size, n_classes):
    tmp1 = OutputType(x=data1.x.astype(np.float32), y=data1.y,
                      weight=data1.weight.astype(np.float32),
                      object_id=data1.object_id.astype(np.int32))
    tmp2 = OutputType(x=data2.x.astype(np.float32), y=data2.y,
                      weight=data2.weight.astype(np.float32),
                      object_id=data2.object_id.astype(np.int32))

    global_step = tf.train.get_or_create_global_step()
    ds1 = tf.data.Dataset.from_tensor_slices(tmp1).repeat(count)
    ds1 = ds1.shuffle(1000, seed=global_step)

    ds2 = tf.data.Dataset.from_tensor_slices(tmp2).repeat(-1)
    ds2 = ds2.shuffle(1000, seed=global_step + 1)

    dataset = tf.data.Dataset.zip((ds1, ds2))
    dataset = dataset.map(
        lambda v1, v2: map_func_mixup(v1, v2, n_classes=n_classes)
    ).batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def map_func_mixup(data1, data2, n_classes):
    alpha = 2.0
    d = tf.distributions.Beta(alpha, alpha)
    r = d.sample()

    x = r * data1.x + (1 - r) * data2.x
    y = tf.squeeze(r * tf.one_hot(data1.y, n_classes) +
                   (1 - r) * tf.one_hot(data2.y, n_classes))
    weight = r * data1.weight + (1 - r) * data2.weight

    return OutputType(x=x, y=y, weight=weight,
                      object_id=(data1.object_id, data2.object_id))


class Classifier(snt.AbstractModule):
    def __init__(self, n_classes, hidden_size, n_layers):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.update_collection_name = 'classifier_bn'

    def _build(self, inputs, is_training):
        h = snt.Linear(output_size=self.hidden_size)(inputs)

        for _ in range(self.n_layers - 2):
            h = snt.Sequential([
                snt.BatchNormV2(
                    data_format='NC', scale=False,
                    update_ops_collection=self.update_collection_name
                ),
                tf.nn.relu,
                snt.Linear(output_size=self.hidden_size)
            ])(h, is_training)

        outputs = snt.Sequential([
            snt.BatchNormV2(
                data_format='NC', scale=False,
                update_ops_collection=self.update_collection_name
            ),
            tf.nn.relu,
            snt.Linear(output_size=self.n_classes)
        ])(h, is_training)

        return outputs


class MixFeatLayer(snt.AbstractModule):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def _build(self, inputs, is_training):
        r = tf.random_normal(shape=[], stddev=self.sigma)
        pi = tf.constant(np.pi, dtype=tf.float32)
        theta = tf.random_uniform(shape=[], minval=-pi, maxval=pi)

        a = r * tf.cos(theta)
        b = r * tf.sin(theta)

        if is_training:
            index = tf.range(tf.shape(inputs)[0])
            index = tf.reshape(tf.random_shuffle(index), [-1, 1])

            outputs = inputs + a * inputs + b * tf.gather_nd(inputs, index)
        else:
            outputs = inputs

        return outputs


class MixFeatClassifier(snt.AbstractModule):
    def __init__(self, n_classes, hidden_size, n_layers, sigma):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.sigma = sigma
        self.update_collection_name = 'classifier_bn'

    def _build(self, inputs, is_training):
        h = snt.Linear(output_size=self.hidden_size)(inputs)
        h = MixFeatLayer(sigma=self.sigma)(h, is_training=is_training)

        for _ in range(self.n_layers - 2):
            h = snt.Sequential([
                snt.BatchNormV2(
                    data_format='NC', scale=False,
                    update_ops_collection=self.update_collection_name
                ),
                tf.nn.relu,
                snt.Linear(output_size=self.hidden_size)
            ])(h, is_training)
            h = MixFeatLayer(sigma=self.sigma)(h, is_training=is_training)

        outputs = snt.Sequential([
            snt.BatchNormV2(
                data_format='NC', scale=False,
                update_ops_collection=self.update_collection_name
            ),
            tf.nn.relu,
            snt.Linear(output_size=self.n_classes)
        ])(h, is_training)

        return outputs


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--data-dir', type=click.Path(exists=True))
@click.option('--model-dir', type=click.Path())
@click.option('--batch-size', type=int, default=1000)
@click.option('--epochs', type=int, default=100)
@click.option('--seed', type=int, default=42)
@click.option('--cv', type=int, default=0)
@click.option('--mixup', is_flag=True)
@click.option('--hidden-size', type=int, default=100)
@click.option('--n-layers', type=int, default=4)
@click.option('--binary', is_flag=True)
@click.option('--mixfeat', is_flag=True)
@click.option('--sigma', type=float, default=0.1,
              help='a parameter of mixfeat')
def train(data_dir, model_dir, batch_size, epochs, seed, cv, mixup,
          hidden_size, n_layers, binary, mixfeat, sigma):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    parameters = {
        'data': {'path': data_dir},
        'seed': seed, 'cv': cv, 'mixup': mixup, 'hidden_size': hidden_size,
        'n_layers': n_layers, 'binary': binary,
        'mixfeat': {'use': mixfeat, 'sigma': sigma}
    }
    with (model_dir / 'parameters.json').open('w') as f:
        json.dump(parameters, f, sort_keys=True, indent=4)

    data_dir = Path(data_dir)
    train_data, test_data, labels = load_data2(
        data_dir=data_dir, binary=binary
    )

    with tf.Graph().as_default() as graph:
        if mixfeat:
            model = MixFeatClassifier(
                n_classes=len(np.unique(train_data.y)),
                hidden_size=hidden_size, n_layers=n_layers, sigma=sigma
            )
        else:
            model = Classifier(
                n_classes=len(np.unique(train_data.y)),
                hidden_size=hidden_size,
                n_layers=n_layers
            )

        train_ops = build_train_operators(
            model=model, train_data=train_data, batch_size=batch_size,
            labels=labels, mixup=mixup
        )
        test_ops = build_test_operators(
            model=model, test_data=test_data, batch_size=batch_size,
            labels=labels
        )

        global_step = tf.train.get_or_create_global_step()
        count_op = tf.assign_add(global_step, 1)

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(str(model_dir))

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

                sess.run(test_ops.initialize)
                while True:
                    try:
                        sess.run(test_ops.update)
                    except tf.errors.OutOfRangeError:
                        break
                summary = sess.run(test_ops.summary)
                writer.add_summary(summary=summary, global_step=step)
                sess.run(test_ops.reset)

                saver.save(sess=sess, save_path=str(model_dir / 'model'),
                           global_step=global_step, write_meta_graph=False)


def build_train_operators(model, train_data, batch_size, labels, mixup):
    if mixup:
        # tmp = (v[train_data.y < model.n_classes - 1] for v in train_data)
        # data2 = Data._make(tmp)

        iterator, next_element = make_dataset_mixup2(
            data1=train_data, data2=train_data, count=1, batch_size=batch_size,
            n_classes=model.n_classes, is_training=True
        )

        logits = model(next_element.x, is_training=True)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(next_element.y), logits=logits
        )
        weighted_loss = loss * next_element.weight

        y = tf.argmax(next_element.y, axis=1)
    else:
        iterator, next_element = make_dataset2(
            data=train_data, count=1, shuffle=True, batch_size=batch_size,
            is_training=True
        )

        logits = model(next_element.x, is_training=True)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=next_element.y, logits=logits
        )
        weighted_loss = loss * next_element.weight

        y = next_element.y

    optimizer = tf.train.AdamOptimizer()
    update_ops = tf.get_collection(model.update_collection_name)
    with tf.control_dependencies(update_ops):
        opt_op = optimizer.minimize(tf.reduce_mean(weighted_loss))

    with tf.variable_scope('train_metrics') as vs:
        total = tf.get_local_variable('total', shape=[], dtype=tf.float32,
                                      initializer=tf.zeros_initializer())
        update_total = tf.assign_add(total, tf.reduce_sum(weighted_loss))

        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.metrics.accuracy(
            labels=y, predictions=predictions
        )

        cm = tf.confusion_matrix(
            labels=y, predictions=predictions,
            num_classes=model.n_classes
        )
        confusion_matrix = tf.get_local_variable(
            'cm', shape=[model.n_classes, model.n_classes], dtype=tf.int32,
            initializer=tf.zeros_initializer()
        )

        local_variables = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES
        )
        reset_op = tf.variables_initializer(local_variables)

    update_cm = tf.assign_add(confusion_matrix, cm)
    cm_image = draw_confusion_matrix(confusion_matrix, labels=labels)

    update_ops = tf.group(update_total, accuracy[1], update_cm)

    summary_op = tf.summary.merge([
        tf.summary.scalar('train/loss', total / tf.to_float(model.n_classes)),
        tf.summary.scalar('train/accuracy', accuracy[0]),
        tf.summary.image('train/confusion_matrix', cm_image)
    ])

    TrainOperators = namedtuple(
        'TrainOperators',
        ['optimize', 'update', 'summary', 'reset', 'initialize']
    )
    ops = TrainOperators(
        optimize=opt_op, update=update_ops, summary=summary_op,
        reset=reset_op, initialize=iterator.initializer
    )

    return ops


def build_test_operators(model, test_data, batch_size, labels):
    iterator, next_element = make_dataset2(
        data=test_data, count=1, shuffle=False, batch_size=batch_size,
        is_training=False
    )

    logits = model(next_element.x, is_training=False)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=next_element.y, logits=logits
    )
    weighted_loss = loss * next_element.weight

    with tf.variable_scope('test_metrics') as vs:
        total = tf.get_local_variable('total', shape=[], dtype=tf.float32,
                                      initializer=tf.zeros_initializer())
        update_total = tf.assign_add(total, tf.reduce_sum(weighted_loss))

        accuracy = tf.metrics.accuracy(
            labels=next_element.y, predictions=tf.argmax(logits, axis=1)
        )

        cm = tf.confusion_matrix(
            labels=next_element.y, predictions=tf.argmax(logits, axis=1),
            num_classes=model.n_classes
        )
        confusion_matrix = tf.get_local_variable(
            'cm', shape=[model.n_classes, model.n_classes], dtype=tf.int32,
            initializer=tf.zeros_initializer()
        )

        local_variables = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES
        )
        reset_op = tf.variables_initializer(local_variables)

    update_cm = tf.assign_add(confusion_matrix, cm)
    cm_image = draw_confusion_matrix(confusion_matrix, labels=labels)

    update_ops = tf.group(update_total, accuracy[1], update_cm)

    summary_op = tf.summary.merge([
        tf.summary.scalar('test/loss', total / model.n_classes),
        tf.summary.scalar('test/accuracy', accuracy[0]),
        tf.summary.image('test/confusion_matrix', cm_image)
    ])

    TestOperators = namedtuple(
        'TestOperators',
        ['update', 'summary', 'reset', 'initialize']
    )
    ops = TestOperators(
        update=update_ops, summary=summary_op,
        reset=reset_op, initialize=iterator.initializer
    )

    return ops


@tfmpl.figure_tensor
def draw_confusion_matrix(confusion_matrix, labels):
    fig = tfmpl.create_figure(figsize=(9, 7))
    ax = fig.add_subplot(111)

    cm = confusion_matrix / np.sum(confusion_matrix, axis=1, keepdims=True)
    annotation = np.around(cm, 2)

    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap='Blues',
                annot=annotation, lw=0.5, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_aspect('equal')

    return fig


@cmd.command()
@click.option('--model-dir', type=click.Path())
@click.option('--batch-size', type=int, default=1000)
def predict(model_dir, batch_size):
    model_dir = Path(model_dir)
    with (model_dir / 'parameters.json').open('r') as f:
        parameters = json.load(f)

    data_dir = Path(parameters['data']['path'])
    # seed = parameters['seed']
    hidden_size = parameters.get('hidden_size', 100)
    n_layers = parameters.get('n_layers', 4)
    binary = parameters.get('binary', False)
    mixfeat = parameters['mixfeat']['use']
    sigma = parameters['mixfeat']['sigma']

    train_data, test_data, labels = load_data2(
        data_dir=data_dir, binary=binary
    )

    with tf.Graph().as_default() as graph:
        if mixfeat:
            model = MixFeatClassifier(
                n_classes=len(np.unique(train_data.y)),
                hidden_size=hidden_size, n_layers=n_layers, sigma=sigma
            )
        else:
            model = Classifier(
                n_classes=len(np.unique(train_data.y)),
                hidden_size=hidden_size,
                n_layers=n_layers
            )

        train_iterator, train_element = make_dataset2(
            data=train_data, count=1, shuffle=False, batch_size=batch_size,
            is_training=False
        )
        test_iterator, test_element = make_dataset2(
            data=test_data, count=1, shuffle=False, batch_size=batch_size,
            is_training=False
        )

        train_logits = model(train_element.x, is_training=False)
        p_train = tf.clip_by_value(
            tf.nn.softmax(train_logits, axis=-1),
            clip_value_min=1e-15, clip_value_max=1 - 1e-15
        )
        p_train = p_train / tf.reduce_sum(p_train, axis=-1, keepdims=True)

        test_logits = model(test_element.x, is_training=False)
        p_test = tf.clip_by_value(
            tf.nn.softmax(test_logits, axis=-1),
            clip_value_min=1e-15, clip_value_max=1 - 1e-15
        )
        p_test = p_test / tf.reduce_sum(p_test, axis=-1, keepdims=True)

        global_step = tf.train.get_or_create_global_step()

        saver = tf.train.Saver()

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=config, graph=graph) as sess:
            checkpoint = tf.train.get_checkpoint_state(model_dir)
            path = checkpoint.model_checkpoint_path
            saver.restore(sess=sess, save_path=path)

            step = sess.run(global_step)

            run_predict(
                p=p_train, y=train_element.y,
                object_id=train_element.object_id, iterator=train_iterator,
                sess=sess, labels=labels,
                output_path=model_dir / 'train{}.csv'.format(step)
            )
            run_predict(
                p=p_test, y=test_element.y,
                object_id=test_element.object_id, iterator=test_iterator,
                sess=sess, labels=labels,
                output_path=model_dir / 'test{}.csv'.format(step)
            )


def run_predict(p, y, object_id, iterator, sess, labels, output_path):
    sess.run(iterator.initializer)
    results = []
    index = []
    targets = []
    while True:
        try:
            tmp = sess.run([p, y, object_id])
            results.append(tmp[0])
            targets.append(tmp[1])
            index.append(tmp[2])
        except tf.errors.OutOfRangeError:
            break
    results = np.vstack(results)
    targets = np.hstack(targets)
    index = np.hstack(index)

    print(compute_log_loss(predictions=results, labels=targets))

    df = pd.DataFrame(
        data=results, index=index,
        columns=['class_{}'.format(c) for c in labels]
    )
    df.sort_index(axis=1, inplace=True)
    df.to_csv(str(output_path), index_label='object_id')


def compute_log_loss(predictions, labels):
    n_classes = len(np.unique(labels))
    w = np.ones(n_classes, dtype=np.float32)
    loss = np.empty_like(w, dtype=np.float32)

    for i in range(n_classes):
        tmp = predictions[labels == i]
        loss[i] = np.average(np.log(tmp[:, i]))
    return -np.average(loss, weights=w)


def main():
    cmd()


if __name__ == '__main__':
    main()
