#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
AUC最適化
positive: 42, 52, 62, 95
negative: 90
クラス90(Ia)が多いので、それを取り除きたい
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
import sklearn.metrics

try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import tfmpl
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import seaborn as sns

from plasticc_a1_classifier4 import map_func, OutputType

__author__ = 'Yasuhiro Imoto'
__date__ = '05/12/2018'


Data = namedtuple(
    'Data',
    ['flux', 'flux_err', 'specz', 'photoz', 'photoz_err', 'target', 'weight',
     'y', 'object_id']
)


def load_data2(data_dir, binary=True):
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
        labels = np.array([t, -1], dtype=np.int32)
        target[original_target != t] = 1
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

    train_data = Data(
        flux=flux[train_index], flux_err=flux_err[train_index],
        specz=specz[train_index], photoz=photoz[train_index],
        photoz_err=photoz_err[train_index], target=y_train, weight=w_train,
        y=y_train, object_id=object_id[train_index]
    )
    test_data = Data(
        flux=flux[test_index], flux_err=flux_err[test_index],
        specz=specz[test_index], photoz=photoz[test_index],
        photoz_err=photoz_err[test_index], target=y_test, weight=w_test,
        y=y_test, object_id=object_id[test_index]
    )

    print(train_count)
    print(test_count)

    return train_data, test_data, labels


def make_dataset(data, shuffle, batch_size, is_training):
    # Ia以外がpositive
    positive = Data._make(v[data.target == 1] for v in data)
    # Ia
    negative = Data._make(v[data.target == 0] for v in data)

    dataset1 = tf.data.Dataset.from_tensor_slices(positive).repeat(-1)
    dataset2 = tf.data.Dataset.from_tensor_slices(negative).repeat(-1)
    if shuffle:
        global_step = tf.train.get_or_create_global_step()
        dataset1 = dataset1.shuffle(1000, seed=global_step)
        dataset2 = dataset2.shuffle(1000, seed=global_step + 1)
    dataset1 = dataset1.map(lambda v: map_func(v, is_training=is_training))
    dataset2 = dataset2.map(lambda v: map_func(v, is_training=is_training))

    dataset = tf.data.Dataset.zip((dataset1, dataset2)).batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


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


class ScoreNetwork(snt.AbstractModule):
    def __init__(self, hidden_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def _build(self, inputs, is_training):
        h = snt.Linear(output_size=self.hidden_size)(inputs)

        for _ in range(self.n_layers - 2):
            h = snt.Sequential([
                snt.LayerNorm(),
                tf.nn.relu,
                snt.Linear(output_size=self.hidden_size)
            ])(h)

        outputs = snt.Sequential([
            snt.LayerNorm(),
            tf.nn.relu,
            snt.Linear(output_size=1, use_bias=False)
        ])(h)

        return outputs


class AUC(snt.AbstractModule):
    def __init__(self):
        super().__init__()

    def _build(self, positive_scores, negative_scores, is_training):
        # 縦がpositive, 横がnegative
        d = tf.reshape(positive_scores, [-1, 1]) - tf.squeeze(negative_scores)

        s = tf.nn.sigmoid(d)
        n_positive = tf.to_float(tf.shape(positive_scores)[0])
        n_negative = tf.to_float(tf.shape(negative_scores)[0])
        auc = tf.reduce_sum(s) / (n_positive * n_negative)

        return auc


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--data-dir', type=click.Path(exists=True))
@click.option('--model-dir', type=click.Path())
@click.option('--batch-size', type=int, default=500)
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

    _, train_data_size = np.unique(train_data.target, return_counts=True)
    # 1epoch辺りの反復回数
    iterations = (max(train_data_size) + batch_size - 1) // batch_size

    with tf.Graph().as_default() as graph:
        model = ScoreNetwork(hidden_size=hidden_size, n_layers=n_layers)

        train_ops = build_train_operators(
            model=model, train_data=train_data, batch_size=batch_size,
            labels=labels
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

            sess.run(train_ops.initialize)
            for _ in trange(step, epochs):
                for _ in range(iterations):
                    sess.run([train_ops.optimize, train_ops.update])
                summary, step = sess.run([train_ops.summary, count_op])
                writer.add_summary(summary=summary, global_step=step)
                sess.run(train_ops.reset)

                sess.run(test_ops.initialize)
                test_score = []
                test_label = []
                while True:
                    try:
                        tmp = sess.run([test_ops.score, test_ops.label])
                        test_score.append(tmp[0])
                        test_label.append(tmp[1])
                    except tf.errors.OutOfRangeError:
                        break
                test_score = np.hstack(test_score)
                test_label = np.hstack(test_label)
                roc_auc = sklearn.metrics.roc_auc_score(
                    y_true=test_label, y_score=test_score
                )
                summary = sess.run(
                    test_ops.summary,
                    feed_dict={
                        test_ops.ph_score: test_score,
                        test_ops.ph_label: test_label,
                        test_ops.ph_auc: roc_auc
                    }
                )
                writer.add_summary(summary=summary, global_step=step)

                saver.save(sess=sess, save_path=str(model_dir / 'model'),
                           global_step=global_step, write_meta_graph=False)


def build_train_operators(model, train_data, batch_size, labels):
    iterator, next_element = make_dataset(
        data=train_data, shuffle=True, batch_size=batch_size,
        is_training=True
    )

    positive_score = model(next_element[0].x, is_training=True)
    negative_score = model(next_element[1].x, is_training=True)

    auc_op = AUC()(positive_score, negative_score, is_training=True)

    optimizer = tf.train.AdamOptimizer()
    opt_op = optimizer.minimize(-auc_op)

    with tf.variable_scope('train_metrics') as vs:
        mean_auc = tf.metrics.mean(auc_op)

        local_variables = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES
        )
        reset_op = tf.variables_initializer(local_variables)

    update_ops = mean_auc[1]

    summary_op = tf.summary.merge([
        tf.summary.scalar('train/auc', mean_auc[0])
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
    logits = tf.squeeze(logits)
    targets = tf.squeeze(next_element.y)

    ph_score = tf.placeholder(dtype=tf.float32, shape=[None])
    ph_label = tf.placeholder(dtype=tf.int32, shape=[None])
    roc_curve = draw_roc_curve(ph_score, ph_label, labels)

    ph_auc = tf.placeholder(dtype=tf.float32, shape=[])

    # 閾値が良くない
    tmp = tf.stack([-ph_score, ph_score], axis=1)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=ph_label, logits=tmp
    )
    l0 = tf.reduce_mean(tf.boolean_mask(loss, tf.equal(ph_label, 0)))
    l1 = tf.reduce_mean(tf.boolean_mask(loss, tf.equal(ph_label, 1)))
    log_loss = (l0 + l1) * 0.5

    # 閾値が良くない
    accuracy = tf.reduce_mean(tf.to_int32(tf.equal(
        tf.greater_equal(ph_score, 0),
        tf.cast(ph_label, tf.bool)
    )))

    summary_op = tf.summary.merge([
        tf.summary.scalar('test/auc', ph_auc),
        tf.summary.scalar('test/loss', log_loss),
        tf.summary.scalar('test/accuracy', accuracy),
        tf.summary.image('test/roc_curve', roc_curve)
    ])

    TestOperators = namedtuple(
        'TestOperators',
        ['summary', 'initialize', 'score', 'label', 'ph_score', 'ph_label',
         'ph_auc']
    )
    ops = TestOperators(
        summary=summary_op,
        initialize=iterator.initializer, score=logits, label=targets,
        ph_score=ph_score, ph_label=ph_label, ph_auc=ph_auc
    )

    return ops


@tfmpl.figure_tensor
def draw_roc_curve(prediction, target, labels):
    fig = tfmpl.create_figure(figsize=(9, 7))
    ax = fig.add_subplot(111)

    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true=target, y_score=prediction)
    roc_auc = sklearn.metrics.roc_auc_score(y_true=target, y_score=prediction)
    ax.plot(fpr, tpr, label='AUC={}'.format(roc_auc))
    ax.plot([0, 1], [0, 1], color='black', linestyle=':')
    ax.set_xlim((-0.01, 1.01))
    ax.set_ylim((-0.01, 1.01))
    ax.grid()

    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.legend(loc='lower right')

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

    train_data, test_data, labels = load_data2(
        data_dir=data_dir, binary=binary
    )

    with tf.Graph().as_default() as graph:
        model = ScoreNetwork(hidden_size=hidden_size, n_layers=n_layers)

        train_iterator, train_element = make_dataset2(
            data=train_data, count=1, shuffle=False, batch_size=batch_size,
            is_training=False
        )
        test_iterator, test_element = make_dataset2(
            data=test_data, count=1, shuffle=False, batch_size=batch_size,
            is_training=False
        )

        train_logits = model(train_element.x, is_training=False)
        train_logits = tf.squeeze(train_logits)
        test_logits = model(test_element.x, is_training=False)
        test_logits = tf.squeeze(test_logits)

        global_step = tf.train.get_or_create_global_step()

        saver = tf.train.Saver()

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=config, graph=graph) as sess:
            checkpoint = tf.train.get_checkpoint_state(model_dir)
            path = checkpoint.model_checkpoint_path
            saver.restore(sess=sess, save_path=path)

            step = sess.run(global_step)

            run_prediction(
                logits=train_logits, y=train_element.y,
                object_id=train_element.object_id, iterator=train_iterator,
                sess=sess, output_path=model_dir / 'train{}.csv'.format(step)
            )
            run_prediction(
                logits=test_logits, y=test_element.y,
                object_id=test_element.object_id, iterator=test_iterator,
                sess=sess, output_path=model_dir / 'test{}.csv'.format(step)
            )


def run_prediction(logits, y, object_id, iterator, sess, output_path):
    sess.run(iterator.initializer)
    results = []
    index = []
    targets = []
    while True:
        try:
            tmp = sess.run([logits, y, object_id])
            results.append(tmp[0])
            targets.append(tmp[1])
            index.append(tmp[2])
        except tf.errors.OutOfRangeError:
            break
    results = np.hstack(results)
    targets = np.hstack(targets)
    index = np.hstack(index)

    data = np.empty((len(results), 2))
    data[:, 0] = results
    data[:, 1] = targets
    df = pd.DataFrame(
        data=data, index=index,
        columns=['score', 'label']
    )
    df.sort_index(axis=1, inplace=True)
    df.to_csv(str(output_path), index_label='object_id')


def main():
    cmd()


if __name__ == '__main__':
    main()
