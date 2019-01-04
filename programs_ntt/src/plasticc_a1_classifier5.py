#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
Between Class Learning (BC-learning)
Learning from Between-class Examples for Deep Sound Recognition
https://github.com/mil-tokyo/bc_learning_sound/

音声信号のクラス分類と非常によく似ている
"""

import json
from collections import namedtuple, Counter
from operator import itemgetter
from pathlib import Path

import click
import numpy as np
import sonnet as snt
import tensorflow as tf
import xarray as xr
from sklearn.model_selection import train_test_split
from tqdm import trange

from plasticc_a1_classifier4 import (draw_confusion_matrix, run_predict)

__author__ = 'Yasuhiro Imoto'
__date__ = '04/12/2018'


Data = namedtuple(
    'Data2',
    ['flux', 'flux_err', 'specz', 'photoz', 'photoz_err', 'target', 'weight',
     'y', 'object_id']
)


def load_data2(data_dir):
    ds = xr.open_dataset(str(data_dir / 'train.nc'))

    shape = ds['flux0'].shape
    flux = np.concatenate(
        [np.reshape(ds['flux{}'.format(i)].values, [-1, 1, 1, shape[-1]])
         for i in range(6)],
        axis=1
    ).astype(np.float32)
    flux_err = np.concatenate(
        [np.reshape(ds['flux_err{}'.format(i)].values, [-1, 1, 1, shape[-1]])
         for i in range(6)],
        axis=1
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


OutputType = namedtuple(
    'OutputType', ['x', 'meta', 'y', 'weight', 'object_id']
)


def make_dataset_bc(data, count, shuffle, batch_size, is_training):
    n_classes = len(np.unique(data.target))
    # クラスごとにデータを分割
    class_data = [
        Data._make([v[data.target == i] for v in data])
        for i in range(n_classes)
    ]
    # i番目のクラス以外
    label_list = np.empty([n_classes, n_classes - 1], dtype=np.int32)
    for i in range(n_classes):
        label_list[i] = [j for j in range(n_classes) if j != i]

    def generator():
        for i in range(len(data[0])):
            data1 = Data._make(v[i] for v in data)
            label1 = data.target[i]

            label2 = int(np.random.choice(label_list[label1], size=1))
            # 一様乱数でサンプルを選択
            j = int(np.random.choice(len(class_data[label2][0]), size=1))
            data2 = Data._make(v[j] for v in class_data[label2])

            yield data1, data2

    types = (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
             tf.int32, tf.float32, tf.int32, tf.int32)
    size = data.flux.shape[-1]
    shapes = (tf.TensorShape([6, 1, size]), tf.TensorShape([6, 1, size]),
              tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]),
              tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]),
              tf.TensorShape([]))
    dataset = tf.data.Dataset.from_generator(
        generator=generator, output_types=(types, types),
        output_shapes=(shapes, shapes)
    )

    if shuffle:
        global_step = tf.train.get_or_create_global_step()
        dataset = dataset.shuffle(1000, seed=global_step)
    dataset = dataset.map(
        lambda v1, v2: map_func_bc(
            data1=v1, data2=v2, n_classes=n_classes, is_training=is_training
        )
    ).batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def map_func_bc(data1, data2, n_classes, is_training):
    v1, g1 = _map_func_bc(data=data1, is_training=is_training)
    v2, g2 = _map_func_bc(data=data2, is_training=is_training)

    alpha = 2.0
    d = tf.distributions.Beta(alpha, alpha)
    r = d.sample()

    # p = 1.0 / (1.0 + tf.pow(10.0, (g1 - g2) / 20) * (1 - r) / (r + 1e-9))
    # x = (p * v1.x + (1 - p) * v2.x) / tf.sqrt(tf.square(p) + tf.square(1 - p))

    x = r * v1.x + (1 - r) * v2.x
    y = r * tf.one_hot(v1.y, n_classes) + (1 - r) * tf.one_hot(v2.y, n_classes)
    meta = r * v1.meta + (1 - r) * v2.meta
    # weight = r * v1.weight + (1 - r) * v2.weight
    weight = tf.maximum(v1.weight, v2.weight)

    return OutputType(x=x, meta=meta, y=y, weight=weight,
                      object_id=(v1.object_id, v2.object_id))


def _map_func_bc(data, is_training):
    (flux, flux_err, specz, photoz, photoz_err, target, weight,
     _, object_id) = data

    if is_training:
        tmp_flux = flux + flux_err * tf.random_normal(shape=tf.shape(flux))
    else:
        tmp_flux = flux
    gain = tf.reduce_mean(tf.reduce_sum(tf.square(tmp_flux), axis=-1))
    # デシベル
    gain_db = 10.0 * tf.log(gain) / tf.log(10.0)
    magnitude = tf.asinh(tmp_flux * 0.5)

    if is_training:
        tmp_specz = tf.to_float(
            tf.multinomial(logits=[[0.0, 0.0]], num_samples=1)
        ) * specz
        tmp_photoz = photoz + photoz_err * tf.random_normal(shape=[])
    else:
        tmp_specz = specz
        tmp_photoz = photoz

    x = magnitude
    meta = tf.concat(
        [tf.reshape(tmp_specz, shape=[1]), tf.reshape(tmp_photoz, [1])],
        axis=0
    )
    return OutputType(x=x, meta=meta, y=target, weight=weight,
                      object_id=object_id), gain_db


def make_dataset(data, count, shuffle, batch_size, is_training):
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

    x = magnitude
    meta = tf.concat(
        [tf.reshape(tmp_specz, shape=[1]), tf.reshape(tmp_photoz, [1])],
        axis=0
    )
    return OutputType(x=x, meta=meta, y=target, weight=weight,
                      object_id=object_id)


class Block(snt.AbstractModule):
    def __init__(self, output_channels, kernel_shape, stride=(1, 1),
                 update_collection_name='env_net_v2_bn'):
        super().__init__()
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.update_collection_name = update_collection_name

    def _build(self, inputs, is_training):
        h = snt.Conv2D(
            output_channels=self.output_channels,
            kernel_shape=self.kernel_shape, stride=self.stride,
            initializers={'w': tf.keras.initializers.he_normal()},
            data_format='NCHW', use_bias=False
        )(inputs)
        h = snt.BatchNormV2(
            data_format='NCHW',
            update_ops_collection=self.update_collection_name
        )(h, is_training=is_training)
        outputs = tf.nn.relu(h)

        return outputs


class EnvNetV2(snt.AbstractModule):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.update_collection_name = 'env_net_v2_bn'

    def _build(self, inputs, meta, is_training):
        h = Block(
            output_channels=32, kernel_shape=(1, 3),
            update_collection_name=self.update_collection_name
        )(inputs, is_training=is_training)
        h = Block(
            output_channels=64, kernel_shape=(1, 3),
            update_collection_name=self.update_collection_name
        )(h, is_training=is_training)

        shape = h.get_shape()
        h = snt.BatchReshape(shape=[1, shape[1].value, shape[3].value])(h)

        h = Block(
            output_channels=32, kernel_shape=(3, 3),
            update_collection_name=self.update_collection_name
        )(h, is_training=is_training)
        h = Block(
            output_channels=32, kernel_shape=(3, 3),
            update_collection_name=self.update_collection_name
        )(h, is_training=is_training)
        h = tf.nn.max_pool(
            h, ksize=(1, 1, 2, 2), strides=(1, 1, 2, 2), padding=snt.SAME,
            data_format='NCHW'
        )

        h = Block(
            output_channels=64, kernel_shape=(3, 3),
            update_collection_name=self.update_collection_name
        )(h, is_training=is_training)
        h = Block(
            output_channels=64, kernel_shape=(3, 3),
            update_collection_name=self.update_collection_name
        )(h, is_training=is_training)
        h = tf.nn.max_pool(
            h, ksize=(1, 1, 2, 2), strides=(1, 1, 2, 2), padding=snt.SAME,
            data_format='NCHW'
        )

        h = snt.BatchFlatten()(h)
        h = tf.concat([h, meta], axis=-1)
        h = snt.Linear(output_size=1024)(h)
        h = tf.nn.relu(h)
        h = tf.layers.dropout(h, rate=0.5, training=is_training)

        h = snt.Linear(output_size=1024)(h)
        h = tf.nn.relu(h)
        h = tf.layers.dropout(h, rate=0.5, training=is_training)

        outputs = snt.Linear(output_size=self.n_classes)(h)

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
@click.option('--bc', is_flag=True)
def train(data_dir, model_dir, batch_size, epochs, seed, cv, bc):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    parameters = {
        'data': {'path': data_dir},
        'seed': seed, 'cv': cv, 'bc': bc
    }
    with (model_dir / 'parameters.json').open('w') as f:
        json.dump(parameters, f, sort_keys=True, indent=4)

    data_dir = Path(data_dir)
    train_data, test_data, labels = load_data2(data_dir=data_dir)

    with tf.Graph().as_default() as graph:
        model = EnvNetV2(n_classes=len(np.unique(train_data.y)))

        train_ops = build_train_operators(
            model=model, train_data=train_data, batch_size=batch_size,
            labels=labels, bc=bc
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


def build_train_operators(model, train_data, batch_size, labels, bc):
    if bc:
        iterator, next_element = make_dataset_bc(
            data=train_data, count=1, shuffle=True, batch_size=batch_size,
            is_training=True
        )

        logits = model(next_element.x, next_element.meta, is_training=True)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(next_element.y), logits=logits
        )
        weighted_loss = loss * next_element.weight

        y = tf.argmax(next_element.y, axis=1)
    else:
        iterator, next_element = make_dataset(
            data=train_data, count=1, shuffle=True, batch_size=batch_size,
            is_training=True
        )

        logits = model(next_element.x, next_element.meta, is_training=True)
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
    iterator, next_element = make_dataset(
        data=test_data, count=1, shuffle=False, batch_size=batch_size,
        is_training=False
    )

    logits = model(next_element.x, next_element.meta, is_training=False)
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


@cmd.command()
@click.option('--model-dir', type=click.Path())
@click.option('--batch-size', type=int, default=1000)
def predict(model_dir, batch_size):
    model_dir = Path(model_dir)
    with (model_dir / 'parameters.json').open('r') as f:
        parameters = json.load(f)

    data_dir = Path(parameters['data']['path'])
    # seed = parameters['seed']

    train_data, test_data, labels = load_data2(data_dir=data_dir)

    with tf.Graph().as_default() as graph:
        model = EnvNetV2(n_classes=len(np.unique(train_data.y)))

        train_iterator, train_element = make_dataset(
            data=train_data, count=1, shuffle=False, batch_size=batch_size,
            is_training=False
        )
        test_iterator, test_element = make_dataset(
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


def main():
    cmd()


if __name__ == '__main__':
    main()
