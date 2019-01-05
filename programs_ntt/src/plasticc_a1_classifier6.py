#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
データを画像にしてCNNで学習する
"""
import json
from collections import namedtuple, defaultdict
from concurrent.futures import ProcessPoolExecutor
from operator import itemgetter
from pathlib import Path

import click
import numpy as np
import pandas as pd
import sonnet as snt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import trange

from data.plasticc_a1_data import make_image2
from plasticc_a1_classifier4 import draw_confusion_matrix, OutputType

__author__ = 'Yasuhiro Imoto'
__date__ = '05/12/2018'


Data = namedtuple('Data', ['df', 'target', 'weight'])


def load_data(data_path, meta_data_path, table_path, binary):
    df = pd.read_csv(
        data_path,
        names=['object_id', 'mjd','passband', 'flux', 'flux_err',
               'detection', 'interpolation']
    )
    meta = pd.read_csv(meta_data_path, index_col=0, header=0)
    table = pd.read_csv(
        table_path, header=None, comment='#',
        delim_whitespace=True, names=['z', 'dm', 'dm0.5', 'factor']
    )

    # 超新星のクラスのみを選ぶ
    sn_classes = (42, 52, 62, 90, 95)
    # object_idで分割
    df_list = defaultdict(list)
    for object_id, group in df.groupby('object_id'):
        tmp = meta.loc[object_id]

        target = int(tmp['target'])
        if target not in sn_classes:
            continue

        df_list[target].append((
            group, tmp['hostgal_specz'], tmp['hostgal_photoz'],
            tmp['hostgal_photoz_err']
        ))

    counts = {key: len(value) for key, value in df_list.items()}
    labels, _ = zip(*sorted(counts.items(), key=itemgetter(1)))

    # train, testに分割
    train_data = []
    test_data = []
    train_weight = []
    test_weight = []
    for i, c in enumerate(labels):
        train_list, test_list = train_test_split(
            df_list[c], test_size=0.3, random_state=c
        )
        train_data.append(train_list)
        test_data.append(test_list)

        train_weight.append(1.0 / len(train_list))
        test_weight.append(1.0 / len(train_list))

    train_dataset = (train_data, train_weight)
    test_dataset = (test_data, test_weight)
    return train_dataset, test_dataset, labels, table


def generate_random_batch(df_all, weight, n_classes, object_id, batch_size,
                          is_training, redshift_table, image_parameters,
                          mixup):
    n = batch_size * 2 if mixup else batch_size

    p = np.array([len(v) for v in df_all], dtype=np.float32)
    p = p / np.sum(p)

    while True:
        batch_x = []
        batch_y = []
        batch_weight = []
        batch_id = []

        for _ in range(n):
            c = int(np.random.choice(n_classes, size=1, p=p))
            df_band = df_all[c]

            i = int(np.random.choice(len(df_band), size=1))
            df, specz, photoz, photoz_err = df_band[i]

            x = make_image2((
                df, specz, photoz, photoz_err, image_parameters,
                is_training
            ))

            batch_x.append(x)
            batch_y.append(c)
            batch_weight.append(weight[c])
            batch_id.append(object_id[c][i])

        batch_x = np.stack(batch_x, axis=0).astype(np.float32)
        batch_y = np.asarray(batch_y, dtype=np.int32)
        batch_weight = np.asarray(batch_weight, dtype=np.float32)
        batch_id = np.asarray(batch_id, dtype=np.int32)

        if mixup:
            alpha = 2
            r = np.random.beta(alpha, alpha, size=[batch_size])

            tmp = np.reshape(r, [-1, 1, 1, 1])
            batch_x = (tmp * batch_x[:batch_size] +
                       (1 - tmp) * batch_x[batch_size:])

            tmp = np.reshape(r, [-1, 1])
            one_hot = tf.keras.utils.to_categorical(
                batch_y, n_classes
            )
            batch_y = (tmp * one_hot[:batch_size] +
                       (1 - tmp) * one_hot[batch_size:])

            batch_weight = (r * batch_weight[:batch_size] +
                            (1 - r) * batch_weight[batch_size:])

            batch_id = np.stack(
                [batch_id[:batch_size], batch_id[batch_size:]], axis=1
            )

        output = OutputType(
            x=batch_x, y=batch_y, weight=batch_weight,
            object_id=batch_id
        )
        yield output


def make_dataset(data, redshift_table, image_parameters, count, shuffle,
                 batch_size, is_training, mixup):
    def generator():
        df_all, weight = data
        n_classes = len(df_all)

        if shuffle:
            object_id = defaultdict(list)
            for c in range(n_classes):
                for df, _, _, _ in df_all[c]:
                    tmp = df['object_id']
                    object_id[c].append(int(tmp.iloc[0]))

            random_batch = generate_random_batch(
                df_all=df_all, weight=weight, n_classes=n_classes,
                object_id=object_id, batch_size=batch_size,
                is_training=is_training, redshift_table=redshift_table,
                image_parameters=image_parameters, mixup=mixup
            )
            for output in random_batch:
                yield output
        else:
            raise NotImplementedError()

    types = OutputType(x=tf.float32, y=tf.int32,
                       weight=tf.float32, object_id=tf.int32)
    x_size = image_parameters['x_size']
    y_size = ((image_parameters['y_max'] - image_parameters['y_min']) *
              image_parameters['y_scale'])
    if mixup:
        y_shape = tf.TensorShape([None, len(data[0])])
        id_shape = tf.TensorShape([None, 2])
    else:
        y_shape = tf.TensorShape([None])
        id_shape = tf.TensorShape([None])
    shapes = OutputType(
        x=tf.TensorShape([None, 6, y_size, x_size]), y=y_shape,
        weight=tf.TensorShape([None]), object_id=id_shape
    )
    dataset = tf.data.Dataset.from_generator(
        generator=generator, output_types=types, output_shapes=shapes
    )
    dataset = dataset.apply(
        tf.contrib.data.prefetch_to_device('/gpu:0')
    )

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


class Coordinate(snt.AbstractModule):
    def __init__(self):
        super().__init__()

    def _build(self, inputs):
        shape = tf.shape(inputs)
        h, w = shape[2], shape[3]
        b = shape[0]
        # 縦方向
        c1 = tf.linspace(-1.0, 1.0, num=h)
        c1 = tf.tile(tf.reshape(c1, [1, 1, -1, 1]), [b, 1, 1, w])
        # 横方向
        c2 = tf.linspace(-1.0, 1.0, num=w)
        c2 = tf.tile(tf.reshape(c2, [1, 1, 1, -1]), [b, 1, h, 1])

        outputs = tf.concat([inputs, c1, c2], axis=1)
        return outputs


class ResidualBlock(snt.AbstractModule):
    def __init__(self, output_channels, kernel_shape, update_collection_name):
        super().__init__()
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.update_collection_name = update_collection_name

    def _build(self, inputs, is_training):
        h = snt.Sequential([
            snt.BatchNormV2(
                data_format='NCHW',
                update_ops_collection=self.update_collection_name
            ),
            tf.nn.relu,
            Coordinate(),
            snt.Conv2D(
                output_channels=self.output_channels,
                kernel_shape=self.kernel_shape,
                stride=(1, 2), data_format='NCHW', use_bias=False,
                initializers={'w': tf.keras.initializers.he_normal()}
            )
        ])(inputs, is_training)
        outputs = snt.Sequential([
            snt.BatchNormV2(
                data_format='NCHW',
                update_ops_collection=self.update_collection_name
            ),
            tf.nn.relu,
            Coordinate(),
            snt.Conv2D(
                output_channels=self.output_channels,
                kernel_shape=self.kernel_shape,
                stride=(1, 1), data_format='NCHW',
                initializers={'w': tf.keras.initializers.he_normal()}
            )
        ])(h, is_training)

        return outputs


class ResidualBottleneckBlock(snt.AbstractModule):
    def __init__(self, output_channels, kernel_shape, stride,
                 update_collection_name):
        super().__init__()
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.update_collection_name = update_collection_name

    def _build(self, inputs, is_training):
        h = snt.Sequential([
            snt.BatchNormV2(
                data_format='NCHW',
                update_ops_collection=self.update_collection_name
            ),
            tf.nn.relu,
            Coordinate(),
            snt.Conv2D(
                output_channels=self.output_channels // 4, kernel_shape=(1, 1),
                stride=(1, 1),
                data_format='NCHW', use_bias=False,
                initializers={'w': tf.keras.initializers.he_normal()}
            )
        ])(inputs, is_training)
        h = snt.Sequential([
            snt.BatchNormV2(
                data_format='NCHW',
                update_ops_collection=self.update_collection_name
            ),
            tf.nn.relu,
            Coordinate(),
            snt.Conv2D(
                output_channels=self.output_channels // 4,
                kernel_shape=self.kernel_shape, stride=self.stride,
                data_format='NCHW', use_bias=False,
                initializers={'w': tf.keras.initializers.he_normal()}
            )
        ])(h, is_training)
        outputs = snt.Sequential([
            snt.BatchNormV2(
                data_format='NCHW',
                update_ops_collection=self.update_collection_name
            ),
            tf.nn.relu,
            Coordinate(),
            snt.Conv2D(
                output_channels=self.output_channels, kernel_shape=(1, 1),
                stride=(1, 1), data_format='NCHW',
                initializers={'w': tf.keras.initializers.he_normal()}
            )
        ])(h, is_training)

        return outputs


class ZeroPadding(snt.AbstractModule):
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels

    def _build(self, inputs):
        shape = tf.shape(inputs)

        pad = tf.zeros([
            shape[0], self.output_channels - shape[1], shape[2], shape[3]
        ])
        outputs = tf.concat([inputs, pad], axis=1)

        return outputs


class Classifier(snt.AbstractModule):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.update_collection_name = 'classifier_bn'

    def _build(self, inputs, is_training):
        h = snt.Sequential([
            Coordinate(),
            snt.Conv2D(
                output_channels=16, kernel_shape=(5, 5), stride=(2, 2),
                data_format='NCHW',
                initializers={'w': tf.keras.initializers.he_normal()}
            ),
            tf.nn.selu,
            snt.Conv2D(
                output_channels=32, kernel_shape=(5, 5), stride=(1, 2),
                data_format='NCHW',
                initializers={'w': tf.keras.initializers.he_normal()}
            )
        ])(inputs)
        # ここで大きさは[20, 40]よりは少し小さい

        g = ResidualBlock(
            output_channels=64, kernel_shape=(3, 3),
            update_collection_name=self.update_collection_name
        )(h, is_training=is_training)
        f = tf.nn.avg_pool(
            h, ksize=(1, 1, 2, 4), strides=(1, 1, 1, 2), padding=snt.SAME,
            data_format='NCHW'
        )
        f = ZeroPadding(output_channels=64)(f)
        h = f + g
        # ここで大きさは[20, 20]よりは少し小さい

        g = ResidualBottleneckBlock(
            output_channels=128, kernel_shape=(3, 3), stride=(2, 2),
            update_collection_name=self.update_collection_name
        )(h, is_training=is_training)
        f = tf.nn.max_pool(
            h, ksize=(1, 1, 2, 2), strides=(1, 1, 2, 2), padding=snt.SAME,
            data_format='NCHW'
        )
        f = ZeroPadding(output_channels=128)(f)

        h = f + g
        # ここで大きさは[10, 10]よりは少し小さい

        g = ResidualBottleneckBlock(
            output_channels=256, kernel_shape=(3, 3), stride=(2, 2),
            update_collection_name=self.update_collection_name
        )(h, is_training=is_training)
        f = tf.nn.max_pool(
            h, ksize=(1, 1, 2, 2), strides=(1, 1, 2, 2), padding=snt.SAME,
            data_format='NCHW'
        )
        f = ZeroPadding(output_channels=256)(f)
        h = f + g
        # ここで大きさは[5, 5]よりは少し小さい

        h = snt.BatchFlatten()(h)
        h = snt.Sequential([
            snt.BatchNormV2(
                data_format='NC',
                update_ops_collection=self.update_collection_name
            ),
            tf.nn.relu,
            snt.Linear(output_size=256)
        ])(h, is_training)
        h = snt.Sequential([
            snt.BatchNormV2(
                data_format='NC',
                update_ops_collection=self.update_collection_name
            ),
            tf.nn.relu,
            snt.Linear(output_size=64)
        ])(h, is_training)
        outputs = snt.Sequential([
            snt.BatchNormV2(
                data_format='NC',
                update_ops_collection=self.update_collection_name
            ),
            tf.nn.relu,
            snt.Linear(output_size=self.n_classes)
        ])(h, is_training)

        return outputs


class Classifier1(snt.AbstractModule):
    def __init__(self, n_classes, n_blocks, n_fc):
        super().__init__()
        self.n_classes = n_classes
        self.n_blocks = n_blocks
        self.n_fc = n_fc
        self.update_collection_name = 'classifier_bn'

    def _build(self, inputs, is_training):
        h = snt.Sequential([
            Coordinate(),
            snt.Conv2D(
                output_channels=16, kernel_shape=(5, 5), stride=(2, 2),
                data_format='NCHW',
                initializers={'w': tf.keras.initializers.he_normal()}
            ),
            tf.nn.selu,
            snt.Conv2D(
                output_channels=32, kernel_shape=(5, 5), stride=(2, 2),
                data_format='NCHW',
                initializers={'w': tf.keras.initializers.he_normal()}
            )
        ])(inputs)

        for _, c in zip(range(self.n_blocks), [64, 128, 256, 512, 1024]):
            g = ResidualBottleneckBlock(
                output_channels=c, kernel_shape=(3, 3), stride=(2, 2),
                update_collection_name=self.update_collection_name
            )(h, is_training=is_training)
            f = tf.nn.max_pool(
                h, ksize=(1, 1, 2, 2), strides=(1, 1, 2, 2), padding=snt.SAME,
                data_format='NCHW'
            )
            f = ZeroPadding(output_channels=c)(f)
            h = f + g

        h = tf.reduce_mean(h, axis=[2, 3])
        for size in [256, 64, 16][3 - self.n_fc:]:
            h = snt.Sequential([
                snt.BatchNormV2(
                    data_format='NC',
                    update_ops_collection=self.update_collection_name
                ),
                tf.nn.relu,
                snt.Linear(output_size=size)
            ])(h, is_training)

        outputs = snt.Sequential([
            snt.BatchNormV2(
                data_format='NC',
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
@click.option('--n-blocks', type=int)
@click.option('--n-fc', type=int)
@click.option('--binary', is_flag=True)
@click.option('--mixfeat', is_flag=True)
@click.option('--sigma', type=float, default=0.1,
              help='a parameter of mixfeat')
def train(data_dir, model_dir, batch_size, epochs, seed, cv, mixup,
          hidden_size, n_layers, n_blocks, n_fc, binary, mixfeat, sigma):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    image_parameters = {
        'x_size': 180, 'x_offset': 3, 'y_min': -15, 'y_max': 15, 'y_scale': 20
    }
    parameters = {
        'data': {'path': data_dir},
        'seed': seed, 'cv': cv, 'mixup': mixup, 'hidden_size': hidden_size,
        'n_layers': n_layers, 'binary': binary,
        'mixfeat': {'use': mixfeat, 'sigma': sigma},
        'image_parameters': image_parameters,
        'n_blocks': n_blocks, 'n_fc': n_fc
    }
    with (model_dir / 'parameters.json').open('w') as f:
        json.dump(parameters, f, sort_keys=True, indent=4)

    data_dir = Path(data_dir)
    data_path = data_dir / 'PLAsTiCC_training_set_bin1_interplt_180days.csv'
    meta_data_path = data_dir / 'training_set_metadata.csv'
    table_path = data_dir.parents[1] / 'external/redshift.tbl'
    train_data, test_data, labels, redshift_table = load_data(
        data_path=data_path, meta_data_path=meta_data_path,
        table_path=table_path, binary=binary
    )

    train_size = 0
    for tmp in train_data[0]:
        train_size += len(tmp)
    n_iterations_train = (train_size + batch_size - 1) // batch_size
    print(train_size, n_iterations_train)
    test_size = 0
    for tmp in test_data[0]:
        test_size += len(tmp)
    n_iterations_test = (test_size + batch_size - 1) // batch_size
    print(test_size, n_iterations_test)

    with tf.Graph().as_default() as graph:
        model = Classifier1(
            n_classes=len(train_data[0]), n_blocks=n_blocks, n_fc=n_fc
        )

        train_ops = build_train_operators(
            model=model, train_data=train_data, redshift_table=redshift_table,
            image_parameters=image_parameters, batch_size=batch_size,
            labels=labels, mixup=mixup
        )
        test_ops = build_test_operators(
            model=model, test_data=test_data, redshift_table=redshift_table,
            image_parameters=image_parameters, batch_size=batch_size,
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
            sess.run(test_ops.initialize)
            for _ in trange(step, epochs):
                for _ in range(n_iterations_train):
                    sess.run([train_ops.optimize, train_ops.update])
                summary, step = sess.run([train_ops.summary, count_op])
                writer.add_summary(summary=summary, global_step=step)
                sess.run(train_ops.reset)

                for _ in range(n_iterations_test):
                    sess.run(test_ops.update)
                summary = sess.run(test_ops.summary)
                writer.add_summary(summary=summary, global_step=step)
                sess.run(test_ops.reset)

                saver.save(sess=sess, save_path=str(model_dir / 'model'),
                           global_step=global_step, write_meta_graph=False)


def build_train_operators(model, train_data, redshift_table, image_parameters,
                          batch_size, labels, mixup):
    iterator, next_element = make_dataset(
        data=train_data, redshift_table=redshift_table,
        image_parameters=image_parameters, count=1, shuffle=True,
        batch_size=batch_size, is_training=True, mixup=mixup
    )

    logits = model(next_element.x, is_training=True)
    if mixup:
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(next_element.y), logits=logits
        )

        y = tf.argmax(next_element.y, axis=1)
        y = tf.reshape(y, [-1, 1])
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=next_element.y, logits=logits
        )
        y = next_element.y
    weighted_loss = loss * next_element.weight

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


def build_test_operators(model, test_data, redshift_table, image_parameters,
                         batch_size, labels):
    iterator, next_element = make_dataset(
        data=test_data, redshift_table=redshift_table,
        image_parameters=image_parameters, count=1, shuffle=True,
        batch_size=batch_size, is_training=False, mixup=False
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


@cmd.command()
@click.option('--data-dir', type=click.Path(exists=True))
@click.option('--model-dir', type=click.Path())
@click.option('--batch-size', type=int, default=64)
def export_image(data_dir, model_dir, batch_size):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    image_parameters = {
        'x_size': 180, 'x_offset': 3, 'y_min': -15, 'y_max': 15, 'y_scale': 20
    }

    data_dir = Path(data_dir)
    data_path = data_dir / 'PLAsTiCC_training_set_bin1_interplt_180days.csv'
    meta_data_path = data_dir / 'training_set_metadata.csv'
    table_path = data_dir.parents[1] / 'external/redshift.tbl'
    train_data, test_data, labels, redshift_table = load_data(
        data_path=data_path, meta_data_path=meta_data_path,
        table_path=table_path, binary=False
    )

    with tf.Graph().as_default() as graph:
        train_iterator, train_element = make_dataset(
            data=train_data, redshift_table=redshift_table,
            image_parameters=image_parameters, count=1, shuffle=True,
            batch_size=batch_size, is_training=True, mixup=False
        )

        test_iterator, test_element = make_dataset(
            data=test_data, redshift_table=redshift_table,
            image_parameters=image_parameters, count=1, shuffle=True,
            batch_size=batch_size, is_training=False, mixup=False
        )

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=config, graph=graph) as sess:
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            sess.run([train_iterator.initializer, test_iterator.initializer])

            train_object = output_data(sess=sess, next_element=train_element)
            test_object = output_data(sess=sess, next_element=test_element)

        np.savetxt(str(model_dir / 'labels.txt'), labels)
        np.savez_compressed(str(model_dir / 'train.npz'), **train_object)
        np.savez_compressed(str(model_dir / 'test.npz'), **test_object)


def output_data(sess, next_element):
    x, y, object_id = [], [], []
    for i in range(10):
        tmp = sess.run([
            next_element.x, next_element.y, next_element.object_id
        ])
        x.append(tmp[0])
        y.append(tmp[1])
        object_id.append(tmp[2])
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    object_id = np.concatenate(object_id, axis=0)

    d = dict(x=x, y=y, object_id=object_id)
    return d


def main():
    cmd()


if __name__ == '__main__':
    main()
