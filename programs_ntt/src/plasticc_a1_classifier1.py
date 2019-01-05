#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import json
import warnings
from pathlib import Path
from collections import namedtuple

import pandas as pd
import numpy as np
import xarray as xr
import tensorflow as tf
import sonnet as snt
import click
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import trange, tqdm

try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import tfmpl
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import seaborn as sns

__author__ = 'Yasuhiro Imoto'
__date__ = '19/11/2018'


def convert_data(data_path):
    ds = xr.open_dataset(data_path)

    field_names = (
        'flux', 'flux_err', 'detected', 'hostgal_specz', 'hostgal_photoz',
        'hostgal_photoz_err', 'distmod', 'mwebv'
    )
    if hasattr(ds, 'target'):
        # 個数が少ないデータを学習対象から取り除く
        label, count = np.unique(ds.target, return_counts=True)
        flag = np.zeros(len(ds.target), dtype=np.bool)
        for i, c in zip(label, count):
            if c < 10:
                # 少ないデータを無視
                continue
            flag = np.logical_or(flag, ds.target == i)

        values = {
            key: getattr(ds, key)[flag].values.astype(np.float32)
            for key in field_names
        }
        object_id = ds.x[flag].values.astype(np.int32)

        target = ds.target[flag].values
        # 学習しないクラスがあるので、その分をつめる
        label_map = []
        counter = 0
        for i, c in zip(label, count):
            if c < 10:
                label_map.append(-1)
                continue
            label_map.append(counter)
            counter += 1
        target = np.asarray([label_map[i] for i in target])

        label_map = np.asarray(label_map)
    else:
        target = np.empty(len(ds.flux), dtype=np.int32)
        target[:] = -1

        label_map, count = None, None

        values = {
            key: getattr(ds, key).values.astype(np.float32)
            for key in field_names
        }
        object_id = ds.x.values.astype(np.int32)

    for key in ('hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err'):
        # nanを0に変換
        values[key] = np.nan_to_num(values[key]).reshape([-1, 1])
        # 名前を変更
        values[key[8:]] = values[key]
        del values[key]

    dataset = Dataset(
        target=target, object_id=object_id, **values
    )
    return dataset, (label_map, count)


Dataset = namedtuple(
    'Dataset',
    'flux, flux_err, detected, specz, photoz, photoz_err, '
    'distmod, mwebv, target, object_id'
)


def setup_dataset(values, use_hostgal, repeat, shuffle, batch_size,
                  mixup=False, n_classes=None):
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
        mixup=mixup, object_id=values.object_id, n_classes=n_classes
    )

    if n_classes is None:
        count = None
    else:
        _, count = np.unique(values.target, return_counts=True)

    return iterator, next_element, count


def make_dataset(flux, flux_err, target, object_id, shuffle, repeat,
                 batch_size, additional_inputs=None,
                 mixup=False, n_classes=None):
    inputs = {'flux': flux, 'flux_err': flux_err, 'target': target,
              'object_id': object_id}
    if additional_inputs is not None:
        inputs.update(additional_inputs)

    if mixup:
        assert isinstance(n_classes, int) and n_classes > 0

        dataset1 = tf.data.Dataset.from_tensor_slices(inputs)
        dataset2 = tf.data.Dataset.from_tensor_slices(inputs)

        global_step = tf.train.get_or_create_global_step()
        dataset2 = dataset2.shuffle(1000, seed=global_step * 2)

        dataset = tf.data.Dataset.zip((dataset1, dataset2)).repeat(repeat)

        if shuffle:
            dataset = dataset.shuffle(1000, seed=global_step)
        dataset = dataset.map(
            lambda v1, v2: map_func_mixup(v1, v2, n_classes=n_classes)
        ).batch(batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.repeat(repeat)
        if shuffle:
            global_step = tf.train.get_or_create_global_step()
            dataset = dataset.shuffle(1000, seed=global_step)
        dataset = dataset.map(map_func)
        dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def map_func(data):
    flux = data['flux']
    flux_err = data['flux_err']
    mag = tf.asinh(
        0.5 * (flux + flux_err * tf.random_normal(shape=tf.shape(flux)))
    )

    y = data['target']

    if 'specz' in data:
        # 時々データをNaN相当(0)にする
        flag = tf.to_float(tf.squeeze(
            tf.multinomial([[tf.log(0.9), tf.log(0.1)]], num_samples=1)
        ))
        data['specz'] = flag * data['specz']
    if 'photoz' in data and 'photoz_err' in data:
        data['photoz'] = (data['photoz'] +
                          tf.random_normal(shape=[]) * data['photoz_err'])

    names = ('flux', 'flux_err', 'target', 'photoz_err', 'object_id')
    keys = [key for key in data.keys() if key not in names]
    # 順序を固定
    keys.sort()

    x = tf.concat(
        [mag] + [tf.reshape(data[key], [-1]) for key in keys],
        axis=0
    )

    return x, y, data['object_id']


def map_func_mixup(data1, data2, n_classes):
    x1, y1, _ = map_func(data=data1)
    x2, y2, _ = map_func(data=data2)

    alpha = 4.0
    distribution = tf.distributions.Beta(alpha, alpha)
    r = distribution.sample()

    x = r * x1 + (1 - r) * x2
    y = r * tf.one_hot(y1, n_classes) + (1 - r) * tf.one_hot(y2, n_classes)

    return x, y


class HighWay(snt.AbstractModule):
    def __init__(self):
        super().__init__()

    def _build(self, inputs, is_training):
        hidden_size = inputs.get_shape()[1].value

        h = snt.Sequential([
            snt.Linear(output_size=hidden_size),
            tf.nn.sigmoid
        ])(inputs)

        gate = snt.Sequential([
            snt.Linear(output_size=hidden_size),
            tf.nn.sigmoid
        ])(inputs)

        outputs = gate * h + (1 - gate) * inputs

        return outputs


class Classifier(snt.AbstractModule):
    def __init__(self, classes, hidden_size, n_highways, drop_rate):
        super().__init__()
        self.classes = classes
        self.hidden_size = hidden_size
        self.n_highways = n_highways
        self.drop_rate = drop_rate

    def _build(self, inputs, is_training):
        h = snt.Sequential([
            snt.BatchNormV2(
                data_format='NC', update_ops_collection='classifier_bn'
            ),
            snt.Linear(output_size=self.hidden_size)
        ])(inputs, is_training)
        if self.drop_rate > 0:
            h = tf.layers.dropout(h, rate=self.drop_rate, training=is_training)

        h = snt.Sequential([
            snt.BatchNormV2(
                data_format='NC', update_ops_collection='classifier_bn'
            ),
            tf.nn.relu,
            snt.Linear(output_size=self.hidden_size)
        ])(h, is_training)
        if self.drop_rate > 0:
            h = tf.layers.dropout(h, rate=self.drop_rate, training=is_training)

        for i in range(self.n_highways):
            h = snt.Sequential([
                snt.BatchNormV2(
                    data_format='NC', update_ops_collection='classifier_bn'
                ),
                tf.nn.relu
            ])(h, is_training)
            h = HighWay()(h, is_training)
            if self.drop_rate > 0:
                h = tf.layers.dropout(h, rate=self.drop_rate,
                                      training=is_training)

        outputs = snt.Sequential([
            snt.BatchNormV2(
                data_format='NC', update_ops_collection='classifier_bn'
            ),
            tf.nn.relu,
            snt.Linear(output_size=self.classes)
        ])(h, is_training)

        return outputs


def learn(train_data, validation_data, n_classes, hidden_size, n_highways,
          drop_rate, mixup, use_hostgal, batch_size, epochs, model_dir):
    with tf.Graph().as_default() as graph:
        train_iterator, train_element, train_count = setup_dataset(
            values=train_data, use_hostgal=use_hostgal, shuffle=True,
            batch_size=batch_size, mixup=mixup, repeat=1, n_classes=n_classes
        )
        (validation_iterator, validation_element,
         validation_count) = setup_dataset(
            values=validation_data, use_hostgal=use_hostgal, shuffle=False,
            batch_size=batch_size, mixup=False, repeat=1, n_classes=n_classes
        )

        model = Classifier(
            classes=n_classes, hidden_size=hidden_size, n_highways=n_highways,
            drop_rate=drop_rate
        )

        train_ops, train_wights = build_train(
            model=model, iterator=train_iterator, next_element=train_element,
            count=train_count, mixup=mixup, n_classes=n_classes
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

        if mixup:
            train_iterator, train_element, train_count = setup_dataset(
                values=train_data, use_hostgal=use_hostgal, shuffle=False,
                batch_size=batch_size, mixup=False, repeat=1,
                n_classes=n_classes
            )
        mode = tf.constant(False, dtype=tf.bool, shape=[])
        train_logits = model(train_element[0], mode)
        train_accuracy = tf.metrics.accuracy(
            labels=train_element[1],
            predictions=tf.argmax(train_logits, axis=1)
        )

        validation_logits = model(validation_element[0], mode)
        validation_accuracy = tf.metrics.accuracy(
            labels=validation_element[1],
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

            sess.run(train_iterator.initializer)
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
            test_score = float(sess.run(validation_accuracy[0]))

        return train_score, test_score


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
@click.option('--hidden-size', type=int, default=200)
@click.option('--n-highways', type=int, default=3)
@click.option('--drop-rate', type=float, default=0.5)
@click.option('--mixup', is_flag=True)
def train(data_dir, model_dir, epochs, batch_size, seed, hostgal, cv,
          hidden_size, n_highways, drop_rate, mixup):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    data_dir = Path(data_dir)
    data_path = data_dir / 'train.nc'

    parameters = {
        'data': {'path': str(data_path), 'hostgal': hostgal},
        'seed': seed, 'cv': cv, 'hidden_size': hidden_size,
        'drop_rate': drop_rate, 'n_highways': n_highways, 'mixup': mixup
    }
    with (model_dir / 'parameters.json').open('w') as f:
        json.dump(parameters, f, sort_keys=True, indent=4)

    dataset, (label, count) = convert_data(data_path=data_path)

    print(label)
    print(count)
    # noinspection PyTypeChecker
    count = count[label >= 0]
    n_classes = len(count)

    if cv > 0:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

        train_score_list, test_score_list = [], []
        for i, (train_index, test_index) in enumerate(
                tqdm(skf.split(dataset.target, dataset.target), total=cv)):
            train_values = Dataset._make(v[train_index] for v in dataset)
            test_values = Dataset._make(v[test_index] for v in dataset)

            tmp_dir = model_dir / str(i)
            if not tmp_dir.exists():
                tmp_dir.mkdir(parents=True)

            train_score, test_score = learn(
                train_data=train_values, validation_data=test_values,
                n_classes=n_classes, hidden_size=hidden_size,
                n_highways=n_highways, drop_rate=drop_rate, mixup=mixup,
                use_hostgal=hostgal, batch_size=batch_size,
                model_dir=tmp_dir, epochs=epochs
            )
            train_score_list.append(train_score)
            test_score_list.append(test_score)

        with (model_dir / 'score.json').open('w') as f:
            json.dump({'train': train_score_list, 'test': test_score_list}, f,
                      indent=4, sort_keys=True)

        return

    tmp = train_test_split(
        *list(dataset), test_size=0.2, random_state=seed,
        stratify=dataset.target
    )

    train_values = Dataset._make(tmp[0::2])
    test_values = Dataset._make(tmp[1::2])

    train_score, test_score = learn(
        train_data=train_values, validation_data=test_values,
        n_classes=n_classes, hidden_size=hidden_size,
        n_highways=n_highways, drop_rate=drop_rate, mixup=mixup,
        use_hostgal=hostgal, batch_size=batch_size,
        model_dir=model_dir, epochs=epochs
    )

    with (model_dir / 'score.json').open('w') as f:
        json.dump({'train': train_score, 'test': test_score}, f,
                  sort_keys=True, indent=4)


def build_train(model, iterator, next_element, count, mixup, n_classes):
    weight = 1.0 / count
    # weight = np.sqrt(weight)
    print(weight)

    train_mode = tf.constant(True, dtype=tf.bool, shape=[])
    logits = model(next_element[0], train_mode)
    if mixup:
        labels = tf.argmax(next_element[1], axis=-1)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(next_element[1]), logits=logits
        )
    else:
        labels = next_element[1]

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
    weight = tf.constant(weight, dtype=tf.float32)
    loss = loss * tf.gather(weight, labels) / tf.reduce_sum(weight)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=1e-3, momentum=0.9, use_nesterov=True
    )
    updates = tf.get_collection_ref('classifier_bn')
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
    validation_mode = tf.constant(False, dtype=tf.bool, shape=[])
    logits = model(next_element[0], validation_mode)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=next_element[1], logits=logits
    )
    # final scoreでクラスごとの重みがほぼ均一になるようにとのことなので、
    # final scoreを計算するときのデータの個数で平均するのだと思う
    # 計算式の方にNがあるので、それを打ち消して実際の個数での平均にする値が
    # wの値と予想
    weight = tf.constant(count / validation_count, dtype=tf.float32)
    m = tf.gather(weight / count, next_element[1])
    n = tf.reduce_sum(weight)
    loss = loss * m / n

    summary_op, update_op, reset_op = make_metrics(
        logits=logits, loss=loss,
        labels=next_element[1], n_classes=n_classes,
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


@tfmpl.figure_tensor
def draw_confusion_matrix(confusion_matrix):
    fig = tfmpl.create_figure()
    ax = fig.add_subplot(111)
    sns.heatmap(confusion_matrix, annot=True, square=True, ax=ax, cbar=False,
                fmt="d", cmap='PuRd')
    return fig


@cmd.command()
@click.option('--model-dir', type=click.Path(exists=True))
@click.option('--batch-size', type=int, default=1000)
def predict(model_dir, batch_size):
    model_dir = Path(model_dir)
    with (model_dir / 'parameters.json').open('r') as f:
        parameters = json.load(f)

    train_path = Path(parameters['data']['path'])
    test_path = train_path.with_name('test.nc')

    dataset, _ = convert_data(test_path)
    if parameters['data']['hostgal']:
        additional_inputs = {
            'specz': dataset.specz, 'photoz': dataset.photoz,
            'photoz_err': dataset.photoz_err
        }
    else:
        additional_inputs = None

    iterator, next_element = make_dataset(
        flux=dataset.flux, flux_err=dataset.flux_err,
        target=dataset.target, object_id=dataset.object_id, shuffle=False,
        repeat=1, batch_size=batch_size, additional_inputs=additional_inputs
    )
    is_training = tf.constant(False, dtype=tf.bool, shape=[])

    n_classes = parameters['n_classes']
    model = Classifier(
        classes=n_classes,
        hidden_size=parameters['hidden_size'],
        n_highways=parameters['n_highways'],
        drop_rate=-1
    )
    logits = model(next_element[0], is_training)
    probability = tf.nn.softmax(logits)
    label = tf.argmax(logits, axis=1)

    object_id = next_element[2]

    global_step = tf.train.create_global_step()

    saver = tf.train.Saver()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        path = checkpoint.model_checkpoint_path
        saver.restore(sess=sess, save_path=path)

        sess.run(iterator.initializer)
        p_data = []
        l_data = []
        label_list = []
        id_list = []
        while True:
            try:
                tmp = sess.run([object_id, probability, logits, label])
                id_list.append(tmp[0])
                label_list.append(tmp[-1])
                p_data.append(tmp[1])
                l_data.append(tmp[2])
            except tf.errors.OutOfRangeError:
                break

        step = sess.run(global_step)
    p_data = np.vstack(p_data)
    l_data = np.vstack(l_data)
    id_list = np.hstack(id_list)
    label_list = np.hstack(label_list)

    df_probability = pd.DataFrame(
        p_data, index=id_list, columns=range(n_classes)
    )
    df_probability['label'] = label_list

    df_logits = pd.DataFrame(l_data, index=id_list, columns=range(n_classes))
    df_logits['label'] = label_list

    df_probability.to_csv(
        str(model_dir / 'probability-{0:05d}.csv'.format(step))
    )
    df_logits.to_csv(str(model_dir / 'logits-{0:05d}.csv'.format(step)))


def main():
    cmd()


if __name__ == '__main__':
    main()
