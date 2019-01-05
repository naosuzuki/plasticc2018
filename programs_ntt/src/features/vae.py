#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
import sonnet as snt
import tensorflow as tf
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import tfmpl
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import seaborn as sns

__author__ = 'Yasuhiro Imoto'
__date__ = '27/12/2018'


def load_data(data_dir, mask_path, max_length, count, shuffle, batch_size,
              use_mask, use_magnitude):
    tmp = np.load(mask_path)
    mask_pattern = tmp['mask']
    mask_size = tmp['size']

    data_dir = Path(data_dir)
    records = [str(f) for f in data_dir.glob('*64.tfrecord')]

    dataset = tf.data.TFRecordDataset(records, compression_type='GZIP')
    dataset = dataset.repeat(count=count)
    if shuffle:
        global_step = tf.train.get_or_create_global_step()
        dataset = dataset.shuffle(1000, seed=global_step)
    dataset = dataset.map(
        map_func=lambda data: parse(
            data=data, mask_pattern_list=mask_pattern,
            mask_size_list=mask_size, max_length=max_length,
            use_mask=use_mask, use_magnitude=use_magnitude
        )
    ).batch(batch_size=batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def parse(data, mask_pattern_list, mask_size_list, max_length, use_mask,
          use_magnitude):
    features = tf.parse_single_example(
        data,
        features={
            'flux': tf.FixedLenFeature([6 * max_length], tf.float32),
            # 'flux_err': tf.FixedLenFeature([6 * max_length], tf.float32),
            'size': tf.FixedLenFeature([], tf.int64),
            'target': tf.FixedLenFeature([], tf.int64),
            'class_id': tf.FixedLenFeature([], tf.int64),
            'object_id': tf.FixedLenFeature([], tf.int64),
            'sub_id': tf.FixedLenFeature([], tf.int64),
            'redshift': tf.FixedLenFeature([], tf.float32)
        }
    )

    # 実際にデータのある部分を取り出す
    size = tf.to_int32(features['size'])
    flux = tf.reshape(features['flux'], [6, max_length])[:, :size]
    # flux_err = tf.reshape(features['flux_err'], [6, max_length])[:, :size]
    # flux = flux + flux_err * tf.random_normal(shape=tf.shape(flux))
    # 場所をランダムに設定
    pad1 = tf.random_uniform(
        shape=[], maxval=tf.to_int32(max_length) - size, dtype=tf.int32
    )
    pad2 = tf.to_int32(max_length) - size - pad1
    flux_pad = tf.pad(flux, [(0, 0), (pad1, pad2)])
    flux_pad = tf.reshape(flux_pad, [6, max_length])

    if use_mask:
        # ランダムにマスクを選択
        r = tf.random_uniform(
            shape=[], maxval=tf.shape(mask_pattern_list)[0], dtype=tf.int32
        )
        mask_size = mask_size_list[r]
        mask = mask_pattern_list[r, :, :mask_size]
        # 場所をランダムに設定
        mask_pad1 = tf.random_uniform(
            shape=[], maxval=max_length - mask_size, dtype=tf.int32
        )
        mask_pad2 = max_length - mask_size - mask_pad1
        mask_pad = tf.pad(mask, [(0, 0), (mask_pad1, mask_pad2)])

        observation = tf.to_float(mask_pad) * flux_pad

        d = {'x': observation, 'mask': mask_pad, 'target': features['target'],
             'object_id': features['object_id'], 'sub_id': features['sub_id']}
    else:
        mask = tf.pad(tf.ones_like(flux), [(0, 0), (pad1, pad2)])
        mask = tf.reshape(mask, [6, max_length])

        d = {'x': flux_pad, 'mask': mask, 'target': features['target'],
             'object_id': features['object_id'], 'sub_id': features['sub_id']}

    if use_magnitude:
        d['x'] = tf.asinh(0.5 * d['x'])

    return d


class VariationalAutoEncoder(snt.AbstractModule):
    def __init__(self):
        super().__init__()

    def _build(self, inputs, is_training):
        h = snt.Linear(output_size=64)(inputs)
        h = tf.nn.selu(h)
        h = snt.Linear(output_size=16)(h)
        h = snt.LayerNorm()(h)
        h = tf.nn.selu(h)

        mu = snt.Linear(output_size=2)(h)
        tmp = snt.Linear(output_size=2)(h)
        sigma = tf.nn.softplus(tmp)

        z = mu + sigma * tf.random_normal(shape=tf.shape(sigma))

        h = snt.Linear(output_size=16)(z)
        h = snt.LayerNorm()(h)
        h = tf.nn.selu(h)

        outputs = snt.Linear(output_size=inputs.get_shape()[1])(h)

        return outputs


@tfmpl.figure_tensor
def draw_light_curve(x, y, mask):
    palette = sns.color_palette()

    figs = tfmpl.create_figures(n=4, figsize=(16, 6))
    for i, fig in enumerate(figs):
        ax = fig.add_subplot(111)

        tmp_x = np.reshape(x[i], [6, -1])
        tmp_y = np.reshape(y[i], [6, -1])
        tmp_mask = np.reshape(mask[i], [6, -1])

        flag = tmp_mask != 0
        t = np.arange(tmp_x.shape[1])[flag]
        tmp_x = tmp_x[:, flag]
        tmp_y = tmp_y[:, flag]
        for j in range(6):
            ax.plot(
                t, tmp_x, ls=':', label='input{}'.format(i), color=palette[i]
            )
            ax.plot(
                t, tmp_y, label='output{}'.format(i), color=palette[i]
            )
        ax.grid()
        ax.legend(loc='best')

    return figs


@click.command()
@click.option('--data-dir', type=click.Path(exists=True))
@click.option('--model-dir', type=click.Path())
@click.option('--mask-path', type=click.Path(exists=True))
@click.option('--batch-size', type=int, default=200)
@click.option('--epochs', type=int, default=100)
@click.option('--max-length', type=int, default=1400)
@click.option('--use-mask', is_flag=True)
@click.option('--use-magnitude', is_flag=True)
def cmd(data_dir, model_dir, mask_path, batch_size, epochs, max_length,
        use_mask, use_magnitude):
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    iterator, next_element = load_data(
        data_dir=data_dir, mask_path=mask_path, max_length=max_length,
        count=epochs, shuffle=True, batch_size=batch_size,
        use_mask=use_mask, use_magnitude=use_magnitude
    )

    model = VariationalAutoEncoder()
    x = snt.BatchFlatten()(next_element['x'])
    mask = snt.BatchFlatten()(next_element['mask'])
    y = model(x, False)
    squared_difference = tf.squared_difference(x, y) * mask
    loss = tf.reduce_sum(squared_difference, axis=1)
    n = tf.reduce_sum(mask, axis=1)
    mean_loss = loss / n

    image_op = draw_light_curve(x[:4], y[:4], mask[:4])

    with tf.variable_scope('metrics') as vs:
        loss_op = tf.metrics.mean(mean_loss)

        local_variables = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES
        )
        reset_op = tf.variables_initializer(local_variables)
    update_op = loss_op[1]
    summary_op = tf.summary.merge([
        tf.summary.scalar('loss', loss_op[0])
    ])
    summary_image_op = tf.summary.merge([
        tf.summary.image('light_curve', image_op)
    ])

    global_step = tf.train.get_or_create_global_step()

    optimizer = tf.train.AdamOptimizer()
    opt_op = optimizer.minimize(
        tf.reduce_mean(mean_loss), global_step=global_step
    )

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(str(model_dir))

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if checkpoint:
            path = checkpoint.model_checkpoint_path
            saver.restore(sess=sess, save_path=path)

            sess.run(tf.local_variables_initializer())
        else:
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

        sess.run(iterator.initializer)

        progress = tqdm()

        n = 50
        while True:
            try:
                for i in range(n - 1):
                    sess.run([opt_op, update_op])
                    progress.update()

                _, _, summary_image = sess.run([
                    opt_op, update_op, summary_image_op
                ])
                progress.update()

                step = sess.run(global_step)
                writer.add_summary(summary=summary_image, global_step=step)

                summary = sess.run(summary_op)
                writer.add_summary(summary=summary, global_step=step)
                sess.run(reset_op)

                saver.save(sess=sess, save_path=str(model_dir / 'model'),
                           global_step=global_step)
            except tf.errors.OutOfRangeError:
                step = sess.run(global_step)
                summary = sess.run(summary_op)
                writer.add_summary(summary=summary, global_step=step)
                sess.run(reset_op)

                saver.save(sess=sess, save_path=str(model_dir / 'model'),
                           global_step=global_step)
                break


def main():
    cmd()


if __name__ == '__main__':
    main()
