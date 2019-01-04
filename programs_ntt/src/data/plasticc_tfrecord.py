#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from pathlib import Path

import click
import numpy as np
import pandas as pd
import sonnet as snt
import tensorflow as tf
from tqdm import tqdm

__author__ = 'Yasuhiro Imoto'
__date__ = '27/12/2018'


def make_record(data_path, record_path, max_length, target, class_id):
    options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP
    )
    with tf.python_io.TFRecordWriter(record_path, options=options) as writer:
        with pd.HDFStore(data_path) as store:
            redshift = store['/redshift']
            for key in tqdm(store.keys(), total=redshift.size):
                if key == '/redshift':
                    continue
                df = store[key]
                df = df[df['interpolated']]

                flux = df.pivot(
                    index='mjd', columns='passband', values='flux'
                )
                flux_err = df.pivot(
                    index='mjd', columns='passband', values='flux_err'
                )
                size = len(flux)

                flux = np.pad(
                    flux, [(0, 0), (0, max_length - size)], 'constant'
                )
                flux_err = np.pad(
                    flux_err, [(0, 0), (0, max_length - size)], 'constant'
                )

                object_id = df['object_id'].values[0]
                sub_id = df['sub_id'].values[0]

                z = redshift.loc[object_id, sub_id]

                example = make_example(
                    flux=flux, flux_err=flux_err, size=size, target=target,
                    class_id=class_id, object_id=object_id, sub_id=sub_id,
                    redshift=z
                )

                writer.write(example.SerializeToString())


def make_example(flux, flux_err, size, target, class_id, object_id, sub_id,
                 redshift):
    example = tf.train.Example(features=tf.train.Features(feature={
        'flux': to_float(flux.astype(np.float32).flatten()),
        'flux_err': to_float(flux_err.astype(np.float32).flatten()),
        'size': to_int64(size),
        'target': to_int64(target),
        'class_id': to_int64(class_id),
        'object_id': to_int64(object_id),
        'sub_id': to_int64(sub_id),
        'redshift': to_float([np.float32(redshift)])
    }))
    return example


def to_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def to_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


@click.command()
@click.option('--data-path', type=click.Path(exists=True))
@click.option('--record-path', type=click.Path())
@click.option('--max-length', type=int, default=1200)
@click.option('--target', type=int)
@click.option('--class-id', type=int)
def cmd(data_path, record_path, max_length, target, class_id):
    make_record(
        data_path=data_path, record_path=record_path, max_length=max_length,
        target=target, class_id=class_id
    )


def main():
    cmd()


if __name__ == '__main__':
    main()
