#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import xarray as xr
import numpy as np
try:
    from tensorflow.data import Dataset
except ImportError:
    from tensorflow.contrib.data import Dataset

from .magnitude import compute_magnitude

__author__ = 'Yasuhiro Imoto'
__date__ = '29/9/2017'


def compute_moments_from_file(file_name, method='modified'):
    ds = xr.open_dataset(file_name)
    band_list = [band if isinstance(band, str) else band.decode()
                 for band in np.unique(ds.band.values)]

    x = tf.placeholder(tf.float32, [None, None])
    mu, sigma2 = tf.nn.moments(compute_magnitude(x, method=method), [0])
    sigma = tf.sqrt(sigma2)

    mean, std = {}, {}
    with tf.Session() as sess:
        for band in band_list:
            tmp = get_band_data(ds, band)
            flux = tmp['{}-flux'.format(band)]
            m, s = sess.run([mu, sigma], feed_dict={x: flux})

            mean[band] = m
            std[band] = s
    return mean, std


def load_data(file_name):
    ds = xr.open_dataset(file_name)

    name = [n.decode() if isinstance(n, bytes) else n
            for n in ds.sample.values]
    label = ds.label.values
    redshift = ds.redshift.values
    band_list = [band.decode() if isinstance(band, bytes) else band
                 for band in np.unique(ds.band.values)]

    return ds, name, redshift, label, band_list


def get_band_data(ds, band):
    flag = ds.band == band.encode()
    if not np.any(flag):
        flag = ds.band == band
    tmp = ds.where(flag, drop=True)
    flux = np.nan_to_num(tmp.flux.values)
    flux_err = np.nan_to_num(tmp.flux_err.values)

    flux_err[flux_err > 1] = 1

    return {'{}-flux'.format(band): flux, '{}-flux_err'.format(band): flux_err}


def make_dataset(file_name, epochs=-1, shuffle=True, return_length=False):
    ds, name, redshift, label, band_list = load_data(file_name)

    d = {'name': name, 'redshift': redshift, 'label': label}
    for band in band_list:
        tmp = get_band_data(ds, band)
        d.update(tmp)

    dataset = Dataset.from_tensor_slices(d)

    if shuffle:
        dataset = dataset.shuffle(10000)
    if epochs <= 0:
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(epochs)

    if return_length:
        return dataset, len(d['name'])
    return dataset


def make_domain_dataset(source_file_name, target_file_name,
                        batch_size, epochs=-1, shuffle=True):
    source_dataset = make_dataset(source_file_name, epochs, shuffle)
    target_dataset = make_dataset(target_file_name, -1, shuffle)

    dataset = Dataset.zip((source_dataset, target_dataset))
    if shuffle:
        dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)

    return dataset
