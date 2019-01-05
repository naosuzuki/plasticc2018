#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import numpy as np

from sn_model.dataset import make_noisy_magnitude, compute_magnitude

__author__ = 'Yasuhiro Imoto'
__date__ = '12/1/2018'


def convert_data(dataset, mean, std, band_data, method='modified',
                 blackout_rate=0, outlier_rate=0, train=True,
                 use_redshift=True):
    # 順序を固定
    band_list = list(band_data.keys())
    band_list.sort()

    # fluxをmagnitudeに変換
    input_list = []
    for band in band_list:
        # バンドごとに変換
        flux = '{}-flux'.format(band)
        flux_err = '{}-flux_err'.format(band)

        if train:
            magnitude = make_noisy_magnitude(
                dataset[flux], dataset[flux_err],
                blackout_rate, outlier_rate, method
            )
        else:
            magnitude = compute_magnitude(dataset[flux], method=method)
        magnitude = (magnitude - mean[band]) / std[band]
        input_list.append(magnitude)
    if use_redshift:
        input_list.append(tf.reshape(tf.to_float(dataset['redshift']), [1]))
    # 横方向にバンドごとに変換したデータを繋げる
    x = tf.concat(input_list, axis=0)
    y = tf.to_int32(dataset['label'])
    name = dataset['name']

    features = {'x': x, 'y': y, 'name': name}
    return features
