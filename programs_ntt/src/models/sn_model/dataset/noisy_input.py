#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from .magnitude import compute_magnitude

__author__ = 'Yasuhiro Imoto'
__date__ = '29/9/2017'


def make_noisy_magnitude(flux, flux_err, blackout_rate, outlier_rate,
                         method='modified'):
    flux = tf.cast(flux, tf.float32)
    flux_err = tf.cast(flux_err, tf.float32)

    noise = tf.random_normal(tf.shape(flux_err)) * flux_err
    magnitude = compute_magnitude(flux + noise, method=method)

    # blackout_rateの確率でFalseを生成する
    blackout_mask = tf.random_uniform(tf.shape(magnitude)) > blackout_rate
    # Falseの部分を0にする
    magnitude = magnitude * tf.to_float(blackout_mask)

    # outlier_rateの確率でTrueを生成する
    outlier_mask = tf.random_uniform(tf.shape(magnitude)) < outlier_rate
    # Trueの部分のみノイズを加える
    outlier = tf.random_normal(tf.shape(magnitude)) * tf.to_float(outlier_mask)
    magnitude = magnitude + outlier

    return magnitude
