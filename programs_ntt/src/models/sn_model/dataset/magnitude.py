#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

__author__ = 'Yasuhiro Imoto'
__date__ = '08/12/2017'


def compute_magnitude(flux, method='modified'):
    if method == 'modified':
        return compute_magnitude_modified(flux)
    else:
        return compute_magnitude_traditional(flux)


def compute_magnitude_modified(flux):
    a = tf.constant(2.5 * np.log10(np.e), dtype=tf.float32)
    magnitude = -a * tf.asinh(tf.cast(tf.multiply(flux, 0.5), tf.float32))
    return magnitude


def compute_magnitude_traditional(flux):
    a = tf.constant(2.5 * np.log10(np.e), dtype=tf.float32)
    tmp = tf.maximum(flux, tf.ones_like(flux) * 0.1)
    magnitude = -a * tf.log(tf.cast(tmp, tf.float32))
    return magnitude
