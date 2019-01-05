#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from .base import BaseGaussian

__author__ = 'Yasuhiro Imoto'
__date__ = '04/1/2018'


class DiagonalCovariance(BaseGaussian):
    def __init__(self, data_size, scope_name, mean, precision):
        super().__init__()

        with tf.variable_scope(scope_name):
            self.mean = tf.get_variable(
                'mu', shape=[data_size], dtype=tf.float32,
                initializer=tf.constant_initializer(mean)
            )
            self.precision = tf.get_variable(
                'sigma', shape=[data_size], dtype=tf.float32,
                initializer=tf.constant_initializer(precision)
            )

    def compute_ll_exponential(self, inputs):
        x = inputs - self.mean
        # mixtureを計算するときに2次元の方が都合がいい
        return -0.5 * tf.reduce_sum(tf.square(x * self.precision), axis=1,
                                    keep_dims=True)

    def compute_ll_determinant(self):
        # 対角要素が常に正である保証はないので、念のために二乗の形で計算する
        return 0.5 * tf.reduce_sum(tf.log(tf.square(self.precision) + 1e-6))
