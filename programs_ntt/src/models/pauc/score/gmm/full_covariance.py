#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from .base import BaseGaussian

__author__ = 'Yasuhiro Imoto'
__date__ = '04/1/2018'


class FullCovariance(BaseGaussian):
    def __init__(self, data_size, scope_name, mean, precision):
        super().__init__()

        self.scope_name = scope_name
        with tf.variable_scope(scope_name):
            self.mean = tf.get_variable(
                'mu', shape=[data_size], dtype=tf.float32,
                initializer=tf.constant_initializer(mean)
            )
            # 実際に使うのは上三角行列だけ
            # sklearnで初期値を求めるので、そちらと同じ形にする
            self.precision = tf.get_variable(
                'sigma', shape=[data_size, data_size], dtype=tf.float32,
                initializer=tf.constant_initializer(precision)
            )
            # U U^Tで精度行列
            self.u = tf.matrix_band_part(self.precision, 0, -1)

    def compute_ll_exponential(self, inputs):
        x = inputs - self.mean
        m = tf.matmul(x, self.u)
        # mixtureを計算するときに2次元の方が都合がいい
        return -0.5 * tf.reduce_sum(tf.square(m), axis=1, keep_dims=True)

    def compute_ll_determinant(self):
        # 対角成分を取り出す
        d = tf.matrix_diag_part(self.u)
        # tf.log(d + 1e-6)でいい気がするが、
        # 勾配法のステップサイズによっては対角成分が負になるかもしれない
        # return tf.reduce_sum(tf.log(d + 1e-6))
        return 0.5 * tf.reduce_sum(tf.log(tf.square(d) + 1e-6))
