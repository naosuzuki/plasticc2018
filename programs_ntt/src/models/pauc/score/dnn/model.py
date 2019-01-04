#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
入力データをスカラーで順位付けするDNNの関数
"""

import tensorflow as tf
try:
    from tensorflow import keras
except ImportError:
    from tensorflow.contrib.keras.python import keras

__author__ = 'Yasuhiro Imoto'
__date__ = '11/12/2017'


def get_basic_model1(input_size, hidden_size):
    with tf.variable_scope('basic_score'):
        x = keras.Input(shape=(input_size,))
        h = keras.layers.Dense(units=hidden_size, name='my_dense1')(x)
        h = keras.layers.PReLU(name='my_prelu1')(h)
        y = keras.layers.Dense(units=1, use_bias=False, name='my_dense2')(h)
    model = keras.models.Model(x, y)
    return model


def basic_score1(input_size, x, hidden_size, reuse):
    # デフォルトの初期値では、極端な値のスコアになる可能性がある
    # その場合にシグモイド関数の勾配がほぼ0で更新できない可能性があるので、
    # 小さな値を設定する
    scale = 1.0 / input_size
    initializer = tf.truncated_normal_initializer(stddev=scale)

    with tf.variable_scope('basic_score', reuse=reuse):
        if isinstance(x, tf.SparseTensor):
            w = tf.get_variable('my_dense1/kernel',
                                shape=[input_size, hidden_size],
                                initializer=initializer)
            b = tf.get_variable('my_dense1/bias', shape=[hidden_size],
                                initializer=tf.zeros_initializer())
            h = tf.sparse_tensor_dense_matmul(x, w) + b
        else:
            h = keras.layers.Dense(units=hidden_size, name='my_dense1',
                                   kernel_initializer=initializer)(x)
        h = keras.layers.PReLU()(h)
        y = keras.layers.Dense(units=1, name='my_dense2')(h)
        # 1階のテンソルにする
        # squeezeでもいいかもしれないが、要素数が1の時に変わってくる
        y = tf.reshape(y, [-1])
    return y


def basic_score2(input_size, x_positive, x_negative, hidden_size, reuse):
    # バッチ正則化に対応
    with tf.variable_scope('basic_score', reuse=reuse):
        x = tf.concat([x_positive, x_negative], axis=0)
        h = keras.layers.Dense(units=hidden_size, use_bias=False,
                               name='my_dense1')(x)

        norm1 = keras.layers.BatchNormalization()(h)
        highway1 = make_highway(hidden_size, norm1, 'highway1')
        norm2 = keras.layers.BatchNormalization()(highway1)
        highway2 = make_highway(hidden_size, norm2, 'highway2')

        norm3 = keras.layers.BatchNormalization()(highway2)
        # 活性化関数を入れないので、バイアスは計算するだけ無駄
        y = keras.layers.Dense(units=1, use_bias=False,
                               name='my_dense2')(norm3)
        y = tf.reshape(y, [-1])

        y_positive, y_negative = tf.split(y, [tf.shape(x_positive)[0],
                                              tf.shape(x_negative)[0]])
    return y_positive, y_negative


def make_highway(input_size, x, scope_name):
    with tf.name_scope(scope_name):
        transformed = keras.layers.Dense(units=input_size)(x)
        # 特に目的があるわけではないが、何となく良さそうなのでELUを使ってみる
        transformed = keras.layers.ELU()(transformed)

        gate = keras.layers.Dense(units=input_size,
                                  activation=keras.activations.sigmoid)(x)

        y = gate * transformed + (1 - gate) * x
    return y
