#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf
try:
    from tensorflow import keras
except ImportError:
    from tensorflow.contrib.keras.python import keras

__author__ = 'Yasuhiro Imoto'
__date__ = '01/9/2017'


class Critic(object):
    def __init__(self, hidden_size, dropout_rate):
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.name = 'critic'

    def __call__(self, x, reuse=True, trainable=True, training=True):
        with tf.variable_scope(self.name, reuse=reuse):
            dense0 = keras.layers.Dense(units=self.hidden_size,
                                        trainable=trainable)(x)
            dense0 = keras.layers.PReLU(trainable=trainable)(dense0)
            h = tf.layers.dropout(dense0, self.dropout_rate,
                                  training=training)
            # highway
            identity1 = h
            transformed1 = keras.layers.Dense(units=self.hidden_size,
                                              trainable=trainable)(h)
            transformed1 = keras.layers.PReLU(
                trainable=trainable)(transformed1)

            gate1 = keras.layers.Dense(units=self.hidden_size,
                                       activation=keras.activations.sigmoid,
                                       trainable=trainable)(h)
            highway1 = gate1 * transformed1 + (1 - gate1) * identity1

            drop1 = tf.layers.dropout(highway1, self.dropout_rate,
                                      training=training)

            # バイアスはあってもなくても関係ないので、
            # テストしやすさのために削除
            dense1 = keras.layers.Dense(units=1, use_bias=False,
                                        trainable=trainable)(drop1)
            return dense1

    @property
    def variables(self):
        return [v for v in tf.global_variables() if self.name in v.name]
