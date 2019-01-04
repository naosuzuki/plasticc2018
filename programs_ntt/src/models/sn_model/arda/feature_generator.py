#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf
try:
    from tensorflow import keras
except ImportError:
    from tensorflow.contrib.keras.python import keras

__author__ = 'Yasuhiro Imoto'
__date__ = '01/9/2017'


class FeatureGenerator(object):
    def __init__(self, hidden_size, dropout_rate):
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.name = 'feature'

    def __call__(self, x, reuse=True, trainable=True, training=True):
        with tf.variable_scope(self.name, reuse=reuse):
            dense1 = keras.layers.Dense(units=self.hidden_size,
                                        trainable=trainable)(x)
            dense1 = keras.layers.PReLU(trainable=trainable)(dense1)

            drop1 = keras.layers.Dropout(self.dropout_rate,
                                         trainable=trainable)(dense1)

            h = drop1
            transformed1 = keras.layers.Dense(units=self.hidden_size,
                                              trainable=trainable)(h)
            transformed1 = keras.layers.PReLU(trainable=trainable
                                              )(transformed1)
            gate1 = keras.layers.Dense(units=self.hidden_size,
                                       activation=keras.activations.sigmoid,
                                       trainable=trainable)(h)
            highway1 = gate1 * transformed1 + (1 - gate1) * transformed1

            transformed2 = keras.layers.Dense(units=self.hidden_size,
                                              trainable=trainable)(highway1)
            transformed2 = keras.layers.PReLU(trainable=trainable
                                              )(transformed2)
            gate2 = keras.layers.Dense(units=self.hidden_size,
                                       activation=keras.activations.sigmoid,
                                       trainable=trainable)(highway1)
            highway2 = gate2 * transformed2 + (1 - gate2) * highway1

            drop2 = keras.layers.Dropout(self.dropout_rate,
                                         trainable=trainable)(highway2)
            return drop2

    @property
    def variables(self):
        return [v for v in tf.global_variables() if self.name in v.name]
