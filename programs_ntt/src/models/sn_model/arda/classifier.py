#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf
try:
    from tensorflow import keras
except ImportError:
    from tensorflow.contrib.keras.python import keras

__author__ = 'Yasuhiro Imoto'
__date__ = '01/9/2017'


class Classifier(object):
    def __init__(self, hidden_size, output_size, dropout_rate):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.name = 'classifier'

    def __call__(self, x, reuse=True, trainable=True, training=True):
        with tf.variable_scope(self.name, reuse=reuse):
            dense0 = keras.layers.Dense(units=self.hidden_size,
                                        trainable=trainable)(x)
            dense0 = keras.layers.PReLU(trainable=trainable)(dense0)

            h = keras.layers.BatchNormalization(
                trainable=trainable)(dense0)
            # h = tf.layers.batch_normalization(dense0,
            #                                   trainable=trainable,
            #                                   training=training)

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

            batch1 = keras.layers.BatchNormalization(
                trainable=trainable)(highway1)
            # batch1 = tf.layers.batch_normalization(highway1,
            #                                        trainable=trainable,
            #                                        training=training)

            dense2 = keras.layers.Dense(units=self.output_size,
                                        trainable=trainable)(batch1)
            return dense2

    @property
    def variables(self):
        return [v for v in tf.global_variables() if self.name in v.name]
