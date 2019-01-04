#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf
try:
    from tensorflow import keras
except ImportError:
    from tensorflow.contrib.keras.python import keras

from sn_model.dataset import compute_magnitude, make_noisy_magnitude

__author__ = 'Yasuhiro Imoto'
__date__ = '29/9/2017'


class DenoisingFeature(object):
    def __init__(self, mean, std, band, input_size, hidden_size,
                 blackout_rate, outlier_rate, method='modified'):
        self.mean = mean
        self.std = std

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.blackout_rate = blackout_rate
        self.outlier_rate = outlier_rate
        self.method = method

        self.name = 'denoising_feature-{}'.format(band)

    def __call__(self, flux, flux_err, reuse=True, trainable=True,
                 training=True):
        return self.encode(flux, flux_err, reuse, trainable, training)

    def encode(self, flux, flux_err, source, reuse=True, trainable=True,
               training=True):
        with tf.variable_scope('{}/encoder'.format(self.name), reuse=reuse):
            if training and source:
                noisy_x = make_noisy_magnitude(
                    flux, flux_err, self.blackout_rate, self.outlier_rate,
                    method=self.method
                )
            else:
                noisy_x = compute_magnitude(flux, method=self.method)
            noisy_x = (noisy_x - self.mean) / self.std

            outputs = noisy_x
            for hidden_size in self.hidden_size:
                h = keras.layers.Dense(units=hidden_size,
                                       trainable=trainable)(outputs)
                outputs = keras.layers.PReLU(trainable=trainable)(h)
        return outputs

    def decode(self, feature, reuse=True, trainable=True,
               training=True):
        with tf.variable_scope('{}/decoder'.format(self.name), reuse=reuse):
            outputs = feature
            for hidden_size in self.hidden_size[-2::-1]:
                h = keras.layers.Dense(units=hidden_size,
                                       trainable=trainable)(outputs)
                outputs = keras.layers.PReLU(trainable=trainable)(h)
            outputs = keras.layers.Dense(units=self.input_size,
                                         trainable=trainable)(outputs)
        return outputs

    @property
    def variables(self):
        return [v for v in tf.global_variables() if self.name in v.name]

    def compute_loss(self, flux, feature, reuse=True, trainable=True,
                     training=True):
        tmp = compute_magnitude(flux)
        inputs = (tmp - self.mean) / self.std

        outputs = self.decode(feature, reuse=reuse, trainable=trainable,
                              training=training)
        return tf.reduce_mean(tf.squared_difference(inputs, outputs)), inputs


class MultipleDenoisingFeature(object):
    def __init__(self, band_data, mean, std,
                 input_size, hidden_size, output_size,
                 blackout_rate, outlier_rate, method='modified'):
        available_band_list = list(band_data.keys())
        available_band_list.sort()
        self.band_list = available_band_list

        self.features = {
            band: DenoisingFeature(mean[band], std[band], band,
                                   input_size[band], hidden_size,
                                   blackout_rate, outlier_rate, method=method)
            for band in available_band_list
        }
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.name = 'denoising_feature-multiple'

    def encode(self, dataset, source, reuse=True, trainable=True,
               training=True):
        encoded = []
        for band in self.band_list:
            flux = '{}-flux'.format(band)
            flux_err = '{}-flux_err'.format(band)

            o = self.features[band].encode(
                dataset[flux], dataset[flux_err], source, reuse=reuse,
                trainable=trainable, training=training)
            encoded.append(o)
        h = tf.concat(encoded, 1)

        with tf.variable_scope('{}/encoder'.format(self.name), reuse=reuse):
            outputs = keras.layers.Dense(units=self.output_size,
                                         trainable=trainable)(h)
            outputs = keras.layers.PReLU(trainable=trainable)(outputs)

        return outputs, encoded

    def decode(self, feature, reuse=True, trainable=True, training=True):
        input_size = sum(self.input_size[band] for band in self.band_list)

        with tf.variable_scope('{}/decoder'.format(self.name), reuse=reuse):
            # 2段階で再構成
            h = keras.layers.Dense(units=(self.output_size + input_size) // 2,
                                   trainable=trainable)(feature)
            h = keras.layers.PReLU(trainable=trainable)(h)
            outputs = keras.layers.Dense(units=input_size,
                                         trainable=trainable)(h)
        return outputs

    @property
    def variables(self):
        return [v for v in tf.global_variables() if self.name in v.name]

    def compute_loss(self, dataset, feature, band_feature, reuse=True,
                     trainable=True, training=True):
        band_loss = []
        input_list = []
        for i, band in enumerate(self.band_list):
            flux = dataset['{}-flux'.format(band)]
            loss, inputs = self.features[band].compute_loss(
                flux, band_feature[i], reuse=reuse, trainable=trainable,
                training=training)
            band_loss.append(loss)
            input_list.append(inputs)

        inputs = tf.concat(input_list, 1)
        outputs = self.decode(feature, reuse=reuse, trainable=trainable,
                              training=training)

        loss = tf.reduce_mean(tf.squared_difference(inputs, outputs))
        return loss, band_loss
