#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

__author__ = 'Yasuhiro Imoto'
__date__ = '28/9/2017'


def make_convolutional_denoising_autoencoder(
        flux, flux_err, mean, std, blackout_rate, outlier_rate,
        reuse=True, trainable=True, training=True):
    mag = (-2.5 * np.log(np.e) * tf.asinh(flux) - mean) / std

    noise = tf.random_normal(tf.shape(flux_err)) * flux_err
    noised_flux = flux + noise
    magnitude = -2.5 * np.log(np.e) * tf.asinh(noised_flux)

    blackout_mask = tf.random_uniform(tf.shape(magnitude)) < blackout_rate
    magnitude = tf.where(blackout_mask, tf.zeros_like(magnitude), magnitude)

    outlier_mask = tf.random_uniform(tf.shape(magnitude)) < outlier_rate
    magnitude = tf.where(outlier_mask,
                         tf.random_normal(tf.shape(magnitude)) + magnitude,
                         magnitude)

    inputs = (magnitude - mean) / std   # type: tf.Tensor
    shape = inputs.get_shape()
    inputs = tf.reshape(inputs, tf.stack([shape[0], 1, shape[1], 1]))

    # sharing weights of convolution layers
    # https://github.com/pkmital/tensorflow_tutorials/blob/master/python/
    # 09_convolutional_autoencoder.py
    with tf.variable_scope('convolutional_denoising_encoder') as vs:
        if reuse:
            vs.reuse_variables()

        conv1 = tf.layers.conv2d(inputs, filters=30, kernel_size=(1, 3),
                                 strides=(1, 2), padding='same',
                                 trainable=trainable)

        alpha = tf.get_variable('prelu/alpha', [conv1.get_shape()[-2], 1],
                                initializer=tf.constant_initializer(0.0),
                                dtype=tf.float32, trainable=trainable)
        pos = tf.nn.relu(conv1)
        neg = alpha * (conv1 - tf.abs(conv1)) * 0.5
        prelu1 = pos + neg

    with tf.variable_scope('convolutional_denoising_decoder') as vs:
        if reuse:
            vs.reuse_variables()

        deconv1 = tf.layers.conv2d_transpose(prelu1, filters=1,
                                             kernel_size=(1, 3),
                                             strides=(1, 2), padding='same',
                                             trainable=trainable,
                                             activation=tf.nn.relu)
    x = tf.squeeze(deconv1)
    cost = tf.reduce_mean(tf.squared_difference(mag, x))
    return cost
