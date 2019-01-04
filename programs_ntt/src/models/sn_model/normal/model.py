#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf
try:
    from tensorflow import keras
except ImportError:
    from tensorflow.contrib.keras.python import keras

from sn_model.dataset import make_noisy_magnitude

__author__ = 'Yasuhiro Imoto'
__date__ = '06/12/2017'


def make_model(input_size, hidden_size, output_size, dropout_rate):
    x = keras.Input(shape=(input_size,))
    dense1 = keras.layers.Dense(units=hidden_size, name='my_dense1')(x)
    drop1 = keras.layers.Dropout(rate=dropout_rate, name='my_dropout1')(dense1)

    # highway1
    transform1 = keras.layers.Dense(
        units=hidden_size, name='my_transform1',
        activation=keras.activations.selu
    )(drop1)
    gate1 = keras.layers.Dense(
        units=hidden_size, activation=keras.activations.sigmoid,
        name='my_gate1'
    )(drop1)
    highway1 = keras.layers.Lambda(
        lambda v: v[0] * v[1] + (1 - v[0]) * v[2]
    )([gate1, transform1, drop1])

    # highway2
    transform2 = keras.layers.Dense(
        units=hidden_size, name='my_transform2',
        activation=keras.activations.selu
    )(highway1)
    gate2 = keras.layers.Dense(
        units=hidden_size, activation=keras.activations.sigmoid,
        name='my_gate2'
    )(highway1)
    highway2 = keras.layers.Lambda(
        lambda v: v[0] * v[1] + (1 - v[0]) * v[2]
    )([gate2, transform2, highway1])

    y = keras.layers.Dense(units=output_size, name='my_dense2',
                           activation=keras.activations.softmax)(highway2)
    model = keras.models.Model(inputs=x, outputs=y)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def setup_model(dataset, classifier_hidden_size, output_size, band_data,
                mean, std, blackout_rate, outlier_rate, dropout_rate, method,
                trainable, training, reuse, *args, **kwargs):
    source_input, _ = dataset

    # 順序を固定
    band_list = list(band_data.keys())
    band_list.sort()

    # fluxをmagnitudeに変換
    input_list = []
    for band in band_list:
        # バンドごとに変換
        flux = '{}-flux'.format(band)
        flux_err = '{}-flux_err'.format(band)

        noisy_magnitude = make_noisy_magnitude(
            source_input[flux], source_input[flux_err],
            blackout_rate, outlier_rate, method
        )
        noisy_magnitude = (noisy_magnitude - mean[band]) / std[band]
        input_list.append(noisy_magnitude)
    # 横方向にバンドごとに変換したデータを繋げる
    magnitude = tf.concat(input_list, axis=1)

    with tf.variable_scope('model', reuse=reuse):
        dense1 = keras.layers.Dense(units=classifier_hidden_size,
                                    trainable=trainable)(magnitude)
        drop1 = keras.layers.Dropout(rate=dropout_rate,
                                     trainable=trainable)(dense1)

        # highway1
        transform1 = keras.layers.Dense(units=classifier_hidden_size,
                                        activation=keras.activations.selu,
                                        trainable=trainable)(drop1)
        gate1 = keras.layers.Dense(units=classifier_hidden_size,
                                   activation=keras.activations.sigmoid,
                                   trainable=trainable)(drop1)
        highway1 = gate1 * transform1 + (1 - gate1) * drop1

        # highway2
        transform2 = keras.layers.Dense(units=classifier_hidden_size,
                                        activation=keras.activations.selu,
                                        trainable=trainable)(highway1)
        gate2 = keras.layers.Dense(units=classifier_hidden_size,
                                   activation=keras.activations.sigmoid,
                                   trainable=trainable)(highway1)
        highway2 = gate2 * transform2 + (1 - gate2) * highway1

        output = keras.layers.Dense(units=output_size,
                                    trainable=trainable)(highway2)
        probability = tf.nn.softmax(output)

    label = source_input['label']
    one_hot_label = tf.one_hot(label, output_size)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=one_hot_label, logits=output
    ))
    accuracy = tf.reduce_mean(keras.metrics.categorical_accuracy(
        y_true=one_hot_label, y_pred=output
    ))

    if training:
        name = 'train'
    else:
        name = 'validation'

    with tf.variable_scope('{}_model_metrics'.format(name)) as vs:
        mean_loss, update_mean_loss = tf.metrics.mean(loss)
        mean_accuracy, update_mean_accuracy = tf.metrics.mean(accuracy)

        local_variables = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_metrics = tf.variables_initializer(local_variables)

    summary_list = [
        tf.summary.scalar('{}/loss'.format(name), mean_loss),
        tf.summary.scalar('{}/accuracy'.format(name), mean_accuracy)
    ]
    summary = tf.summary.merge(summary_list)

    if training:
        collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(collection):
            optimizer_op = tf.train.AdamOptimizer().minimize(loss)

        update_op = tf.group(optimizer_op, update_mean_loss,
                             update_mean_accuracy)

        global_step = tf.train.get_or_create_global_step()
        increment_step = tf.assign_add(global_step, 1)
    else:
        update_op = tf.group(update_mean_loss, update_mean_accuracy)

        increment_step = tf.no_op()

    d = {'summary': summary, 'reset': reset_metrics,
         'classifier_output': probability, 'classifier': update_op,
         'increment_step': increment_step}
    return d
