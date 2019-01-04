#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from .classifier import Classifier
from .critic import Critic
from .feature_generator import FeatureGenerator

from sn_model.dataset import compute_magnitude, make_noisy_magnitude

__author__ = 'Yasuhiro Imoto'
__date__ = '04/9/2017'


def setup_model(dataset, mean, std, band_data, feature_size,
                critic_hidden_size, classifier_hidden_size,
                output_size, dropout_rate, blackout_rate, outlier_rate,
                lambda1, lambda2, lambda3, training=True, reuse=False,
                method='modified'):
    source_input, target_input = dataset

    band_list = list(band_data.keys())
    band_list.sort()
    input_size = band_data

    # fluxをmagnitudeに変換
    source_input_list = []
    target_input_list = []
    for band in band_list:
        # バンドごとに変換
        flux = '{}-flux'.format(band)
        flux_err = '{}-flux_err'.format(band)

        if training:
            source_magnitude = make_noisy_magnitude(
                source_input[flux], source_input[flux_err],
                blackout_rate, outlier_rate, method
            )
        else:
            source_magnitude = compute_magnitude(source_input[flux], method)
        source_magnitude = (source_magnitude - mean[band]) / std[band]
        source_input_list.append(source_magnitude)

        target_magnitude = compute_magnitude(target_input[flux], method)
        target_magnitude = (target_magnitude - mean[band]) / std[band]
        target_input_list.append(target_magnitude)
    # 横方向にバンドごとに変換したデータを繋げる
    source_magnitude = tf.concat(source_input_list, axis=1)
    target_magnitude = tf.concat(target_input_list, axis=1)

    feature = FeatureGenerator(feature_size, dropout_rate)
    critic = Critic(critic_hidden_size, dropout_rate)
    classifier = Classifier(classifier_hidden_size, output_size, dropout_rate)

    # critic
    critic_update, critic_summary_list, critic_reset_metrics = setup_critic(
        source_magnitude, target_magnitude, feature, critic, reuse, training,
        lambda2
    )

    # classifier
    tmp = setup_classifier(
        source_magnitude, target_magnitude, source_input['label'], output_size,
        feature, critic, classifier, reuse, training, lambda1
    )
    classifier_update, classifier_summary_list = tmp[:2]
    classifier_reset_metrics, classifier_output = tmp[2:]

    if training:
        global_step = tf.train.get_or_create_global_step()
        increment_step = tf.assign_add(global_step, 1)
    else:
        increment_step = tf.no_op()

    summary = tf.summary.merge(critic_summary_list + classifier_summary_list)
    reset = tf.group(critic_reset_metrics, classifier_reset_metrics)

    return {'critic': critic_update, 'classifier': classifier_update,
            'summary': summary, 'reset': reset,
            'classifier_output': classifier_output,
            'increment_step': increment_step}


def setup_critic(source_input, target_input, feature, critic, reuse, training,
                 lambda2):
    source_feature = feature(source_input, reuse=reuse, trainable=False,
                             training=training)
    source_critic = critic(source_feature, reuse=reuse, trainable=True,
                           training=training)
    target_feature = feature(target_input, trainable=False, training=training)
    target_critic = critic(target_feature, trainable=True, training=training)
    wasserstein_distance = (tf.reduce_mean(source_critic) -
                            tf.reduce_mean(target_critic))
    # penalty
    epsilon = tf.random_uniform([tf.shape(source_feature)[0], 1],
                                minval=0, maxval=1)
    h = source_feature * epsilon + target_feature * (1 - epsilon)
    c = critic(h, trainable=True, training=training)
    g = tf.gradients(c, h)[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(g), axis=1))
    penalty = tf.reduce_mean(tf.square(norm - 1)) * lambda2

    loss = -wasserstein_distance + penalty

    if training:
        name = 'train'
    else:
        name = 'validation'

    with tf.variable_scope('{}_critic_metrics'.format(name)) as vs:
        mean_loss, update_mean_loss = tf.metrics.mean(loss)
        mean_distance, update_mean_distance = tf.metrics.mean(
            wasserstein_distance)
        mean_penalty, update_mean_penalty = tf.metrics.mean(penalty)

        variables = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_metrics = tf.variables_initializer(variables)

    summary_list = [
        tf.summary.scalar('{}/critic/loss'.format(name), mean_loss),
        tf.summary.scalar('{}/critic/Wasserstein-distance'.format(name),
                          mean_distance),
        tf.summary.scalar('{}/critic/penalty'.format(name), mean_penalty)
    ]

    if training:
        collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(collection):
            optimize = tf.train.AdamOptimizer().minimize(
                loss, var_list=critic.variables)

        critic_update = tf.group(optimize, update_mean_loss,
                                 update_mean_distance, update_mean_penalty)
    else:
        critic_update = tf.group(update_mean_loss,
                                 update_mean_distance, update_mean_penalty)

    return critic_update, summary_list, reset_metrics


def setup_classifier(source_input, target_input, y, output_size, feature,
                     critic, classifier, reuse, training, lambda1):
    source_feature = feature(source_input, trainable=True, training=training)
    source_critic = critic(source_feature, trainable=False, training=training)
    target_feature = feature(target_input, trainable=True, training=training)
    target_critic = critic(target_feature, trainable=False, training=training)
    wasserstein_distance = (tf.reduce_mean(source_critic) -
                            tf.reduce_mean(target_critic))
    source_classifier = classifier(source_feature, reuse=reuse,
                                   trainable=True, training=training)

    label = tf.one_hot(y, output_size)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=label, logits=source_classifier))

    loss = cross_entropy + lambda1 * wasserstein_distance

    if training:
        name = 'train'
    else:
        name = 'validation'

    prediction = tf.cast(tf.argmax(source_classifier, axis=1), tf.int32)
    with tf.variable_scope('{}_classifier_metrics'.format(name)) as vs:
        mean_loss, update_mean_loss = tf.metrics.mean(loss)
        mean_distance, update_mean_distance = tf.metrics.mean(
            wasserstein_distance)
        mean_cross_entropy, update_mean_cross_entropy = tf.metrics.mean(
            cross_entropy)
        mean_accuracy, update_mean_accuracy = tf.metrics.accuracy(
            labels=y, predictions=prediction)

        variables = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_metrics = tf.variables_initializer(variables)

    summary_list = [
        tf.summary.scalar('{}/classifier/loss'.format(name), mean_loss),
        tf.summary.scalar('{}/classifier/Wasserstein-distance'.format(name),
                          mean_distance),
        tf.summary.scalar('{}/classifier/cross_entropy'.format(name),
                          mean_cross_entropy),
        tf.summary.scalar('{}/classifier/accuracy'.format(name),
                          mean_accuracy)
    ]

    if training:
        collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(collection):
            optimize = tf.train.AdamOptimizer().minimize(
                loss, var_list=feature.variables + classifier.variables)

        classifier_update = tf.group(
            optimize, update_mean_loss, update_mean_distance,
            update_mean_cross_entropy, update_mean_accuracy)
    else:
        classifier_update = tf.group(
            update_mean_loss, update_mean_distance, update_mean_cross_entropy,
            update_mean_accuracy)

    return classifier_update, summary_list, reset_metrics, source_classifier
