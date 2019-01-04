#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from sn_model.ae.denoising import MultipleDenoisingFeature
from sn_model.arda.classifier import Classifier
from sn_model.arda.critic import Critic

__author__ = 'Yasuhiro Imoto'
__date__ = '29/9/2017'


def setup_model(dataset, mean, std, band_data, feature_size,
                critic_hidden_size, classifier_hidden_size,
                output_size, dropout_rate, blackout_rate, outlier_rate,
                lambda1, lambda2, lambda3, training=True, reuse=False,
                method='modified'):
    source_input, target_input = dataset

    band_list = list(band_data.keys())
    band_list.sort()
    input_size = band_data

    band_wise_encoder_size = [2]
    multi_feature = MultipleDenoisingFeature(
        band_data, mean, std, input_size, band_wise_encoder_size,
        feature_size, blackout_rate, outlier_rate, method=method
    )
    critic = Critic(critic_hidden_size, dropout_rate)
    classifier = Classifier(classifier_hidden_size, output_size, dropout_rate)

    tmp = setup_critic(multi_feature, dataset, critic, reuse, training,
                       lambda2)
    critic_update, critic_summary_list, critic_reset_metrics = tmp

    tmp = setup_classifier(multi_feature, dataset, critic, classifier,
                           source_input['label'],
                           reuse, training, lambda1, lambda3)
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


def setup_critic(multi_feature, dataset, critic, reuse, training, lambda2):
    source_dataset, target_dataset = dataset
    source_feature, _ = multi_feature.encode(source_dataset, source=True,
                                             reuse=reuse, trainable=False,
                                             training=training)
    target_feature, _ = multi_feature.encode(target_dataset, source=False,
                                             trainable=False,
                                             training=training)

    source_critic = critic(source_feature, reuse=reuse, trainable=True,
                           training=training)
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


def setup_classifier(multi_feature, dataset, critic, classifier, label,
                     reuse, training, lambda1, lambda3):
    source_dataset, target_dataset = dataset
    source_feature, band_feature = multi_feature.encode(source_dataset,
                                                        source=True,
                                                        trainable=True,
                                                        training=training)
    target_feature, _ = multi_feature.encode(target_dataset, source=False,
                                             trainable=True,
                                             training=training)

    multi_band_loss, band_loss = multi_feature.compute_loss(
        source_dataset, source_feature, band_feature, reuse=reuse,
        trainable=True, training=training)
    reconstuction_loss = (len(multi_feature.band_list) * multi_band_loss +
                          tf.reduce_sum(band_loss))

    source_critic = critic(source_feature, trainable=False, training=training)
    target_critic = critic(target_feature, trainable=False, training=training)
    wasserstein_distance = (tf.reduce_mean(source_critic) -
                            tf.reduce_mean(target_critic))

    source_classifier = classifier(source_feature, reuse=reuse, trainable=True,
                                   training=training)
    one_hot_label = tf.one_hot(label, classifier.output_size)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=one_hot_label, logits=source_classifier
    ))

    loss = (cross_entropy + lambda1 * wasserstein_distance +
            lambda3 * reconstuction_loss)

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
            labels=label, predictions=prediction)

        mean_reconstruction_loss, update_mean_reconstruction_loss = [], []
        for l in band_loss:
            tmp = tf.metrics.mean(l)
            mean_reconstruction_loss.append(tmp[0])
            update_mean_reconstruction_loss.append(tmp[1])
        mean_multi_loss, update_mean_multi_loss = tf.metrics.mean(
            multi_band_loss)

        variables = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_metrics = tf.variables_initializer(variables)

    summary_list = [
        tf.summary.scalar('{}/classifier/loss'.format(name),
                          mean_loss),
        tf.summary.scalar('{}/classifier/Wasserstein-distance'.format(name),
                          mean_distance),
        tf.summary.scalar('{}/classifier/cross_entropy'.format(name),
                          mean_cross_entropy),
        tf.summary.scalar('{}/classifier/accuracy'.format(name),
                          mean_accuracy),
        tf.summary.scalar(
            '{}/classifier/multi-reconstruction-loss'.format(name),
            mean_multi_loss)
    ]
    for band, reconstuction_loss in zip(multi_feature.band_list,
                                        mean_reconstruction_loss):
        summary = tf.summary.scalar(
            '{}/feature/reconstruction-loss/{}'.format(name, band),
            reconstuction_loss
        )
        summary_list.append(summary)

    if training:
        collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(collection):
            variables = multi_feature.variables
            for f in multi_feature.features.values():
                variables.extend(f.variables)

            optimize = tf.train.AdamOptimizer().minimize(
                loss, var_list=variables + classifier.variables)

        classifier_update = tf.group(
            optimize, update_mean_loss, update_mean_distance,
            update_mean_cross_entropy, update_mean_accuracy,
            update_mean_multi_loss, *update_mean_reconstruction_loss)
    else:
        classifier_update = tf.group(
            update_mean_loss, update_mean_distance, update_mean_cross_entropy,
            update_mean_accuracy, update_mean_multi_loss,
            *update_mean_reconstruction_loss)

    return classifier_update, summary_list, reset_metrics, source_classifier
