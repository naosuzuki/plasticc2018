#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf

__author__ = 'Yasuhiro Imoto'
__date__ = '06/10/2017'


def generate_virtual_adversarial_perturbation(x, p_logit, epsilon, xi,
                                              num_power_iterations,
                                              feature, classifier,
                                              training=True):
    d = tf.random_normal(tf.shape(x))

    for _ in range(num_power_iterations):
        d = xi * tf.nn.l2_normalize(d, 1)

        f = feature(x + d, trainable=True, training=training)
        q_logit = classifier(f, trainable=True, training=training)
        d_kl = compute_kl_divergence(p_logit, q_logit)

        g = tf.gradients(d_kl, d)[0]
        d = tf.stop_gradient(g)

    return epsilon * d


def compute_kl_divergence(p_logit, q_logit):
    p = tf.nn.softmax(p_logit, 1)
    log_p = tf.nn.log_softmax(p_logit, 1)
    log_q = tf.nn.log_softmax(q_logit, 1)
    p_log_p = tf.reduce_mean(tf.reduce_sum(p * log_p, 1))
    p_log_q = tf.reduce_mean(tf.reduce_sum(p * log_q, 1))
    return p_log_p - p_log_q


def compute_virtual_adversarial_loss(x, p_logit, epsilon, xi,
                                     num_power_iterations, feature, classifier,
                                     training=True):
    r_vadv = generate_virtual_adversarial_perturbation(x, p_logit, epsilon, xi,
                                                       num_power_iterations,
                                                       feature, classifier,
                                                       training=training)
    f = feature(x + r_vadv, trainable=True, training=training)
    q_logit = classifier(f, trainable=True, training=training)

    p_logit = tf.stop_gradient(p_logit)

    loss = compute_kl_divergence(p_logit, q_logit)
    return loss
