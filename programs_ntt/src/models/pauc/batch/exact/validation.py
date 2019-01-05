#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf

__author__ = 'Yasuhiro Imoto'
__date__ = '28/12/2017'


def make_validation_operator(positive_score, negative_score, beta, iterator):
    if beta == 1.0:
        v = tf.count_nonzero(positive_score > negative_score, dtype=tf.float32)
    else:
        n_negative = tf.shape(negative_score)[0]
        raw_threshold = tf.to_float(n_negative) * beta
        threshold = tf.floor(raw_threshold)
        k = tf.to_int32(threshold + 1)

        # betaに対応する個数を選び出す
        values, _ = tf.nn.top_k(negative_score, k, sorted=False)
        # betaの範囲の端のデータ
        min_value = tf.reduce_min(values)

        v0 = tf.count_nonzero(positive_score > values, dtype=tf.float32)
        v1 = tf.count_nonzero(positive_score > min_value, dtype=tf.float32)
        # v0はbetaの外側の余計な部分までを含んでいるので、それを取り除く
        v = v0 - (1.0 - (raw_threshold - threshold)) * v1

    with tf.variable_scope('validation_metrics') as vs:
        # histogram用にpositive dataを蓄える
        tmp = tf.contrib.metrics.streaming_concat(positive_score)
        positive_op, update_positive_op = tmp

        total = tf.get_local_variable(name='total', shape=[], dtype=tf.float32,
                                      initializer=tf.zeros_initializer)
        add_total = tf.assign_add(total, v)

        local_variable = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES
        )
        reset_op = tf.variables_initializer(local_variable + [total])

    if beta == 1.0:
        pauc = total / (tf.to_float(tf.shape(negative_score)[0]) *
                        tf.to_float(tf.shape(positive_op)[0]))
    else:
        # noinspection PyUnboundLocalVariable
        pauc = total / (raw_threshold * tf.to_float(tf.shape(positive_op)[0]))

    summary_list = [
        tf.summary.scalar('validation/pAUC-exact', pauc),
        tf.summary.histogram('validation/score/positive', positive_op),
        tf.summary.histogram('validation/score/negative', negative_score)
    ]
    summary_op = tf.summary.merge(summary_list)

    update_op = tf.group(add_total, update_positive_op)
    return iterator, update_op, pauc, summary_op, reset_op
