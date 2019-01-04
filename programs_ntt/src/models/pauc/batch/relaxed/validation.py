#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf

__author__ = 'Yasuhiro Imoto'
__date__ = '28/12/2017'


def make_validation_operator(positive_score, negative_score, gamma, iterator):
    count = tf.count_nonzero(positive_score > negative_score,
                             axis=1, dtype=tf.float32)
    # positive dataについて平均
    mean_positive = count / tf.to_float(tf.shape(positive_score)[0])
    tmp = tf.pow(mean_positive, gamma)
    with tf.variable_scope('validation_metrics') as vs:
        # negative dataの方向について平均を集計
        mean_op, update_mean_op = tf.metrics.mean(tmp)

        # histogram用にnegative dataを蓄える
        tmp = tf.contrib.metrics.streaming_concat(negative_score)
        negative_op, update_negative_op = tmp

        local_variable = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES
        )
        reset_op = tf.variables_initializer(local_variable)

    summary_list = [
        tf.summary.scalar('validation/pAUC-relaxed',
                          tf.pow(mean_op, 1.0 / gamma)),
        tf.summary.histogram('validation/score/positive', positive_score),
        tf.summary.histogram('validation/score/negative', negative_op)
    ]
    summary_op = tf.summary.merge(summary_list)

    update_op = tf.group(update_mean_op, update_negative_op)
    return iterator, update_op, mean_op, summary_op, reset_op
