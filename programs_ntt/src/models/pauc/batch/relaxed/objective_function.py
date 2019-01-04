#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from .data import get_data
from .validation import make_validation_operator
from .gradient import compute_gradient
from pauc.constraints import get_constraints

__author__ = 'Yasuhiro Imoto'
__date__ = '28/12/2017'


def make_objective_function(data, batch_size, model, gamma, lambda_constraints,
                            train):
    positive_data, negative_data, iterator = get_data(data, batch_size)

    # ブロードキャストのために横ベクトルに変換
    positive_score = tf.reshape(model(tf.constant(positive_data)), [-1])
    # 縦ベクトルのまま
    negative_score = model(negative_data)

    if not train:
        return make_validation_operator(
            positive_score=positive_score, negative_score=negative_score,
            gamma=gamma, iterator=iterator
        )

    with tf.variable_scope('train_metrics') as vs:
        # histogram用にnegative dataを蓄える
        tmp = tf.contrib.metrics.streaming_concat(negative_score)
        negative_op, update_negative_op = tmp

        local_variable = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES
        )

    # 負例についてはバッチ単位でシグモイドを計算
    d = positive_score - negative_score
    s = tf.reduce_sum(tf.sigmoid(d), axis=1)
    s = tf.reduce_sum(tf.pow(s + 1e-6, gamma))

    # 合計値を蓄える
    total = tf.get_local_variable(name='total', shape=[], dtype=tf.float32,
                                  initializer=tf.zeros_initializer)
    add_total = tf.assign_add(total, s)

    # 重みの制約
    tmp = get_constraints(lambda_constraints)
    constraints, summary_list1, summary_list2 = tmp

    # 勾配とその更新処理を計算
    apply_op, grads, add_grads = compute_gradient(s, total=total,
                                                  constraints=constraints)

    reset_op = tf.variables_initializer([total] + grads + local_variable)
    add_op = tf.group(add_total, update_negative_op, *add_grads)

    n_positive = tf.constant(data['positive'].shape[0], dtype=tf.float32)
    n_negative = tf.constant(data['negative'].shape[0], dtype=tf.float32)
    tmp = (tf.log(total) / gamma -
           tf.log(n_positive) - tf.log(n_negative) / gamma)
    loss = -tmp
    if constraints is None:
        total_loss = loss
    else:
        total_loss = loss + constraints
    score = tf.exp(tmp)

    summary_list1.extend([
        tf.summary.scalar('loss/total', total_loss),
        tf.summary.scalar('loss/main', loss),
        tf.summary.scalar('train/pAUC-relaxed', score)
    ])
    summary_op1 = tf.summary.merge(summary_list1)

    summary_list2.extend([
        tf.summary.histogram('train/score/positive', positive_score),
        tf.summary.histogram('train/score/negative', negative_op)
    ])
    summary_op2 = tf.summary.merge(summary_list2)

    return (iterator, add_op, apply_op, reset_op, summary_op1, summary_op2,
            # 学習と独立してsummary_op2を評価するために必要
            update_negative_op)
