#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from .data import get_data
from .validation import make_validation_operator
from .gradient import compute_gradient
from pauc.constraints import get_constraints

__author__ = 'Yasuhiro Imoto'
__date__ = '28/12/2017'


def make_objective_function(data, batch_size, model, beta, lambda_constraints,
                            train):
    positive_data, negative_data, iterator = get_data(data, batch_size)

    # 縦ベクトルのまま
    positive_score = model(positive_data)
    # ブロードキャストのために横ベクトルに変換
    negative_score = tf.reshape(model(tf.constant(negative_data)), [-1])

    if not train:
        return make_validation_operator(
            positive_score=positive_score, negative_score=negative_score,
            beta=beta, iterator=iterator
        )

    with tf.variable_scope('train_metrics') as vs:
        # histogram用にpositive dataを蓄える
        tmp = tf.contrib.metrics.streaming_concat(positive_score)
        positive_op, update_positive_op = tmp

        local_variable = tf.contrib.framework.get_variables(
            vs, collection=tf.GraphKeys.LOCAL_VARIABLES
        )

    if beta == 1.0:
        v = tf.reduce_sum(tf.sigmoid(positive_score - negative_score))
    else:
        n_negative = tf.shape(negative_score)[0]
        raw_threshold = tf.to_float(n_negative) * beta
        threshold = tf.floor(raw_threshold)
        k = tf.to_int32(threshold + 1)

        # betaに対応する個数を選び出す
        values, _ = tf.nn.top_k(negative_score, k, sorted=False)
        # betaの範囲の端のデータ
        min_value = tf.reduce_min(values)

        v0 = tf.reduce_sum(tf.sigmoid(positive_score - values))
        v1 = tf.reduce_sum(tf.sigmoid(positive_score - min_value))
        # v0はbetaの外側の余計な部分までを含んでいるので、それを取り除く
        v = v0 - (1.0 - (raw_threshold - threshold)) * v1

    # 合計値を蓄える
    total = tf.get_local_variable(name='train_total', shape=[],
                                  dtype=tf.float32,
                                  initializer=tf.zeros_initializer)
    add_total = tf.assign_add(total, v)

    # 重みの制約
    tmp = get_constraints(lambda_constraints)
    constraints, summary_list1, summary_list2 = tmp

    # 勾配とその更新処理を計算
    apply_op, grads, add_grads = compute_gradient(v, constraints=constraints)

    reset_op = tf.variables_initializer([total] + grads + local_variable)
    add_op = tf.group(add_total, update_positive_op, *add_grads)

    n_positive = tf.constant(data['positive'].shape[0], dtype=tf.float32)
    n_negative = tf.constant(data['negative'].shape[0], dtype=tf.float32)
    pauc = total / (beta * n_positive * n_negative)
    loss = -pauc
    if constraints is None:
        total_loss = loss
    else:
        total_loss = loss + constraints

    summary_list1.extend([
        tf.summary.scalar('loss/total', total_loss),
        tf.summary.scalar('loss/main', loss),
        tf.summary.scalar('train/pAUC-exact', pauc)
    ])
    summary_op1 = tf.summary.merge(summary_list1)

    summary_list2.extend([
        tf.summary.histogram('train/score/positive', positive_op),
        tf.summary.histogram('train/score/negative', negative_score)
    ])
    summary_op2 = tf.summary.merge(summary_list2)

    return (iterator, add_op, apply_op, reset_op, summary_op1, summary_op2,
            # 学習と独立してsummary_op2を評価するために必要
            update_positive_op)
