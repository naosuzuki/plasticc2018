#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import re

import tensorflow as tf

from score.dnn.model import basic_score1
from .objective_function import compute_relaxed_pauc, compute_exact_pauc

__author__ = 'Yasuhiro Imoto'
__date__ = '13/12/2017'


def setup_relaxed_model(input_data, hidden_size, lambda_constraints,
                        reuse=False, gamma=0.1, training=True, batch_size=100):
    """
    緩和されたpAUCのモデル

    :param input_data:
    :param hidden_size:
    :param lambda_constraints:
    :param reuse:
    :param gamma:
    :param training:
    :param batch_size:
    :return:
    """
    return setup_model_helper(
        input_data, hidden_size, lambda_constraints, mode='relaxed',
        reuse=reuse, training=training, batch_size=batch_size, gamma=gamma
    )


def setup_exact_model(input_data, hidden_size, lambda_constraints,
                      reuse=False, beta=0.1, training=True, batch_size=100):
    return setup_model_helper(
        input_data, hidden_size, lambda_constraints, mode='exact',
        reuse=reuse, training=training, batch_size=batch_size, beta=beta
    )


def setup_model_helper(input_data, hidden_size, lambda_constraints, mode,
                       reuse=False, training=True, batch_size=100, **kwargs):
    # 変換済みの値
    # そのままネットワークに入力する
    input_positive, input_negative = input_data.get_data()
    positive_score = basic_score1(input_data.feature_size, input_positive,
                                  hidden_size, reuse=reuse)
    negative_score = basic_score1(input_data.feature_size, input_negative,
                                  hidden_size, reuse=True)
    if mode == 'relaxed':
        gamma = kwargs['gamma']

        direct = True
        relaxed_pauc = compute_relaxed_pauc(
            positive_score, negative_score, gamma=gamma, training=training,
            batch_size=batch_size, direct=direct
        )

        if direct:
            score = tf.exp(relaxed_pauc)
            loss = -relaxed_pauc
        else:
            if training:
                score = tf.exp(-relaxed_pauc)
                loss = relaxed_pauc
            else:
                score = relaxed_pauc
                loss = -relaxed_pauc
    else:
        beta = kwargs['beta']
        exact_pauc = compute_exact_pauc(positive_score, negative_score,
                                        beta=beta, training=training,
                                        batch_size=batch_size)
        score = exact_pauc
        loss = -exact_pauc

    # basic_scoreというスコープの中で、ネットワークを構成した
    # 全結合の重みはmy_dense1/kernelのような形になっている
    # Denseの名前を指定しなかったときは、dense, dense_1のようになるので、
    # kernelの名前は変わらない
    # 先にtrainのネットワークを作っているので、optimizerの変数と区別する必要がある
    kernels = [v for v in tf.global_variables('basic_score')
               if 'kernel:0' in v.name]
    # 重みの名前を取り出す
    r = re.compile(r'/(.+)/kernel')
    names = []
    for v in kernels:
        m = r.search(v.name)
        name = m.group(1)
        names.append(name)

    if training:
        # L1-norm制約
        constraints = [tf.reduce_mean(tf.abs(v)) for v in kernels]
        c = [w * constraint
             for w, constraint in zip(lambda_constraints, constraints)]

        total_loss = loss + tf.add_n(c)

        optimizer = tf.train.AdamOptimizer()
        opt = optimizer.minimize(total_loss)

        # 毎回評価するもの
        summary_list1 = [
            tf.summary.scalar('loss/total', total_loss),
            tf.summary.scalar('loss/main', loss),
            tf.summary.scalar('train/pAUC-{}'.format(mode), score)
        ] + [
            tf.summary.scalar('loss/constraint/{}'.format(name), constraint)
            for name, constraint in zip(names, constraints)
        ]
        summary1 = tf.summary.merge(summary_list1)

        # 偶に評価するもの
        summary_list2 = [
            tf.summary.histogram('train/score/positive', positive_score),
            tf.summary.histogram('train/score/negative', negative_score)
        ] + [
            tf.summary.histogram('train/weight/{}'.format(name), v)
            for name, v in zip(names, kernels)
        ]
        summary2 = tf.summary.merge(summary_list2)
    else:
        # 何もしないダミー
        opt = tf.no_op()

        # 毎回評価するもの
        summary1 = tf.no_op()

        # 偶に評価するもの
        summary_list2 = [
            tf.summary.scalar('validation/pAUC-{}'.format(mode), score),
            tf.summary.histogram('validation/score/positive', positive_score),
            tf.summary.histogram('validation/score/negative', negative_score)
        ] + [
            tf.summary.histogram('validation/weight/{}'.format(name), v)
            for name, v in zip(names, kernels)
        ]
        summary2 = tf.summary.merge(summary_list2)

    return opt, summary1, summary2
