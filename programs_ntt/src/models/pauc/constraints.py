#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import re

import tensorflow as tf

__author__ = 'Yasuhiro Imoto'
__date__ = '28/12/2017'


def get_constraints(lambda_constraints):
    if lambda_constraints is None:
        # モデルがDNNでない場合は、制約条件は不要
        cost = None
        summary_list1, summary_list2 = [], []
        return cost, summary_list1, summary_list2

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

    # L1-norm制約
    constraints = [tf.reduce_mean(tf.abs(v)) for v in kernels]
    c = [w * constraint
         for w, constraint in zip(lambda_constraints, constraints)]
    cost = tf.add_n(c)

    # 毎回評価する
    summary_list1 = [
        tf.summary.scalar('loss/constraint/{}'.format(name), constraint)
        for name, constraint in zip(names, constraints)
    ]

    # 偶に評価する
    summary_list2 = [
        tf.summary.histogram('train/weight/{}'.format(name), v)
        for name, v in zip(names, kernels)
    ]

    return cost, summary_list1, summary_list2
