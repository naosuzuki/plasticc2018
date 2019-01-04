#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import tensorflow as tf

__author__ = 'Yasuhiro Imoto'
__date__ = '28/12/2017'


def compute_gradient(s, constraints):
    optimizer = tf.train.AdamOptimizer()
    # 最小化するので符号を反転
    grads_and_vars = optimizer.compute_gradients(-s)

    # 勾配を保存する変数
    grads = []
    add_grads = []
    for i, (g, _) in enumerate(grads_and_vars):
        shape = g.get_shape().as_list()
        v = tf.get_local_variable(name='my_gradients{}'.format(i),
                                  shape=shape, dtype=tf.float32,
                                  initializer=tf.zeros_initializer)
        grads.append(v)
        add_grads.append(tf.assign_add(v, g))

    if constraints is None:
        total_grads_and_vars = [(g, v) for g, (_, v) in
                                zip(grads, grads_and_vars)]
    else:
        # 制約の勾配
        # 元の変数は必要ないので、変数を上書きする
        grads_and_vars = optimizer.compute_gradients(constraints)

        total_grads_and_vars = [
            (g + cg if cg is not None else g, v)
            for g, (cg, v) in zip(grads, grads_and_vars)
        ]

    global_step = tf.train.create_global_step()
    # 勾配に基づいて更新
    apply_op = optimizer.apply_gradients(total_grads_and_vars,
                                         global_step=global_step)

    return apply_op, grads, add_grads
