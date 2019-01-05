#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
対数尤度比とかを計算するモデル
"""

import numpy as np
import tensorflow as tf

from .full_covariance import FullCovariance
from .diagonal_covariance import DiagonalCovariance
from .initial_value import compute_initial_value

__author__ = 'Yasuhiro Imoto'
__date__ = '04/1/2018'


class Model(object):
    def __init__(self, positive_data, negative_data,
                 positive_components, negative_components, model_type):
        self.positive_gmm = GMM(
            data=positive_data, n_components=positive_components,
            model_type=model_type, scope_name='GMM-positive'
        )
        self.negative_gmm = GMM(
            data=negative_data, n_components=negative_components,
            model_type=model_type, scope_name='GMM-negative'
        )

    def __call__(self, inputs):
        """
        スコア(対数尤度比)を返す

        :param inputs:
        :return:
        """
        return self.compute_log_likelihood_ratio(inputs=inputs)

    def compute_log_likelihood_ratio(self, inputs):
        p = self.positive_gmm(inputs=inputs)
        n = self.negative_gmm(inputs=inputs)
        return p - n


class GMM(object):
    def __init__(self, data, n_components, model_type, scope_name):
        if isinstance(data, int):
            # 予測などで、初期値が不要な場合
            data_size = data

            mean_list = [0 for _ in range(n_components)]
            precision_list = mean_list
            # 値は何でもいい
            ratio = np.zeros(n_components, dtype=np.float32)
        else:
            ratio, mean_list, precision_list = compute_initial_value(
                data=data, model_type=model_type, n_components=n_components
            )
            data_size = data.shape[1]

        with tf.variable_scope(scope_name):
            name = 'Gaussian-{}{{}}'.format(model_type)
            if model_type == 'full':
                model = FullCovariance
            elif model_type == 'diagonal':
                model = DiagonalCovariance
            else:
                raise NotImplementedError('unknown GMM model')

            # 各ガウス分布
            self.components = [
                model(data_size=data_size, scope_name=name.format(i),
                      mean=mean, precision=precision)
                for i, (mean, precision) in enumerate(zip(mean_list,
                                                          precision_list))
            ]
            # 混合比
            self.log_ratio = tf.get_variable(
                'pi', shape=[n_components], dtype=tf.float32,
                initializer=tf.constant_initializer(np.log(ratio + 1e-6))
            )

    def __call__(self, inputs):
        return self.compute_mixture_log_likelihood(inputs=inputs)

    def compute_mixture_log_likelihood(self, inputs):
        # 指数を取ったときの合計が1になるように調整する
        normalized = self.log_ratio - tf.reduce_logsumexp(self.log_ratio)

        # 縦にバッチ、横に各ガウス分布の対数尤度
        ll = tf.concat([c(inputs) for c in self.components], axis=1)

        tmp = normalized + ll
        # DNNの出力と同じで2次元にする
        mixture_ll = tf.reduce_logsumexp(tmp, axis=1, keep_dims=True)

        return mixture_ll
