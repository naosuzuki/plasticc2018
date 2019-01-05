#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from sklearn.mixture import BayesianGaussianMixture

__author__ = 'Yasuhiro Imoto'
__date__ = '04/1/2018'


def compute_initial_value(data, model_type, n_components):
    if model_type == 'full':
        covariance_type = 'full'
    elif model_type == 'diagonal':
        covariance_type = 'diag'
    else:
        raise NotImplementedError('unknown model type')

    bgm = BayesianGaussianMixture(n_components=n_components,
                                  covariance_type=covariance_type)
    bgm.fit(data)

    mixture_ratio = bgm.weights_
    mean_list = bgm.means_
    precision_list = bgm.precisions_cholesky_

    return mixture_ratio, mean_list, precision_list
