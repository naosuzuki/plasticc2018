#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from .dnn import get_basic_model1
from .gmm import Model as GaussianModel

__author__ = 'Yasuhiro Imoto'
__date__ = '04/1/2018'


def get_score_model(model_type, positive_data, negative_data, **kwargs):
    if model_type == 'dnn':
        if isinstance(positive_data, int):
            # GMMのpredictionはデータの代わりにサイズを渡すので、それに合わせる
            input_size = positive_data
        else:
            input_size = positive_data.shape[1]
        return get_basic_model1(input_size=input_size, **kwargs)
    elif model_type == 'gmm-full':
        return GaussianModel(
            positive_data=positive_data, negative_data=negative_data,
            model_type='full', **kwargs
        )
    elif model_type == 'gmm-diagonal':
        return GaussianModel(
            positive_data=positive_data, negative_data=negative_data,
            model_type='diagonal', **kwargs
        )
    else:
        raise NotImplementedError('unknown score model')
