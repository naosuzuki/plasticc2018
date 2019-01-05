#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
GMMの基本となるクラス
共分散行列がフルランク行列が対角行列
"""

from abc import ABCMeta, abstractmethod

__author__ = 'Yasuhiro Imoto'
__date__ = '04/1/2018'


class BaseGaussian(metaclass=ABCMeta):
    """
    ガウス分布を表現するクラス
    """
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.compute_log_likelihood(inputs=inputs)

    def compute_log_likelihood(self, inputs):
        """
        対数尤度比を求める

        :param inputs:
        :return:
        """
        e = self.compute_ll_exponential(inputs=inputs)
        d = self.compute_ll_determinant()
        return e + d

    @abstractmethod
    def compute_ll_exponential(self, inputs):
        """
        expの部分の尤度を求める

        :param inputs:
        :return:
        """
        pass

    @abstractmethod
    def compute_ll_determinant(self):
        """
        正規化項の部分の対数尤度を求める

        :return:
        """
        pass
