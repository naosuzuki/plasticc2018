#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import tensorflow as tf

from sn_model.dataset import compute_magnitude, make_noisy_magnitude
__author__ = 'Yasuhiro Imoto'
__date__ = '12/12/2017'


class Loader(object):
    """
    全てのデータを一度に読み出す
    """
    def __init__(self, data_path, band_data, mean, std, use_noise,
                 blackout_rate, outlier_rate, use_redshift,
                 positive_label, negative_label, method='modified'):
        self.use_noise = use_noise
        self.blackout_rate = blackout_rate
        self.outlier_rate = outlier_rate

        self.use_redshift = use_redshift

        self.method = method

        # どのバンドを使うかの情報のみで十分
        self.band_list = list(band_data.keys())
        # 順序を固定
        self.band_list.sort()

        tmp = self.load_data(data_path, mean, std,
                             positive_label, negative_label)
        self.positive_data, self.positive_flux, self.positive_flux_err = tmp[0]
        self.negative_data, self.negative_flux, self.negative_flux_err = tmp[1]
        self.mean, self.std = tmp[2]

        # use_noiseがFalseの場合は、毎回同じ値を返すので、値を記録する
        self.cache = None

    @property
    def feature_size(self):
        """
        特徴量の次元数

        :return:
        """
        if self.use_redshift:
            return self.positive_flux.shape[1] + 1
        else:
            return self.positive_flux.shape[1]

    def load_data(self, path, mean, std, positive_label, negative_label):
        """
        データの読み込み
        netcdf4形式を想定

        :param path:
        :param mean:
        :param std:
        :param positive_label:
        :param negative_label:
        :return:
        """
        data = xr.open_dataset(path)

        # positiveデータだけを選択
        positive_data = self.select_label(data, positive_label)
        # 必要なバンドを選択、並び替え
        positive_flux, positive_flux_err, mu, sigma = self.select_band(
            positive_data, mean, std
        )
        # negativeデータだけを選択
        negative_data = self.select_label(data, negative_label)
        # 必要なバンドを選択、並び替え
        # 平均と標準偏差はpositiveの時と同じなので省略
        negative_flux, negative_flux_err, _, _ = self.select_band(
            negative_data, mean, std
        )

        return ((positive_data, positive_flux, positive_flux_err),
                (negative_data, negative_flux, negative_flux_err),
                (mu, sigma))

    def select_band(self, ds, mean, std):
        flux, flux_err = [], []
        mu, sigma = [], []

        # 必要なバンドの選択と並び替えを行う
        for band in self.band_list:
            tmp = self.get_band(ds, band)
            flux.append(tmp['{}-flux'.format(band)])
            flux_err.append(tmp['{}-flux_err'.format(band)])

            mu.append(mean[band])
            sigma.append(std[band])

        flux = np.hstack(flux)
        flux_err = np.hstack(flux_err)
        mu = np.hstack(mu)
        sigma = np.hstack(sigma)

        return flux, flux_err, mu, sigma

    @staticmethod
    def select_label(ds, label_list):
        """
        label_listで指定されたラベルのデータを抜き出す

        :param ds:
        :param label_list:
        :return:
        """
        flag = ds.label == label_list[0]
        for label in label_list[1:]:
            flag = np.logical_or(flag, ds.label == label)
        tmp = ds.where(flag, drop=True)
        return tmp

    @staticmethod
    def get_band(ds, band):
        tmp = ds.where(ds.band == band.encode(), drop=True)
        flux = np.nan_to_num(tmp.flux.values)
        flux_err = np.nan_to_num(tmp.flux_err.values)

        flux_err[flux_err > 1] = 1

        return {'{}-flux'.format(band): flux,
                '{}-flux_err'.format(band): flux_err}

    def compute_magnitude(self, flux, flux_err):
        if self.use_noise:
            magnitude = make_noisy_magnitude(
                flux, flux_err, self.blackout_rate, self.outlier_rate,
                method=self.method
            )
        else:
            magnitude = compute_magnitude(flux, method=self.method)
        magnitude = (magnitude - self.mean) / self.std
        return magnitude

    def get_data(self):
        """
        特徴量のデータ(正例と負例の組)を取り出す
        fluxはmagnitudeに変換する

        :return:
        """
        if not self.use_noise and self.cache is not None:
            # 計算済みの値を返す
            return self.cache

        positive_feature = self.convert_data(
            self.positive_flux, self.positive_flux_err, self.positive_data
        )
        negative_feature = self.convert_data(
            self.negative_flux, self.negative_flux_err, self.negative_data
        )
        data = positive_feature, negative_feature

        if not self.use_noise:
            self.cache = data

        return data

    def convert_data(self, flux, flux_err, ds):
        """
        fluxとflux_errからmagnitudeを求める(設定に応じてノイズを加える)
        また、設定によってはredshiftを特徴量に加える

        get_dataから呼び出されることを想定

        :param flux:
        :param flux_err:
        :param ds:
        :return:
        """
        data = self.compute_magnitude(flux, flux_err)
        if self.use_redshift:
            # concatで繋げられるように2次元配列に変形
            redshift = ds.redshift.values.reshape([-1, 1])
            data = tf.concat([data, redshift], axis=1)

        return data

