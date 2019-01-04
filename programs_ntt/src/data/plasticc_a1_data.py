#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
観測日が揃っていないので、画像にして入力してみる
そのためのデータを生成する
"""

import numpy as np
import pandas as pd

from .cosmology import Cosmology

__author__ = 'Yasuhiro Imoto'
__date__ = '05/12/2018'


def make_image(df, x_size, x_offset, y_min, y_max, y_scale):
    """

    :param df: redshiftで正規化されたfluxのデータ
    :param x_size: 画像の横幅
    :param x_offset: 最初の観測をプロットする横軸の位置
    :param y_min: magnitudeの最小値
    :param y_max: magnitudeの最大値
    :param y_scale: magnitudeが1変化するときのピクセル数
    :return:
    """
    y_size = (y_max - y_min) * y_scale
    image = np.zeros([6, (y_max - y_min) * y_scale, x_size])

    min_mjd = np.min(df['mjd']) - x_offset
    x_array = (df['mjd'].values - min_mjd).astype(np.int32)
    y_array = np.round(
        (np.arcsinh(df['flux'] * 0.5) - y_min) * y_scale
    ).astype(np.int32)

    flag_x = np.logical_and(x_array >= 0, x_array < x_size)
    flag_y = np.logical_and(y_array >= 0, y_array < y_size)
    flag = np.logical_and(flag_x, flag_y)

    band = df.loc[flag, 'passband'].values.astype(np.int32)

    index = (band, y_array[flag], x_array[flag])
    image[index] = 1

    return image


def modify_flux(df, specz, photoz, photoz_err, is_training):
    if is_training:
        if specz is None:
            z = photoz + photoz_err * np.random.randn()
        else:
            r = np.random.rand()
            if r < 0.1:
                z = specz
            else:
                z = photoz + photoz_err * np.random.randn()
    else:
        z = specz or photoz

    z = max(z, 0.01)

    lcdm = Cosmology()
    dist_mod = lcdm.DistMod(z)
    z_norm = 0.5
    dist_mod_norm = lcdm.DistMod(z_norm)
    factor = 10.0 ** (0.4 * (dist_mod - dist_mod_norm))

    n = len(df)
    flux = df['flux'] + df['flux_err'] * np.random.randn(n)

    new_df = pd.DataFrame(flux * factor, columns=['flux'])
    new_df['mjd'] = df['mjd']
    new_df['passband'] = df['passband']

    return new_df


def make_image2(args):
    df, specz, photoz, photoz_err, image_parameters, is_training = args

    df = modify_flux(
        df=df, specz=specz, photoz=photoz,
        photoz_err=photoz_err, is_training=is_training
    )
    image = make_image(
        df=df, x_size=image_parameters['x_size'],
        x_offset=image_parameters['x_offset'],
        y_min=image_parameters['y_min'],
        y_max=image_parameters['y_max'],
        y_scale=image_parameters['y_scale']
    )
    return image
