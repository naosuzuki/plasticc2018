#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from pathlib import Path

import click
import numpy as np
import pandas as pd
import dask.dataframe as dd
import scipy.optimize
from george.kernels import Matern32Kernel
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
import george

from cosmology import Cosmology
from fit_gp_augmentation import (compute_tau, negative_log_likelihood,
                                 grad_negative_log_likelihood)

__author__ = 'Yasuhiro Imoto'
__date__ = '28/12/2018'


def select_extra_galactic(data_path, meta_path):
    df = dd.read_csv(data_path, header=0)
    meta = dd.read_csv(meta_path, header=0)

    df = df[df['object_id'] % 1000 == 0]
    photoz = meta[['object_id', 'hostgal_photoz']]
    df = df.merge(photoz, how='left', on='object_id')
    df = df[df['hostgal_photoz'] > 0]

    return df


def fit(group):
    wave_length = np.array([3670.7, 4826.9, 6223.2, 7546.0, 8690.9, 9710.3])

    time = group['mjd'] - np.min(group['mjd'])
    flux = group['flux']
    flux_err = group['flux_err']

    w = wave_length[group['passband'].values]

    # GPで補間
    tau = compute_tau(df=group)
    kernel = np.var(flux) * Matern32Kernel(tau ** 2, ndim=2)
    gp = george.GP(kernel=kernel)

    x = np.stack([time, w], axis=1)
    gp.compute(x, flux_err)

    result = scipy.optimize.minimize(
        negative_log_likelihood, gp.get_parameter_vector(),
        args=(gp, flux), jac=grad_negative_log_likelihood,
        method='L-BFGS-B'
    )
    gp.set_parameter_vector(result.x)

    # GPによる光度曲線
    t = np.arange(-10, int(np.ceil(np.max(time))) + 10)
    n = len(t)
    df_list = []
    for band, w in enumerate(wave_length):
        x = np.empty((n, 2))
        x[:, 0] = t
        x[:, 1] = w
        p, p_var = gp.predict(flux, x, return_var=True)

        tmp = pd.DataFrame({
            'mjd': t, 'passband': [band] * n, 'flux': p,
            'flux_err': np.sqrt(p_var), 'detected': [0] * n,
            'interpolated': [True] * n
        })
        df_list.append(tmp)

    new_df = pd.concat(df_list, axis=0, ignore_index=True)
    new_df['object_id'] = group['object_id'].values[0]
    new_df['sub_id'] = 0

    # 並び替え
    new_df = new_df[['object_id', 'sub_id', 'mjd', 'passband',
                     'flux', 'flux_err', 'detected', 'interpolated']]
    new_df.sort_values(['mjd', 'passband'], inplace=True)

    return new_df


@click.command()
@click.option('--data-path', type=click.Path(exists=True),
              default='../../data/raw/test_set.csv')
@click.option('--meta-path', type=click.Path(exists=True),
              default='../../data/raw/test_set_metadata.csv')
@click.option('--output-path', type=click.Path(),
              default='../../data/processed/test_set.h5')
def cmd(data_path, meta_path, output_path):
    df = select_extra_galactic(data_path=data_path, meta_path=meta_path)

    data = df.groupby('object_id').apply(fit, meta=object).compute()
    with pd.HDFStore(output_path) as store:
        for object_id, value in tqdm(data.iteritems(), total=len(data)):
            store['id{}'.format(object_id)] = value


def main():
    cmd()


if __name__ == '__main__':
    main()
