#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from pathlib import Path

import click
import george
import numpy as np
import pandas as pd
import scipy.optimize
from george.kernels import Matern32Kernel
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

from cosmology import Cosmology

__author__ = 'Yasuhiro Imoto'
__date__ = '25/12/2018'


def select_extra_galactic(data_dir, output_dir):
    meta = pd.read_csv(data_dir / 'training_set_metadata.csv', header=0)
    target_map = {
        object_id: target
        for object_id, target in zip(
            meta['object_id'].values, meta['target'].values
        )
    }
    # extra galacticのクラスラベルの一覧
    extra_meta = meta[meta['hostgal_photoz'] > 0]
    extra_class_list = extra_meta['target'].value_counts(ascending=True).index

    # クラスごとに分割
    for target in tqdm(extra_class_list):
        df = pd.read_csv(
            data_dir / 'training_set.csv', header=0, chunksize=1000
        )

        df_list = []
        for tmp in df:
            flag = np.empty(len(tmp), dtype=np.bool)
            for i, object_id in enumerate(tmp['object_id'].values):
                flag[i] = target_map[object_id] == target
            df_list.append(tmp[flag])

        df = pd.concat(df_list, axis=0, ignore_index=True)

        path = output_dir / 'data{}.pickle'.format(target)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        df.to_pickle(path)


def augment_extra_galactic(df_path, meta_path, output_path):
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    df = pd.read_pickle(df_path)

    index_set = set(df['object_id'].values)
    meta = pd.read_csv(meta_path, header=0)
    flag = np.zeros(len(meta), dtype=np.bool)
    for i, object_id in enumerate(meta['object_id'].values):
        flag[i] = object_id in index_set
    meta = meta[flag]

    wave_length = np.array([3670.7, 4826.9, 6223.2, 7546.0, 8690.9, 9710.3])

    size = int(np.round(10000 / len(index_set)))
    redshift = np.empty((len(index_set), size))
    id_list = []
    with pd.HDFStore(str(output_path)) as store:
        for j, (object_id, group) in enumerate(tqdm(df.groupby('object_id'))):
            tmp_meta = meta[meta['object_id'] == object_id]
            specz = tmp_meta['hostgal_specz'].values[0]
            if np.isnan(specz):
                z = tmp_meta['hostgal_photoz'].values[0]
                z_err = tmp_meta['hostgal_photoz_err'].values[0]
            else:
                z = specz
                z_err = None

            w = wave_length[group['passband'].values]

            for i in range(size):
                new_df, new_z = interpolate_data(
                    group=group, object_id=object_id, sub_id=i, z=z,
                    z_err=z_err, wave_length=wave_length, w=w
                )

                store['id{}/sub{}'.format(object_id, i)] = new_df

                redshift[j, i] = new_z
            id_list.append(object_id)
        table = pd.DataFrame(redshift, index=id_list, columns=range(size))
        store['redshift'] = table


def interpolate_data(group, object_id, sub_id, z, z_err, wave_length, w):
    # redshiftの値を設定
    old_z = z
    if z_err is not None:
        old_z = z_err * np.random.randn()

    new_z = old_z * (1 + np.random.randn() * 0.1)

    scale = (1 + new_z) / (1 + old_z)

    time = (group['mjd'] - np.min(group['mjd'])) * scale
    flux, flux_err = convert_flux(
        flux=group['flux'], flux_err=group['flux_err'],
        old_z=old_z, new_z=new_z
    )

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
    # オリジナルの観測データを追加
    df_base = pd.DataFrame(
        {'mjd': time, 'passband': group['passband'], 'flux': flux,
         'flux_err': flux_err, 'detected': group['detected'],
         'interpolated': [False] * len(time)}
    )
    df_list.append(df_base)

    new_df = pd.concat(df_list, axis=0, ignore_index=True)
    new_df['object_id'] = object_id
    new_df['sub_id'] = sub_id

    # 並び替え
    new_df = new_df[['object_id', 'sub_id', 'mjd', 'passband',
                     'flux', 'flux_err', 'detected', 'interpolated']]
    new_df.sort_values(['mjd', 'passband'], inplace=True)

    return new_df, new_z


def convert_flux(flux, flux_err, old_z, new_z):
    lcdm = Cosmology()

    dist_mod_new = lcdm.DistMod(new_z)
    dist_mod_old = lcdm.DistMod(old_z)
    factor = 10.0 ** (0.4 * (dist_mod_new - dist_mod_old))

    return flux * factor, flux_err * factor


def compute_autocorrelation(x):
    """Compute the autocorrelation function of a time series."""
    n = len(x)
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    return acf / acf[0]


def estimate_tau(t, y):
    """Estimate the correlation length of a time series."""
    dt = np.min(np.diff(t))
    tt = np.arange(t.min(), t.max(), dt)
    yy = np.interp(tt, t, y, 1)
    f = compute_autocorrelation(yy)
    fs = gaussian_filter(f, 50)
    w = dt * np.arange(len(f))
    m = np.arange(1, len(fs) - 1)[(fs[1:-1] > fs[2:]) & (fs[1:-1] > fs[:-2])]
    if len(m):
        return w[m[np.argmax(fs[m])]]
    return w[-1]


def compute_tau(df):
    tmp = []
    for _, group in df.groupby('passband'):
        if len(group) < 3:
            continue
        t = group['mjd']
        y = group['flux']
        tmp.append(estimate_tau(t=t, y=y))
    tau = np.mean(tmp) * 0.25

    return tau


def negative_log_likelihood(p, gp, y):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25


def grad_negative_log_likelihood(p, gp, y):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)


@click.command()
@click.option('--data-dir', type=click.Path(exists=True),
              default='../../data/raw')
@click.option('--output-dir', type=click.Path())
@click.option('--target', type=click.Choice([
    '15', '42', '52', '62', '64', '67', '88', '90', '95'
]))
def cmd(data_dir, output_dir, target):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # select_extra_galactic(data_dir=data_dir, output_dir=output_dir)

    if target is None:
        for target in (15, 42, 52, 62, 64, 67, 88, 90, 95):
            augment_extra_galactic(
                df_path=output_dir / 'data{}.pickle'.format(target),
                meta_path=data_dir / 'training_set_metadata.csv',
                output_path=output_dir / 'data{}_00.h5'.format(target)
            )
    else:
        target = int(target)
        augment_extra_galactic(
            df_path=output_dir / 'data{}.pickle'.format(target),
            meta_path=data_dir / 'training_set_metadata.csv',
            output_path=output_dir / 'data{}_00.h5'.format(target)
        )


def main():
    cmd()


if __name__ == '__main__':
    main()
