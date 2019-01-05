#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from pathlib import Path
from itertools import product

import click
import numpy as np
import pandas as pd
import xarray as xr
import dask.dataframe as dd
from astropy.io import fits
from tqdm import tqdm

from cosmology import Cosmology

__author__ = 'Yasuhiro Imoto'
__date__ = '14/12/2018'


def convert(test, test_meta):
    lcdm = Cosmology()
    z_norm = 0.5
    dist_mod_norm = lcdm.DistMod(z_norm)

    features = []

    current_id = test[0]['object_id']
    data_list = []

    aggregator = {'magnitude': ['max', 'mean', 'std', 'median', 'skew']}
    for object_id, band, flux in zip(tqdm(test['object_id']),
                                     test['passband'], test['flux']):
        #     print(object_id, band, flux)
        if object_id != current_id:
            factor = compute_factor(
                lcdm=lcdm, dist_mod_norm=dist_mod_norm, test_meta=test_meta,
                current_id=current_id
            )

            df = pd.DataFrame(data_list, columns=['passband', 'flux'])
            df['magnitude'] = np.arcsinh(df['flux'].values * factor * 0.5)

            whole_feature = df.agg(aggregator)
            band_feature = df.groupby('passband').agg(aggregator)

            feature = convert_feature(
                whole_feature=whole_feature, band_feature=band_feature,
                object_id=current_id
            )
            features.append(feature)

            current_id = object_id
            data_list = []
        data_list.append((band, flux))

    factor = compute_factor(
        lcdm=lcdm, dist_mod_norm=dist_mod_norm, test_meta=test_meta,
        current_id=current_id
    )

    df = pd.DataFrame(data_list, columns=['passband', 'flux'])
    df['magnitude'] = np.arcsinh(df['flux'] * factor * 0.5)

    whole_feature = df.agg(aggregator)
    band_feature = df.groupby('passband').agg(aggregator)

    feature = convert_feature(
        whole_feature=whole_feature, band_feature=band_feature,
        object_id=current_id
    )
    features.append(feature)

    df = pd.concat(features, axis=0)
    df.index.name = 'object_id'

    return df


def compute_factor(lcdm, dist_mod_norm, test_meta, current_id):
    tmp = test_meta[test_meta['object_id'] == current_id]
    specz = tmp['hostgal_specz']
    photoz = tmp['hostgal_photoz']
    if np.isnan(specz):
        z = photoz
    else:
        z = specz
    dist_mod = lcdm.DistMod(z)
    factor = 10.0 ** (0.4 * (dist_mod - dist_mod_norm))

    return factor


def format_whole_feature(whole_feature, object_id):
    whole_feature = whole_feature.T
    new_columns = [
        'magnitude_{}'.format(name) for name in whole_feature.columns
    ]

    whole_feature.index = [object_id]
    whole_feature.columns = new_columns

    return whole_feature


def format_band_feature(band_feature, object_id):
    # noinspection PyUnresolvedReferences
    band_feature.columns = [
        'magnitude_{}'.format(band_feature.columns.levels[1][i])
        for i in band_feature.columns.labels[1]
    ]

    new_feature = {}
    for name, band in product(band_feature.columns, band_feature.index):
        new_feature[name + str(band)] = band_feature.loc[band, name]
    new_feature = pd.DataFrame(new_feature, index=[object_id])

    return new_feature


def convert_feature(whole_feature, band_feature, object_id):
    whole_feature = format_whole_feature(
        whole_feature=whole_feature, object_id=object_id
    )
    band_feature = format_band_feature(
        band_feature=band_feature, object_id=object_id
    )
    df = pd.concat([whole_feature, band_feature], axis=1)
    return df


@click.command()
@click.option('--index', type=int)
@click.option('--n-splits', type=int, default=50)
def cmd(index, n_splits):
    output_dir = Path('/home/imoto/crest_auto/data/interim/test_features_v2')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    print('loading...')
    table = fits.open(
        '/home/imoto/crest_auto/data/interim/test_set_v2.fits', memmap=True
    )
    test = table[1].data

    meta_table = fits.open(
        '/home/imoto/crest_auto/data/interim/test_set_metadata.fits',
        memmap=True
    )
    test_meta = meta_table[1].data

    test_meta = test_meta[test_meta['hostgal_photoz'] > 0]
    id_set = [i for i in test_meta['object_id']]

    n = (len(id_set) + n_splits - 1) // n_splits

    print('selecting...')
    flag = np.empty(len(test), dtype=np.bool)
    id_set = set(id_set[index * n:(index + 1) * n])
    for i, object_id in enumerate(test['object_id']):
        flag[i] = object_id in id_set
    test = test[flag]

    test = test[test['flux'] > 0]

    df = convert(test=test, test_meta=test_meta)
    df.to_pickle(output_dir / 'df{0:02d}.pickle'.format(index))

    table.close()
    meta_table.close()


def main():
    cmd()


if __name__ == '__main__':
    main()
