#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import warnings

import click
import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed

from cosmology import Cosmology

__author__ = 'Yasuhiro Imoto'
__date__ = '18/12/2018'


def compute_factor(specz, photoz, photoz_err, object_id, is_training):
    lcdm = Cosmology()

    if is_training:
        # ときどきphotozの方を使う
        if specz > 0 and np.random.rand() < 0.1:
            z = specz
        else:
            z = photoz + photoz_err * np.random.randn()
    else:
        if specz > 0:
            z = specz
        else:
            z = photoz
    z = max(z, 0.01)

    dist_mod = lcdm.DistMod(z)
    z_norm = 0.5
    dist_mod_norm = lcdm.DistMod(z_norm)
    factor = 10.0 ** (0.4 * (dist_mod - dist_mod_norm))

    return object_id, factor


def normalize_flux(df, meta, n_jobs, is_training):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        factors = Parallel(n_jobs=n_jobs)(
            delayed(compute_factor)(specz, photoz, photoz_err, object_id,
                                    is_training)
            for specz, photoz, photoz_err, object_id in zip(
                meta['hostgal_specz'], meta['hostgal_photoz'],
                meta['hostgal_photoz_err'], meta['object_id']
            )
        )
    factor_array = np.empty(len(df))
    for object_id, factor in factors:
        flag = df['object_id'] == object_id
        factor_array[flag] = factor
    df['flux'] *= factor_array
    df['flux_err'] *= factor_array

    return df, meta


def make_feature_whole(df):
    """
    バンドの違いを気にせず全体を処理
    :param df:
    :return:
    """
    aggregators = {'magnitude': ['max', 'mean', 'median', 'std', 'skew']}

    agg_df = df.groupby('object_id').agg(aggregators)
    new_columns = [
        k + '_' + agg for k in aggregators.keys() for agg in aggregators[k]
    ]
    agg_df.columns = new_columns

    return agg_df


def make_feature_band(df):
    """
    バンドごとに処理
    :param df:
    :return:
    """
    aggregators = {'magnitude': ['max', 'mean', 'median', 'std', 'skew']}

    agg_df = df.groupby(['object_id', 'passband']).agg(aggregators)
    new_columns = [
        k + '_' + agg for k in aggregators.keys() for agg in aggregators[k]
    ]
    agg_df.columns = new_columns

    # バンドについて縦に並んでいるのを横に並べ替え
    agg_df = agg_df.unstack()

    # MultiIndexになっているので、フラットなものに変更
    c = agg_df.columns
    # noinspection PyUnresolvedReferences
    new_columns = [
        '{}_{}'.format(c.levels[0][i], c.levels[1][j])
        for i, j in zip(*c.labels)
    ]
    agg_df.columns = new_columns

    return agg_df


def make_feature(df, meta, n_jobs, is_training):
    # redshiftが0.5の距離に揃える
    df, meta = normalize_flux(
        df=df, meta=meta, n_jobs=n_jobs, is_training=is_training
    )

    df['magnitude'] = np.arcsinh(df['flux'] * 0.5)

    agg_df = make_feature_whole(df=df)
    agg_df_band = make_feature_band(df=df)

    full_df = agg_df.reset_index().merge(
        right=agg_df_band.reset_index(),
        how='outer',
        on='object_id'
    )

    mean = full_df.mean(axis=0)
    full_df.fillna(mean, inplace=True)

    # カラムの順序が一定ではないようなので、sort
    full_df.sort_index(axis=1, inplace=True)

    meta = meta.set_index('object_id')
    full_df = full_df.set_index('object_id')
    full_df['target'] = meta['target']

    return full_df


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--dataset-path', type=click.Path(exists=True),
              default='../../data/raw/training_set.csv')
@click.option('--dataset_meta-path', type=click.Path(exists=True),
              default='../../data/raw/training_set_metadata.csv')
@click.option('--output-path', type=click.Path(),
              default='../../data/processed/training_set_static_feature.csv')
def convert_sn(dataset_path, dataset_meta_path, output_path):
    classes = (95, 52, 62, 42, 90)

    df = pd.read_csv(dataset_path, header=0)
    meta = pd.read_csv(dataset_meta_path, header=0)

    flag = np.zeros(len(meta), dtype=np.bool)
    for c in classes:
        flag = np.logical_or(flag, meta['target'] == c)
    meta = meta[flag]

    flag = np.zeros(len(df), dtype=np.bool)
    for object_id in meta['object_id']:
        flag = np.logical_or(flag, df['object_id'] == object_id)
    df = df[flag]
    df = df[df['flux'] > 0]

    df, meta = normalize_flux(
        df=df, meta=meta, n_jobs=4, is_training=False
    )

    feature = make_feature(df=df, meta=meta, n_jobs=4, is_training=False)
    feature.to_csv(output_path)


def main():
    cmd()


if __name__ == '__main__':
    main()
