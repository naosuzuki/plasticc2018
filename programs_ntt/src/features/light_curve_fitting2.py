#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import dask.dataframe as dd
import sncosmo
from numba import jit
from astropy.table import Table
from tqdm import tqdm

__author__ = 'Yasuhiro Imoto'
__date__ = '25/12/2018'


def fit(data, source):
    band_name = 'ugrizy'

    name_list = [
        'lsst{}'.format(band_name[band])
        for band in data['band']
    ]
    data['band'] = name_list
    del data['object_id']

    model = sncosmo.Model(source=source)
    parameter_names = model.param_names

    n = len(parameter_names)
    features = np.empty((n * 2 + 1,))

    table = Table.from_pandas(data)

    model = sncosmo.Model(source=source)
    try:
        result, fitted_model = sncosmo.fit_lc(
            data=table, model=model, vparam_names=parameter_names,
            bounds={'z': [0.1, 10.0]}
        )

        features[:n] = result.parameters
        for j, name in enumerate(parameter_names):
            features[n + j] = result.errors[name]
        features[-1] = result.chisq
    except (RuntimeError, sncosmo.fitting.DataQualityError) as e:
        print(data)
        features[:] = -1
        print(e)

    index = (
            parameter_names +
            ['{}_err'.format(name) for name in parameter_names] +
            ['chi_square']
    )
    features = pd.Series(features, index=index)
    return features


@click.command()
@click.option('--data-dir', type=click.Path(exists=True))
@click.option('--source')
@click.option('--index', type=int)
@click.option('--n-split', type=int)
@click.option('--is-training', is_flag=True)
def cmd(data_dir, source, index, n_split, is_training):
    data_dir = Path(data_dir)
    if is_training:
        dataset_path = data_dir / 'training_set.csv'
        meta_path = data_dir / 'training_metadata'
    else:
        dataset_path = data_dir / 'test_set.csv'

    df = dd.read_csv(dataset_path, header=0)
    df_selected = df[df['object_id'] % n_split == index]
    df_selected = df_selected[['object_id', 'mjd', 'passband',
                               'flux', 'flux_err']]
    df_selected.columns = ['object_id', 'time', 'band', 'flux', 'fluxerr']
    df_selected['zp'] = 25.0
    df_selected['zpsys'] = 'ab'
    # band_name = 'ugrizy'

    # name_list = [
    #     'lsst{}'.format(band_name[band])
    #     for band in df_selected['band']
    # ]
    # df_selected['band'] = name_list

    model = sncosmo.Model(source=source)
    parameter_names = model.param_names
    name_list = (
            parameter_names +
            ['{}_err'.format(name) for name in parameter_names] +
            ['chi_square']
    )
    meta = {name: 'f8' for name in name_list}

    df_selected = df_selected.groupby('object_id').apply(
        lambda x: fit(data=x, source=source),
        meta=meta
    )
    ddf = df_selected.compute()
    print(ddf)


def main():
    cmd()


if __name__ == '__main__':
    main()
