#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import os
import warnings
import time
from itertools import product

import dask.dataframe as dd
import luigi
import numpy as np
import pandas as pd
import sncosmo
from astropy.table import Table
from tqdm import tqdm
from numba import jit

__author__ = 'Yasuhiro Imoto'
__date__ = '21/12/2018'


class TrainingSet(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../../data/raw/training_set.csv')


class TrainingSetMetadata(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../../data/raw/training_set_metadata.csv')


class TestSet(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../../data/raw/test_set.csv')


class TestSetMetadata(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../../data/raw/test_set_metadata.csv')


class Split(luigi.Task):
    index = luigi.IntParameter(default=0)  # type: int
    n_split = luigi.IntParameter(default=500)  # type: int
    is_training = luigi.BoolParameter()  # type: bool

    output_dir = luigi.Parameter(
        default='../../data/interim/dataset'
    )  # type: str

    def requires(self):
        if self.is_training:
            d = dict(dataset=TrainingSet(), metadata=TrainingSetMetadata())
        else:
            d = dict(dataset=TestSet(), metadata=TestSetMetadata())
        return d

    def output(self):
        meta = dd.read_csv(self.input()['metadata'].path, header=0)

        selected = meta.loc[meta['object_id'] % self.n_split == self.index,
                            'object_id'].compute()

        mode = 'training' if self.is_training else 'test'
        target_list = []
        for i in selected.values:
            path = os.path.join(
                self.output_dir, mode, '{}.pickle'.format(i)
            )
            target = luigi.LocalTarget(path)
            target_list.append(target)
        return target_list

    def run(self):
        mode = 'training' if self.is_training else 'test'
        output_dir = os.path.join(self.output_dir, mode)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df = dd.read_csv(self.input()['dataset'].path, header=0)
        df_selected = df[df['object_id'] % self.n_split == self.index]
        df_selected = df_selected[['object_id', 'mjd', 'passband',
                                   'flux', 'flux_err']]
        df_selected.columns = ['object_id', 'time', 'band', 'flux', 'fluxerr']
        df_selected['zp'] = 25.0
        df_selected['zpsys'] = 'ab'
        df_selected = df_selected.compute()

        band_name = 'ugrizy'

        name_list = [
            'lsst{}'.format(band_name[band])
            for band in df_selected['band'].values
        ]
        df_selected['band'] = name_list

        for object_id, group in df_selected.groupby('object_id'):
            group.to_pickle(
                os.path.join(output_dir, '{}.pickle'.format(object_id))
            )


class Fit(luigi.Task):
    """
    モデル一覧
    候補が複数あるタイプについては何となくで選択

    Ia		salt2, salt2-extended
    Ib/c	nugent-sn1bc, nugent-hyper
    Ib		s11-2005hl, s11-2006jo
    Ic		s11-2006fo, snana-2004fe
    IIP		nugent-sn2p, snana-2007pg
    IIL		nugent-sn2l
    IIn		nugent-sn2n, snana-2006ix
    IIL/P	s11-2004hx
    II-pec	snana-2007ms
    PopIII	whalen-z15b, whalen-z40g
    """
    index = luigi.IntParameter(default=0)   # type: int
    n_split = luigi.IntParameter(default=500)   # type: int
    is_training = luigi.BoolParameter()     # type: bool

    output_dir = luigi.Parameter(
        default='../../data/interim/features'
    )  # type: str

    source = luigi.ChoiceParameter(choices=[
        'salt2', 'salt2-extended', 'nugent-sn1bc', 'nugent-hyper',
        's11-2005hl', 's11-2006jo', 's11-2006fo', 'snana-2004fe',
        'nugent-sn2p', 'snana-2007pg', 'nugent-sn2l',
        'nugent-sn2n', 'snana-2006ix', 's11-2004hx',
        'snana-2007ms', 'whalen-z15b', 'whalen-z40g'
    ])  # type: str

    def requires(self):
        if self.is_training:
            d = dict(dataset=TrainingSet(), metadata=TrainingSetMetadata())
        else:
            d = dict(dataset=TestSet(), metadata=TestSetMetadata())

        # for i in range(self.index, 4096, self.n_split):
        #     task = Split(index=i, n_split=4096, is_training=self.is_training)
        #     d[i] = task

        return d

    def output(self):
        mode = 'training' if self.is_training else 'test'
        path = os.path.join(
            self.output_dir, mode, self.source,
            '{0:03d}.pickle'.format(self.index)
        )
        target = luigi.LocalTarget(path)

        return target

    def run(self):
        begin = time.time()

        meta = dd.read_csv(self.input()['metadata'].path, header=0)

        selected = meta[meta['object_id'] % self.n_split == self.index]
        selected = selected[['object_id', 'hostgal_photoz']]
        meta = selected.compute()

        model = sncosmo.Model(source=self.source)
        parameter_names = model.param_names
        n = len(parameter_names)
        features = np.empty((len(meta), n * 2 + 1))

        mode = 'training' if self.is_training else 'test'
        data_dir = os.path.join('../../data/interim/dataset', mode)

        index_list = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i, (object_id, z) in enumerate(zip(
                    meta['object_id'].values, meta['hostgal_photoz'].values)):
                index_list.append(object_id)

                df = pd.read_pickle(
                    os.path.join(data_dir, '{}.pickle'.format(object_id))
                )
                table = Table.from_pandas(df)

                if np.isnan(z) or z == 0:
                    z_range = [0.0, 1.0]
                else:
                    z_range = [np.max([z - 0.5, 0]), z + 0.5]
                model = sncosmo.Model(source=self.source)
                try:
                    result, fitted_model = sncosmo.fit_lc(
                        data=table, model=model, vparam_names=parameter_names,
                        bounds={'z': z_range}
                    )
                except (RuntimeError, sncosmo.fitting.DataQualityError):
                    features[i] = -1
                    # print(object_id)
                    continue

                features[i, :n] = result.parameters
                for j, name in enumerate(parameter_names):
                    features[i, n + j] = result.errors[name]
                features[i, -1] = result.chisq

        columns = (parameter_names +
                   ['{}_err'.format(name) for name in parameter_names] +
                   ['chi_square'])
        features = pd.DataFrame(
            features, columns=columns, index=index_list
        )
        features.index.name = 'object_id'

        self.output().makedirs()
        # features.to_csv(self.output().path)
        features.to_pickle(self.output().path)

        end = time.time()
        print(end - begin)


class Merge(luigi.Task):
    n_split = luigi.IntParameter(default=500)  # type: int
    is_training = luigi.BoolParameter()  # type: bool

    output_dir = luigi.Parameter(
        default='../../data/interim/features'
    )  # type: str

    source = luigi.ChoiceParameter(choices=[
        'salt2', 'salt2-extended', 'nugent-sn1bc', 'nugent-hyper',
        's11-2005hl', 's11-2006jo', 's11-2006fo', 'snana-2004fe',
        'nugent-sn2p', 'snana-2007pg', 'nugent-sn2l',
        'nugent-sn2n', 'snana-2006ix', 's11-2004hx',
        'snana-2007ms', 'whalen-z15b', 'whalen-z40g'
    ])  # type: str

    def requires(self):
        t = [Fit(index=i, n_split=self.n_split, output_dir=self.output_dir,
                 source=self.source, is_training=self.is_training)
             for i in range(self.n_split)]
        return t

    def output(self):
        mode = 'training' if self.is_training else 'test'
        path = os.path.join(
            self.output_dir,
            '{}_{}_feature.pickle'.format(mode, self.source)
        )
        return luigi.LocalTarget(path)

    def run(self):
        df_list = []
        for obj in self.input():
            df_list.append(pd.read_pickle(obj.path))
        df = pd.concat(df_list, axis=0, sort=True)

        df.to_pickle(self.output().path)


class ModelDownload(luigi.WrapperTask):
    def requires(self):
        source_list = [
            'salt2', 'salt2-extended', 'nugent-sn1bc', 'nugent-hyper',
            's11-2005hl', 's11-2006jo', 's11-2006fo', 'snana-2004fe',
            'nugent-sn2p', 'snana-2007pg', 'nugent-sn2l',
            'nugent-sn2n', 'snana-2006ix', 's11-2004hx',
            'snana-2007ms', 'whalen-z15b', 'whalen-z40g'
        ]

        tasks = [Fit(index=0, n_split=256, is_training=True, source=s)
                 for s in source_list]
        return tasks


class ConvertGroup1(luigi.WrapperTask):
    def requires(self):
        source_list = [
            'salt2', 'nugent-hyper', 's11-2006jo', 'snana-2004fe',
            'nugent-sn2n', 's11-2004hx'
        ]

        tasks = []
        for source, index in product(source_list, range(512)):
            tasks.append(
                Fit(index=index, n_split=512, is_training=False, source=source)
            )
        return tasks


class ConvertGroup2(luigi.WrapperTask):
    def requires(self):
        source_list = [
            'snana-2007pg', 'snana-2006ix', 'whalen-z40g',
            'nugent-sn2p', 'nugent-sn2l'
        ]

        tasks = []
        for source, index in product(source_list, range(512)):
            tasks.append(
                Fit(index=index, n_split=512, is_training=False, source=source)
            )
        return tasks


if __name__ == '__main__':
    luigi.run()
