#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import os

import pandas as pd
import luigi

__author__ = 'Yasuhiro Imoto'
__date__ = '13/12/2017'


class ParameterFile(luigi.ExternalTask):
    file_path = luigi.Parameter(
        '../../data/raw/param.lst/param.lst')   # type: str

    def output(self):
        return luigi.LocalTarget(self.file_path)


class ParameterLoader(luigi.Task):
    file_path = luigi.Parameter(
        '../../data/raw/param.lst/param.lst')   # type: str
    working_dir = luigi.Parameter(
        default='../../data/interim/real_bogus')  # type: str

    def requires(self):
        return ParameterFile(file_path=self.file_path)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.working_dir,
                                              'param.pickle'))

    def run(self):
        names = [
            'obsdate', 'filter', 'tract', 'patch',
            'ra', 'dec', 'xc', 'yc',
            'real/bogus', 'mag', 'magerr', 'elongation.norm',
            'fwhm.norm', 'significance.abs', 'residual', 'psffit.sigma.ratio',
            'psffit.peak.ratio', 'frac.det', 'density', 'density.good',
            'baPsf', 'sigmaPsf'
        ]
        df = pd.read_csv(self.input().path, delim_whitespace=True, header=None,
                         names=names)

        self.output().makedirs()
        df.to_pickle(self.output().path)


class RealBogusBandSplitter(luigi.Task):
    file_path = luigi.Parameter(
        '../../data/raw/param.lst/param.lst')  # type: str
    working_dir = luigi.Parameter(
        default='../../data/interim/real_bogus')  # type: str
    output_dir = luigi.Parameter(
        default='../../data/processed/real_bogus')  # type: str

    def requires(self):
        return ParameterLoader(file_path=self.file_path,
                               working_dir=self.working_dir)

    def output(self):
        # バンドの種類は手動で以下の5つと確認
        target = {
            name: luigi.LocalTarget(
                os.path.join(self.output_dir, 'param_{}.pickle'.format(name))
            )
            for name in ('HSC-I2', 'HSC-G', 'HSC-Y', 'HSC-Z', 'HSC-R2')
        }
        return target

    def run(self):
        df = pd.read_pickle(self.input().path)  # type: pd.DataFrame
        for band, obj in self.output().items():
            # bandに対応するデータのみを抜き出す
            tmp = df.loc[df['filter'] == band]

            obj.makedirs()
            tmp.to_pickle(obj.path)


if __name__ == '__main__':
    luigi.run()
