#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import luigi
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


def make_file_list(root_dir, sn_type=None):
    """
    指定されたディレクトリ内にあるindividual_parameter.datを再帰的に探索する
    `sn_type` がNoneの場合はテストデータと判断してパターンを変更する
    :param root_dir:
    :param sn_type:
    :return:
    """
    name_list = []

    if sn_type is not None:
        # 訓練データ
        r = re.compile(r'^{}[_-]'.format(sn_type))

        for root, dirs, files in os.walk(root_dir):
            if 'individual_parameter.dat' in files:
                root_path = Path(root)
                path = root_path.joinpath('individual_parameter.dat')
                m = r.match(root_path.name)
                if m is not None:
                    name_list.append(path)
    else:
        # テストデータ
        r = re.compile(r'\d+[a-z]+\.dat')

        for root, dirs, files in os.walk(root_dir):
            for name in files:
                m = r.search(name)
                if m is not None:
                    root_path = Path(root)
                    path = root_path.joinpath(name)
                    name_list.append(path)

    return name_list


def load_data_list(file_list, data_type, n_workers=2):
    df_list = joblib.Parallel(n_jobs=n_workers, verbose=8)([
        joblib.delayed(load_dat)(path) for path in file_list
    ])
    if data_type == 'train':
        name_list = joblib.Parallel(n_jobs=n_workers)([
            joblib.delayed(get_train_data_name)(path) for path in file_list
        ])
    else:
        name_list = joblib.Parallel(n_jobs=n_workers)([
            joblib.delayed(get_test_data_name)(path) for path in file_list
        ])
    return df_list, name_list


def load_dat(file_path):
    # 無効値を表す記号がNANだったと思う
    # 違う場合は、修正が必要
    df = pd.read_csv(str(file_path), delim_whitespace=True, header=0,
                     na_values='NAN')
    return df


def get_train_data_name(file_path):
    """
    TRAIN_ROOT/hoge/NAME/individual_parameter.datのNAMEの部分を抜き出す

    :param file_path:
    :return:
    """
    return Path(file_path).parent.name


test_data_name_pattern = re.compile(r'\d+([a-z]+)\.dat')


def get_test_data_name(file_path):
    m = test_data_name_pattern.search(str(file_path))
    if m is None:
        return Path(file_path).name
    else:
        return m.group(1)


def convert_data_format(df_list, name_list):
    """
    pd.DataFrameのlistをxr.Datasetに変換する
    読み込んだindividual_parameter.datの観測データの順序は全て同じと想定
    また、基準日からの経過日数も全て同じと想定
    違う場合はソートやマージが必要だが、想定するデータ数では
    かなりのコストになるので、省略している
    :param df_list:
    :param name_list:
    :return:
    """
    r = re.compile('_z(\d+\.\d+)_')

    n_points = df_list[0].shape[0]
    n_samples = len(df_list)

    flux = np.empty((n_samples, n_points))
    flux_err = np.empty_like(flux)
    sn_epoch = np.empty_like(flux)
    for i, df in enumerate(df_list):
        flux[i] = df['Flux']
        flux_err[i] = df['Flux_err']
        sn_epoch[i] = df['SN_epoch[day]']

    redshift = []
    for name in name_list:
        m = r.search(name)
        if m is None:
            # テストデータの場合
            v = -1
        else:
            v = float(m.group(1))
        redshift.append(v)
    redshift = np.asarray(redshift)

    # テストデータの場合、先頭の数字は名前を取得するときに取り除いている
    # 訓練データはそもそもの対象外
    # name_list = convert_name(name_list)

    band = df_list[0]['Filter']
    index = df_list[0]['Index']
    elapsed_day = df_list[0]['Elapsed_days[day]']

    # 横方向の軸'feature'は各列にユニークな値を割り振るためにだけ存在し、
    # ほとんど意味はない(実際になくてもエラーにならないかもしれない)
    # 実行上は'band'と'index'の組み合わせで必要な特徴を取り出すことになる
    ds = xr.Dataset({'flux': (['sample', 'feature'], flux),
                     'flux_err': (['sample', 'feature'], flux_err),
                     'sn_epoch': (['sample', 'feature'], sn_epoch),
                     'redshift': (['sample'], redshift)},
                    coords={'feature': list(range(len(band))),
                            'band': ('feature', band),
                            'index': ('feature', index),
                            'elapsed_day': ('feature', elapsed_day),
                            'sample': name_list})
    return ds


def convert_name(name_list):
    r = re.compile(r'^\d+')  # 名前の先頭が数字
    new_list = [r.sub('', name) for name in name_list]
    return new_list


class InputTrainDirectory(luigi.ExternalTask):
    """
    クラスごとに学習データが存在するかを確認する
    ただし、各ファイルを確認するのは大変なので、ディレクトリの有無で判定する
    """
    train_root = luigi.Parameter(default='../../data/raw/train')  # type: str

    def output(self):
        path = Path(self.train_root)
        return luigi.LocalTarget(str(path))


class InputTestDirectory(luigi.ExternalTask):
    test_root = luigi.Parameter(default='../../data/raw/test')  # type: str

    def output(self):
        return luigi.LocalTarget(self.test_root)


class FileList(luigi.Task):
    data_dir = luigi.Parameter()  # type: str
    data_type = luigi.ChoiceParameter(choices=['train', 'test'])  # type: str
    sn_type = luigi.Parameter(default=None)  # type: str

    working_dir = luigi.Parameter(
        default='../../data/interim/train')  # type: str

    def requires(self):
        if self.data_type == 'train':
            return InputTrainDirectory(train_root=self.data_dir)
        else:
            return InputTestDirectory(test_root=self.data_dir)

    def output(self):
        path = Path(self.working_dir)
        if self.data_type == 'train':
            assert self.sn_type is not None, 'you forgot to set --sn-type='
            name = '{}-file_list-{}.pickle'.format(self.data_type,
                                                   self.sn_type)
            path = path.joinpath(name)
        else:
            path = path.joinpath('test-file_list.pickle')
        return luigi.LocalTarget(str(path))

    def run(self):
        """
        指定されたディレクトリを再帰的に巡回してデータのファイル一覧を作成する
        :return:
        """
        if self.data_type == 'train':
            file_list = make_file_list(self.input().path, sn_type=self.sn_type)
        else:
            file_list = make_file_list(self.input().path)

        self.output().makedirs()
        joblib.dump(file_list, self.output().path)


class DataLoader(luigi.Task):
    data_dir = luigi.Parameter()  # type: str
    data_type = luigi.ChoiceParameter(choices=['train', 'test'])  # type: str
    sn_type = luigi.Parameter(default=None)  # type: str

    working_dir = luigi.Parameter(
        default='../../data/interim/train')  # type: str
    n_workers = luigi.IntParameter(default=1)   # type: int

    def requires(self):
        return FileList(data_dir=self.data_dir, data_type=self.data_type,
                        sn_type=self.sn_type, working_dir=self.working_dir)

    def output(self):
        path = Path(self.working_dir)
        if self.data_type == 'train':
            path = path.joinpath('dataset.{}.all.nc'.format(self.sn_type))
        else:
            path = path.joinpath('dataset.test.all.nc')
        return luigi.LocalTarget(str(path))

    def run(self):
        file_list = joblib.load(self.input().path)
        df_list, name_list = load_data_list(file_list, self.data_type,
                                            n_workers=self.n_workers)
        ds = convert_data_format(df_list, name_list)

        self.output().makedirs()
        ds.to_netcdf(self.output().path, format='NETCDF4_CLASSIC')


class DataSplitter(luigi.Task):
    train_root = luigi.Parameter(default='../../data/raw/train')  # type: str
    sn_type = luigi.Parameter()  # type: str

    working_dir = luigi.Parameter(
        default='../../data/interim/train')  # type: str
    n_workers = luigi.IntParameter(default=1)  # type: int

    split_ratio = luigi.TupleParameter(default=(8, 1, 1))  # type: tuple
    seed = luigi.IntParameter(default=0x5eed)   # type: int

    def requires(self):
        return DataLoader(data_dir=self.train_root, data_type='train',
                          sn_type=self.sn_type, working_dir=self.working_dir,
                          n_workers=self.n_workers)

    def output(self):
        path = Path(self.working_dir)
        outputs = {
            key: luigi.LocalTarget(str(path.joinpath(
                'dataset.{}.{}.ratio{}_{}_{}.seed{}.nc'.format(
                    self.sn_type, key, self.split_ratio[0],
                    self.split_ratio[1], self.split_ratio[2], self.seed
                ))))
            for key in ('tr', 'va', 'te')
        }
        return outputs

    def run(self):
        ds = xr.open_dataset(self.input().path)  # type: xr.Dataset
        n_samples = ds.flux.shape[0]

        ratio1 = np.sum(self.split_ratio[1:]) / np.sum(self.split_ratio)
        train_index, tmp = train_test_split(
            list(range(n_samples)), test_size=ratio1, random_state=self.seed
        )

        ratio2 = self.split_ratio[2] / np.sum(self.split_ratio[1:])
        validation_index, test_index = train_test_split(
            tmp, test_size=ratio2, random_state=self.seed + 1
        )

        train_ds = ds.isel(sample=train_index)
        validation_ds = ds.isel(sample=validation_index)
        test_ds = ds.isel(sample=test_index)
        split = {'tr': train_ds, 'va': validation_ds, 'te': test_ds}

        for key in split:
            self.output()[key].makedirs()
            split[key].to_netcdf(self.output()[key].path,
                                 format='NETCDF4_CLASSIC')


class TrainDataCombiner(luigi.Task):
    train_root = luigi.Parameter(default='../../data/raw/train')  # type: str
    sn_types = luigi.DictParameter()  # type: dict

    working_dir = luigi.Parameter(
        default='../../data/interim/train')  # type: str
    n_workers = luigi.IntParameter(default=1)  # type: int

    split_ratio = luigi.TupleParameter(default=(8, 1, 1))  # type: tuple
    seed = luigi.IntParameter(default=0x5eed)  # type: int

    train_data_type = luigi.ChoiceParameter(choices=['tr', 'va', 'te'])
    output_dir = luigi.Parameter(
        default='../../data/processed/train')  # type: str
    suffix = luigi.Parameter(default='')  # type: str

    def requires(self):
        tasks = {
            sn_type: DataSplitter(
                train_root=self.train_root, sn_type=sn_type,
                working_dir=self.working_dir, n_workers=self.n_workers,
                split_ratio=self.split_ratio, seed=self.seed)
            for sn_type in self.sn_types
        }
        return tasks

    def output(self):
        path = Path(self.output_dir)

        if self.suffix == '':
            path = path.joinpath('dataset.{}.nc'.format(self.train_data_type))
        else:
            path = path.joinpath('dataset.{}-{}.nc'.format(
                self.train_data_type, self.suffix))

        return luigi.LocalTarget(str(path))

    def run(self):
        ds_list = []
        for sn_type, label in self.sn_types.items():
            path = self.input()[sn_type][self.train_data_type].path
            ds = xr.open_dataset(path)

            sample_names = ds.sample
            tmp = np.empty(len(sample_names.values))
            tmp[:] = int(label)

            da = xr.DataArray(tmp, coords=[sample_names], dims=['sample'])
            # ラベルを追加
            ds['label'] = da

            ds_list.append(ds)

        # データを一つにまとめる
        ds = xr.concat(ds_list, 'sample')   # type: xr.Dataset
        # シャッフル
        index = list(range(ds.flux.shape[0]))
        np.random.shuffle(index)
        ds = ds.isel(sample=index)

        self.output().makedirs()
        ds.to_netcdf(self.output().path, format='NETCDF4_CLASSIC')


class TrainData(luigi.WrapperTask):
    train_root = luigi.Parameter(default='../../data/raw/train')  # type: str
    sn_types = luigi.DictParameter()  # type: dict

    working_dir = luigi.Parameter(
        default='../../data/interim/train')  # type: str
    n_workers = luigi.IntParameter(default=1)  # type: int

    split_ratio = luigi.TupleParameter(default=(8, 1, 1))  # type: tuple
    seed = luigi.IntParameter(default=0x5eed)  # type: int

    output_dir = luigi.Parameter(
        default='../../data/processed/train')  # type: str
    suffix = luigi.Parameter(default='')  # type: str

    def requires(self):
        tasks = {
            key: TrainDataCombiner(
                train_root=self.train_root, sn_types=self.sn_types,
                working_dir=self.working_dir, n_workers=self.n_workers,
                split_ratio=self.split_ratio, seed=self.seed,
                train_data_type=key, output_dir=self.output_dir,
                suffix=self.suffix)
            for key in ('tr', 'va', 'te')
        }
        return tasks


class TestData(luigi.Task):
    test_root = luigi.Parameter(default='../../data/raw/test')  # type: str
    sn_types = luigi.DictParameter(default={})  # type: dict

    working_dir = luigi.Parameter(
        default='../../data/interim/test')  # type: str
    n_workers = luigi.IntParameter(default=1)  # type: int

    output_dir = luigi.Parameter(
        default='../../data/processed/test')  # type: str
    suffix = luigi.Parameter(default='')  # type: str
    label_file_path = luigi.Parameter(default=None)  # type: str

    def requires(self):
        return DataLoader(data_dir=self.test_root, data_type='test',
                          sn_type='dummy', working_dir=self.working_dir,
                          n_workers=self.n_workers)

    def output(self):
        path = Path(self.output_dir)
        if self.suffix == '':
            path = path.joinpath('dataset.test.all.nc')
        else:
            path = path.joinpath('dataset.test.all-{}.nc'.format(self.suffix))
        return luigi.LocalTarget(str(path))

    def run(self):
        labels = {}

        if self.label_file_path is not None:
            label_file = Path(self.label_file_path)
            if label_file.exists():
                # ファイルが存在するなら読み込む
                # ファイルは各行に名前とタイプがスペース区切りであるとする
                df = pd.read_csv(label_file, delim_whitespace=True,
                                 header=None, index_col=0)

                # IInはnが小文字でいろいろと不都合なので、大文字に変換する
                r = re.compile(r'IIn')

                for key, row in df.iterrows():
                    t = r.sub('IIN', row[1])
                    labels[key] = self.sn_types[t]

        ds = xr.open_dataset(self.input().path)

        sample_names = ds.sample
        tmp = np.empty(len(sample_names.values))
        for i, name in enumerate(sample_names.values):
            if not isinstance(name, str):
                name = name.decode()
            tmp[i] = labels.get(name, -1)
        da = xr.DataArray(tmp, coords=[sample_names], dims=['sample'])
        ds['label'] = da

        self.output().makedirs()
        ds.to_netcdf(self.output().path, format='NETCDF4_CLASSIC')


if __name__ == '__main__':
    luigi.run()
