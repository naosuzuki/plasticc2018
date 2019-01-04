#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import os

import click
import numpy as np
import pandas as pd
import xarray as xr

__author__ = 'Yasuhiro Imoto'
__date__ = '20/2/2018'


@click.command()
@click.option('--sn_class_dir', type=click.Path(exists=True, dir_okay=True))
@click.option('--redshift_dir', type=click.Path(exists=True, dir_okay=True))
@click.option('--sn_epoch_dir', type=click.Path(exists=True, dir_okay=True))
@click.option('--output_dir', type=click.Path(exists=False, dir_okay=True))
def cmd(sn_class_dir, redshift_dir, sn_epoch_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sn_class_result = xr.open_dataset(os.path.join(sn_class_dir, 'real.nc'))
    redshift_result = xr.open_dataset(os.path.join(redshift_dir, 'real.nc'))
    sn_epoch_result = xr.open_dataset(os.path.join(sn_epoch_dir, 'real.nc'))

    merged = merge(sn_class_result=sn_class_result,
                   redshift_result=redshift_result,
                   sn_epoch_result=sn_epoch_result)  # type: pd.DataFrame

    merged.to_csv(os.path.join(output_dir, 'real.csv'))


def merge(sn_class_result, redshift_result, sn_epoch_result):
    """
    各タスクの結果を一つのファイルにまとめる

    :param sn_class_result:
    :param redshift_result:
    :param sn_epoch_result:
    :return:
    """
    # sn_classの結果
    sn_class_prediction = sn_class_result.prediction.values
    name_list = get_name_list(sn_class_result)
    n_classes = sn_class_prediction.shape[1]
    df_sn_class = pd.DataFrame(sn_class_prediction, index=name_list,
                               columns=range(n_classes))

    redshift_prediction = redshift_result.prediction.values
    df_redshift = pd.DataFrame(redshift_prediction, columns=['redshift'],
                               index=get_name_list(redshift_result))

    sn_epoch_prediction = sn_epoch_result.prediction.values
    df_sn_epoch = pd.DataFrame(sn_epoch_prediction, columns=['sn_epoch'],
                               index=get_name_list(sn_epoch_result))
    # 結合
    df = pd.concat([df_sn_class, df_redshift, df_sn_epoch],
                   axis=1)  # type: pd.DataFrame
    # 以前の処理では、クラスの確率で並び替えていたので、同じ仕様にする
    df.sort_values(list(range(n_classes)), ascending=False, inplace=True)

    return df


def get_name_list(ds):
    name_list = ds.sample.values
    # 型がstrではないと思うので、変換
    if isinstance(name_list[0], bytes):
        name_list = [name.decode() for name in name_list]
    return name_list


def main():
    cmd()


if __name__ == '__main__':
    main()
