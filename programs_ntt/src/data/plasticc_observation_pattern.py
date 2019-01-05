#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

from pathlib import Path

import click
import numpy as np
import pandas as pd
import dask.dataframe as dd

__author__ = 'Yasuhiro Imoto'
__date__ = '27/12/2018'


@click.command()
@click.option('--data-path', type=click.Path(exists=True),
              default='../../data/raw/training_set.csv')
@click.option('--max-length', type=int, default=1100)
@click.option('--output-path', type=click.Path())
def cmd(data_path, max_length, output_path):
    output_path = Path(output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    df = dd.read_csv(data_path, header=0)
    tmp = df.groupby('object_id').apply(
        lambda x: make_mask(x, max_length=max_length),
        meta={'mask': np.float64}
    ).compute()

    mask = tmp.values.reshape([-1, 6, max_length])
    flag = np.logical_or(mask[:, 0], mask[:, 1])
    for i in range(2, 6):
        flag = np.logical_or(flag, mask[:, i])
    index = np.arange(max_length) + 1
    tmp = flag * index
    size = np.max(tmp, axis=1)

    np.savez_compressed(str(output_path), mask=mask, size=size)


def make_mask(df, max_length):
    df['mjd'] = df['mjd'] - np.min(df['mjd'])

    mask = np.zeros((6, max_length), dtype=np.bool)
    for i, row in df.iterrows():
        mask[int(row['passband']), int(np.round(row['mjd']))] = True

    mask = pd.DataFrame({'mask': mask.flat})
    return mask


def main():
    cmd()


if __name__ == '__main__':
    main()
