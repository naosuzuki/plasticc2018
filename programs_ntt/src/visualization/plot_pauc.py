#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import glob
import os
import re

import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import matplotlib.pyplot as plt
    import seaborn as sns

__author__ = 'Yasuhiro Imoto'
__date__ = '15/12/2017'


def get_file_list(dir_name):
    return glob.glob(os.path.join(dir_name, '*', 'data-test.csv'))


def load_result(file_path):
    df = pd.read_csv(file_path, delim_whitespace=True, header=None,
                     names=['score', 'label'])
    return df


def compute_tpr(df, fpr):
    """
    指定されたFPRのときのTPRを求める

    :param df:
    :param fpr:
    :return:
    """
    positive = df.loc[df['label'] == 1, 'score']
    negative = df.loc[df['label'] == 0, 'score']

    # FPRに対応するスコアを求める
    s = -np.sort(-negative)
    threshold = s[int(len(negative) * fpr)]

    tpr = np.count_nonzero(positive > threshold) / len(positive)
    return tpr


def get_parameters(file_list):
    r = re.compile(r'(HSC-.+)-(.+)-(.+)-(.+)[/\\]')
    parameters = []
    for f in file_list:
        m = r.search(f)
        band = m.group(1)
        gamma = float(m.group(2))
        weight1 = float(m.group(3))
        weight2 = float(m.group(4))
        parameters.append((band, gamma, weight1, weight2))

    df = pd.DataFrame(parameters,
                      columns=['band', 'gamma', 'weight1', 'weight2'])
    return df


def compute_result(file_list):
    tpr_list = []
    for f in file_list:
        df = load_result(f)
        tpr1 = compute_tpr(df, 0.01)
        tpr5 = compute_tpr(df, 0.05)
        tpr10 = compute_tpr(df, 0.1)
        tpr_list.append((tpr1, tpr5, tpr10))
    tpr = pd.DataFrame(tpr_list, columns=['TPR@1', 'TPR@5', 'TPR@10'])

    parameters = get_parameters(file_list)

    df = pd.concat([parameters, tpr], axis=1)
    return df


def plot_result(df, band, file_name):
    # noinspection PyTypeChecker
    fig, axes = plt.subplots(nrows=3, sharex=True)

    #
    df.sort_values('gamma', inplace=True)
    band_data = df.loc[df['band'] == band]
    for (weight1, weight2), group in band_data.groupby(['weight1', 'weight2']):
        label = r'$\lambda$=({}, {})'.format(weight1, weight2)
        for ax, i in zip(axes, [1, 5, 10]):
            ax.plot(group['gamma'], group['TPR@{}'.format(i)], label=label)
    for ax, fpr in zip(axes, [0.01, 0.05, 0.1]):
        ax.set_title('FPR={}'.format(fpr))
        ax.set_ylabel('TPR')
        ax.grid()
        ax.legend(loc='best')
    axes[-1].set_xlabel(r'$\gamma$')
    fig.suptitle(band)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.savefig(file_name, bbox_inches='tight')
    plt.close()


def main():
    result_dir = '../../models/real_bogus/pauc/relaxed'
    file_list = get_file_list(result_dir)

    df = compute_result(file_list)  # type: pd.DataFrame
    output_dir = '../../reports/real_bogus/pauc/relaxed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for band in df['band'].unique():
        # noinspection PyTypeChecker
        plot_result(df, band, os.path.join(output_dir, '{}.png'.format(band)))


if __name__ == '__main__':
    main()
