#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
学習したモデルの精度の確認を行う
pauc_exactとpauc_relaxedはスコア関数は共通なので、どちらでも動く
また、random_forestとxgboostも関数の形式が共通なので、どちらでも動く

それぞれで関数を用意する必要はなかった
"""

import json
import os

import click
import numpy as np
import pandas as pd
import tensorflow as tf

from pauc import predict_pauc
from pauc.score import get_score_model
from random_forest import predict_random_forest
from train_real_bogus_model import convert_data, split_data

__author__ = 'Yasuhiro Imoto'
__date__ = '14/12/2017'


@click.group()
def predict_model():
    pass


def predict_pauc_model(data_path, output_dir):
    with open(os.path.join(output_dir, 'parameters.json'), 'r') as f:
        parameters = json.load(f)
    split_ratio = parameters['split_ratio']
    seed = parameters['seed']

    data, label = convert_data(data_path=data_path)

    d = {}
    model_type = parameters['model_type']
    if model_type == 'dnn':
        d['hidden_size'] = parameters['hidden_size']
    else:
        d['positive_components'] = parameters['positive_components']
        d['negative_components'] = parameters['negative_components']

    data_size = data.shape[1]
    model = get_score_model(model_type=model_type, positive_data=data_size,
                            negative_data=data_size, **d)

    mean = np.array(parameters['mean']).astype(np.float32)
    std = np.array(parameters['std']).astype(np.float32)

    # データ全体を変換
    predict_pauc({'x': data, 'y': label}, output_dir,
                 'data-all.csv', model=model, mean=mean, std=std)

    # 学習結果の確認
    dataset = split_data(ds=data, label=label,
                         split_ratio=split_ratio, seed=seed)
    for name in ('train', 'validation', 'test'):
        predict_pauc(dataset[name], output_dir,
                     'data-{}.csv'.format(name),
                     model=model, mean=mean, std=std)

        score_file = os.path.join(output_dir, 'data-{}.csv'.format(name))
        df = summarize_result(score_file, [0.01, 0.05, 0.1, 0.5, 1.0])
        df.to_csv(os.path.join(output_dir, 'summary-{}.csv'.format(name)))


def summarize_result(file_name, fpr_list):
    df = pd.read_csv(file_name, header=None, delim_whitespace=True,
                     names=['score', 'label'])
    label = df['label']
    positive = df.loc[label == 1, 'score']
    negative = df.loc[label == 0, 'score']

    negative = -np.sort(-negative)

    result = []
    for fpr in fpr_list:
        pauc, tpr = compute_pauc(positive, negative, fpr)
        result.append((pauc, tpr))
    df = pd.DataFrame(result, index=fpr_list, columns=['pAUC', 'TPR'])

    return df


def compute_pauc(positive, negative, fpr):
    if fpr == 1.0:
        tpr = 1.0

        size = float(len(positive) * len(negative))
        pauc = np.count_nonzero(positive[:, np.newaxis] > negative) / size
    else:
        n = int(fpr * len(negative))
        threshold = negative[n]

        m = float(len(positive))
        tpr = np.count_nonzero(positive > threshold) / m

        flag = positive[:, np.newaxis] > negative[:n]
        pauc = np.count_nonzero(flag) / (m * n)

    return pauc, tpr


@predict_model.command(name='pauc')
@click.option('--data-path', type=click.Path(exists=True))
@click.option('--output-dir', type=click.Path(file_okay=False))
def pauc_model(data_path, output_dir):
    predict_pauc_model(data_path, output_dir)


@predict_model.command()
@click.option('--data-path', type=str)
@click.option('--data_path', type=str)
@click.option('--output-dir', type=str)
@click.option('--output_dir', type=str)
def pauc_relaxed(data_path, output_dir):
    predict_pauc_model(data_path, output_dir)


@predict_model.command()
@click.option('--data-path', type=str)
@click.option('--data_path', type=str)
@click.option('--output-dir', type=str)
@click.option('--output_dir', type=str)
def pauc_exact(data_path, output_dir):
    predict_pauc_model(data_path, output_dir)


def predict_rf_model(data_path, output_dir):
    with open(os.path.join(output_dir, 'parameters.json'), 'r') as f:
        parameters = json.load(f)
    split_ratio = parameters['split_ratio']
    seed = parameters['seed']

    data, label = convert_data(data_path=data_path)

    predict_random_forest({'x': data, 'y': label}, output_dir,
                          'data-all.csv')

    # 学習結果の確認
    dataset = split_data(ds=data, label=label,
                         split_ratio=split_ratio, seed=seed)
    for name in ('train', 'validation', 'test'):
        predict_random_forest(dataset[name], output_dir,
                              'data-{}.csv'.format(name))

        score_file = os.path.join(output_dir, 'data-{}.csv'.format(name))
        df = summarize_result(score_file, [0.01, 0.05, 0.1, 0.5, 1.0])
        df.to_csv(os.path.join(output_dir, 'summary-{}.csv'.format(name)))


@predict_model.command()
@click.option('--data_path', type=str)
@click.option('--output_dir', type=str)
def random_forest(data_path, output_dir):
    predict_rf_model(data_path=data_path, output_dir=output_dir)


@predict_model.command()
@click.option('--data_path', type=str)
@click.option('--output_dir', type=str)
def xgboost(data_path, output_dir):
    predict_rf_model(data_path=data_path, output_dir=output_dir)


def main():
    predict_model()


if __name__ == '__main__':
    main()
