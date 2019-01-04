#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import json
import os

import click
import numpy as np
import pandas as pd
import xarray as xr

try:
    from tensorflow import keras
except ImportError:
    from tensorflow.contrib.keras.python import keras

from train_normal_classifier_model import NovaTypeSequence
from train_normal_redshift_regressor import r2_score

__author__ = 'Yasuhiro Imoto'
__date__ = '28/8/2017'


@click.command()
@click.argument('task_type',
                type=click.Choice(['sn_class', 'redshift', 'sn_epoch']))
@click.option('--batch_size', type=int, default=1000)
@click.option('--train_data_path',
              type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('--validation_data_path',
              type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('--test_data_path',
              type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('--real_data_path', type=click.Path(exists=True, dir_okay=False),
              default=None)
@click.option('--output_dir', type=click.Path(exists=True, dir_okay=True))
def run_prediction(task_type, batch_size, train_data_path,
                   validation_data_path, test_data_path, real_data_path,
                   output_dir):
    model_path = os.path.join(output_dir, 'model.h5')
    assert os.path.exists(model_path), 'model file is not found.'

    summary_path = os.path.join(output_dir, 'summary.json')
    assert os.path.exists(summary_path), 'summary file is not found.'

    with open(summary_path, 'r') as f:
        parameters = json.load(f)
    band_data = parameters['band_data']
    method = parameters['method']
    mean = np.asarray(parameters['mean'])
    std = np.asarray(parameters['std'])
    output_size = parameters.get('output_size', 1)
    use_redshift = parameters.get('use_redshift', False)

    model = keras.models.load_model(model_path,
                                    custom_objects={'r2_score': r2_score})
    for tag, path in zip(['train', 'validation', 'test', 'real'],
                         [train_data_path, validation_data_path,
                          test_data_path, real_data_path]):
        if path is None:
            continue

        # predictionに関しては、どのタスクでも入力データは同じ
        # redshiftの回帰タスクについては、use_redshift=Falseでよい
        data = NovaTypeSequence(
            file_path=path, band_data=band_data, n_classes=output_size,
            batch_size=batch_size, use_redshift=use_redshift,
            train=False, method=method, mean=mean, std=std, prediction=True
        )
        prediction = model.predict_generator(generator=data, steps=len(data))
        name_list = data.ds.name
        if isinstance(name_list[0], bytes):
            name_list = [name.decode() for name in name_list]

        if task_type == 'sn_class':
            target_value = data.ds.label
            target_name = 'label'
        elif task_type == 'redshift':
            target_value = data.ds.redshift
            target_name = 'redshift'

            # 学習時に正規化をしたので、その逆変換
            prediction = 2 * prediction + 1
        elif task_type == 'sn_epoch':
            # 明るさのピーク日付と観測の基準日との日数差
            target_value = (np.min(data.ds.sn_epoch, axis=1) -
                            data.ds.min_elapsed_day)
            target_name = 'sn_epoch'

            # 学習時に正規化をしたので、その逆変換
            prediction = 30 * prediction
        else:
            raise ValueError()

        df = pd.DataFrame(prediction, index=name_list)
        df[target_name] = target_value
        df.to_csv(os.path.join(output_dir, '{}.csv'.format(tag)))

        if task_type == 'sn_class':
            ds = xr.Dataset(
                {'prediction': (['sample', 'nova_type'], prediction),
                 target_name: (['sample'], target_value)},
                coords={'sample': name_list, 'nova_type': range(output_size)}
            )
        else:
            ds = xr.Dataset(
                {'prediction': (['sample'], np.squeeze(prediction)),
                 target_name: (['sample'], target_value)},
                coords={'sample': name_list}
            )
        ds.to_netcdf(os.path.join(output_dir, '{}.nc'.format(tag)),
                     format='NETCDF4_CLASSIC')


def main():
    run_prediction()


if __name__ == '__main__':
    main()
