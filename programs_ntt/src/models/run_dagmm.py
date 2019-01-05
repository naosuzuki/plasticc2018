#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import xarray as xr
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

try:
    from dagmm import DAGMM
except ImportError:
    import sys
    sys.path.append(r'C:\Users\imoto\Documents\DAGMM')
    from dagmm import DAGMM

__author__ = 'Yasuhiro Imoto'
__date__ = '13/6/2018'


def load_data(path):
    ds = xr.open_dataset(path)
    # numpy形式に変換
    flux = ds.flux.values
    label = ds.label.values

    return flux, label


def print_metric(model, x, y):
    y_pred = model.predict(x)
    anomaly_energy_threshold = np.percentile(y_pred, 80)
    print("Energy thleshold to detect anomaly : {0:.3f}"
          .format(anomaly_energy_threshold))
    y_pred_flag = np.where(y_pred >= anomaly_energy_threshold, 1, 0)
    prec, recall, fscore, _ = precision_recall_fscore_support(
        y, y_pred_flag, average="binary"
    )
    print(" Precision = {0:.3f}".format(prec))
    print(" Recall    = {0:.3f}".format(recall))
    print(" F1-Score  = {0:.3f}".format(fscore))


def main():
    model = DAGMM(
        comp_hiddens=[60, 30, 10, 1], comp_activation=tf.nn.tanh,
        est_hiddens=[10, 4], est_dropout_ratio=0.5, est_activation=tf.nn.tanh,
        learning_rate=0.0001, epoch_size=1000, minibatch_size=1024,
        random_seed=1111
    )

    x_train, y_train = load_data(
        '../../data/processed/180112/dataset_selected/train/'
        'dataset.tr-2classes.nc'
    )
    print('# of samples: {}'.format(x_train.shape[0]))
    print('# of features: {}'.format(x_train.shape[1]))

    model.fit(x_train)

    print_metric(model=model, x=x_train, y=y_train)

    x_test, y_test = load_data(
        '../../data/processed/180112/dataset_selected/train/'
        'dataset.te-2classes.nc'
    )
    print_metric(model=model, x=x_test, y=y_test)


if __name__ == '__main__':
    main()
