#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
import math

import click
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow import keras
from sklearn.externals import joblib
import sklearn.utils
import pandas as pd

__author__ = 'Yasuhiro Imoto'
__date__ = '29/10/2018'


def make_model(input_shape, n_classes):
    inputs = keras.Input(shape=input_shape)

    dense1 = keras.layers.Dense(units=100, input_shape=input_shape)
    h = dense1(inputs)

    for i in range(4):
        s = keras.Sequential([
            keras.layers.BatchNormalization(
                input_shape=dense1.output_shape[1:]
            ),
            keras.layers.Activation('selu'),
            keras.layers.Dense(units=100),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('selu'),
            keras.layers.Dense(units=100)
        ])
        g = s(h)
        h = keras.layers.Add()([g, h])

    dense2 = keras.layers.Dense(
        units=n_classes, activation='softmax',
        input_shape=dense1.output_shape[1:]
    )

    outputs = dense2(h)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


class Dataset(keras.utils.Sequence):
    def __init__(self, flux, flux_err, label, batch_size, noise, n_classes,
                 name):
        v = sklearn.utils.shuffle(
            flux.astype(np.float32), flux_err.astype(np.float32),
            label.astype(np.int32), name
        )
        self.flux, self.flux_err, self.label, self.name = v
        self.batch_size = batch_size
        self.noise = noise
        self.n_classes = n_classes

    def __len__(self):
        return math.ceil(self.flux.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        s = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        flux = self.flux[s]
        flux_err = self.flux_err[s]
        label = self.label[s]

        if self.noise:
            v = flux + np.random.randn(*flux.shape) * flux_err
        else:
            v = flux
        x = np.arcsinh(v * 0.5)
        y = keras.utils.to_categorical(label, self.n_classes)

        return x, y

    def on_epoch_end(self):
        v = sklearn.utils.shuffle(
            self.flux, self.flux_err, self.label, self.name
        )
        self.flux, self.flux_err, self.label, self.name = v


def load_dataset(file_path, noise, batch_size):
    ds = xr.open_dataset(file_path)

    flux = ds.flux.data.astype(np.float32)
    flux_err = ds.flux_err.data.astype(np.float32)
    label = ds.label.data.astype(np.int32)
    n_classes = len(np.unique(label))

    name = [name.decode() for name in ds.sample.data]

    dataset = Dataset(
        flux=flux, flux_err=flux_err, label=label, batch_size=batch_size,
        noise=noise, n_classes=n_classes, name=name
    )
    return dataset


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--train-data', type=click.Path(exists=True))
@click.option('--test-data', type=click.Path(exists=True))
@click.option('--output-dir', type=click.Path())
@click.option('--epochs', type=int, default=100)
@click.option('--batch-size', type=int, default=1000)
def train(train_data, test_data, output_dir, epochs, batch_size):
    train_dataset = load_dataset(
        file_path=train_data, noise=True, batch_size=batch_size
    )
    test_dataset = load_dataset(
        file_path=test_data, noise=False, batch_size=batch_size
    )
    model = make_model(input_shape=(train_dataset.flux.shape[1],),
                       n_classes=len(np.unique(train_dataset.label)))
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.categorical_crossentropy,
        metrics=[keras.metrics.categorical_accuracy]
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    epoch_path = os.path.join(output_dir, 'epoch.json')
    if os.path.exists(epoch_path):
        with open(epoch_path) as f:
            count = json.load(f)
    else:
        count = 0

    model.fit_generator(
        generator=train_dataset, epochs=epochs, validation_data=test_dataset,
        callbacks=[
            keras.callbacks.TensorBoard(log_dir=output_dir, write_graph=False),
            keras.callbacks.CSVLogger(
                os.path.join(output_dir, 'log{}.csv'.format(epochs))
            )
        ],
        initial_epoch=count
    )

    count += epochs
    with open(epoch_path, 'w') as f:
        json.dump(count, f)

    model.save(filepath=os.path.join(output_dir, 'model.h5'))

    for key, dataset in (('train', train_dataset), ('test', test_dataset)):
        name = []
        label = []
        prediction = []
        for i in range(len(dataset)):
            x, y = dataset[i]
            p = model.predict_on_batch(x)
            prediction.append(p)
            label.append(np.argmax(y, axis=-1))
            name.extend(dataset.name[i * batch_size:(i + 1) * batch_size])
        prediction = np.vstack(prediction)
        label = np.hstack(label)

        df = pd.DataFrame(
            data=prediction, index=name,
            columns=['Ia', 'Ib', 'Ic', 'IIL', 'IIN', 'IIP']
        )
        df['label'] = label

        df.sort_index(inplace=True)

        df.to_csv(os.path.join(output_dir, '{}.csv'.format(key)))


def main():
    cmd()


if __name__ == '__main__':
    main()
