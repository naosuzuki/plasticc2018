#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
https://www.kaggle.com/meaninglesslives/simple-neural-net-for-time-series-classification
かなり雑な方法だけど5-fold cvでLog Lossが 0.96671
"""
import json
import warnings
from collections import Counter
from pathlib import Path

import click
import numpy as np
import pandas as pd
import sklearn.utils
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ProgbarLogger
from keras.layers import Dense, BatchNormalization, Dropout
from keras.metrics import top_k_categorical_accuracy
from keras.models import Sequential
from keras.utils import to_categorical, Sequence
from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data.cosmology import Cosmology

try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import matplotlib.pyplot as plt
    import seaborn as sns

__author__ = 'Yasuhiro Imoto'
__date__ = '10/12/2018'


class DatasetStatic(Sequence):
    def __init__(self, df, meta, index, standard_scaler, batch_size,
                 is_training, n_jobs, class_map, shuffle_only,
                 use_flux, use_magnitude, use_soft_label):
        self.meta = meta.iloc[index].copy()
        flag = np.zeros(len(df), dtype=np.bool)
        for i in self.meta['object_id']:
            flag = np.logical_or(flag, df['object_id'] == i)
        self.df = df[flag].copy()

        self.ss = standard_scaler
        self.is_training = is_training
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.class_map = class_map

        self.epoch = 0

        self.x, self.y, self.object_id = make_feature(
            train=self.df.copy(), meta_train=self.meta, n_jobs=n_jobs,
            is_training=is_training, class_map=class_map,
            use_flux=use_flux, use_magnitude=use_magnitude,
            use_soft_label=use_soft_label
        )
        self.x = self.x.astype(np.float64)
        if is_training or shuffle_only:
            self.x = standard_scaler.fit_transform(self.x)
        else:
            # 訓練データより後に呼び出す必要がある
            self.x = standard_scaler.transform(self.x)
        self.shuffle_only = shuffle_only

        self.use_flux = use_flux
        self.use_magnitude = use_magnitude
        self.use_soft_label = use_soft_label

    def __len__(self):
        return (len(self.y) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        s = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_x = self.x[s]
        batch_y = self.y[s]

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.is_training:
            self.epoch += 1

            if self.epoch % 25 == 0:
                x, y, object_id = make_feature(
                    train=self.df.copy(), meta_train=self.meta,
                    n_jobs=self.n_jobs, is_training=self.is_training,
                    class_map=self.class_map,
                    use_flux=self.use_flux, use_magnitude=self.use_magnitude,
                    use_soft_label=self.use_soft_label
                )

                x = x.astype(np.float64)
                x = self.ss.transform(x)

                self.x, self.y, self.object_id = sklearn.utils.shuffle(
                    x, y, object_id, random_state=self.epoch
                )
        elif self.shuffle_only:
            self.x, self.y, self.object_id = sklearn.utils.shuffle(
                self.x, self.y, self.object_id, random_state=self.epoch
            )


class DatasetAstronomy(Sequence):
    def __init__(self, df, target, object_id, index, standard_scaler,
                 batch_size, is_training, class_map):
        self.df = df.iloc[index]
        self.y = target[index]
        self.object_id = object_id[index]

        self.ss = standard_scaler
        self.is_training = is_training
        self.batch_size = batch_size

        self.class_map = class_map

        if is_training:
            self.x = standard_scaler.fit_transform(self.df)
        else:
            self.x = standard_scaler.transform(self.df)

    def __len__(self):
        return (len(self.y) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        s = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_x = self.x[s]
        batch_y = self.y[s]

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.is_training:
            self.x, self.y, self.object_id = sklearn.utils.shuffle(
                self.x, self.y, self.object_id
            )


class DatasetCombined(Sequence):
    def __init__(self, df, target, object_id, index, standard_scaler,
                 batch_size, is_training, class_map):
        self.x = df.iloc[index]
        self.y = target[index]
        self.object_id = object_id[index]

        self.ss = standard_scaler
        self.is_training = is_training
        self.batch_size = batch_size

        self.class_map = class_map

        if is_training:
            self.x = standard_scaler.fit_transform(self.x)
        else:
            self.x = standard_scaler.transform(self.x)

    def __len__(self):
        return (len(self.y) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        s = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_x = self.x[s]
        batch_y = self.y[s]

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.is_training:
            self.x, self.y, self.object_id = sklearn.utils.shuffle(
                self.x, self.y, self.object_id
            )


class DatasetTest(Sequence):
    def __init__(self, df, meta, standard_scaler, batch_size, n_jobs,
                 use_flux, use_magnitude):
        self.x, self.y, self.object_id = make_feature(
            train=df, meta_train=meta, n_jobs=n_jobs,
            is_training=False, class_map=None,
            use_flux=use_flux, use_magnitude=use_magnitude,
            use_soft_label=False
        )
        self.x = self.x.astype(np.float64)
        self.x = standard_scaler.transform(self.x)

        self.batch_size = batch_size

        self.use_flux = use_flux
        self.use_magnitude = use_magnitude

    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        s = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_x = self.x[s]
        batch_y = None

        return batch_x, batch_y


def compute_factor(specz, photoz, photoz_err, object_id, is_training):
    lcdm = Cosmology()

    if is_training:
        # ときどきphotozの方を使う
        if specz > 0 and np.random.rand() < 0.1:
            z = specz
        else:
            z = photoz + photoz_err * np.random.randn()
    else:
        if specz > 0:
            z = specz
        else:
            z = photoz
    z = max(z, 0.01)

    dist_mod = lcdm.DistMod(z)
    z_norm = 0.5
    dist_mod_norm = lcdm.DistMod(z_norm)
    factor = 10.0 ** (0.4 * (dist_mod - dist_mod_norm))

    return object_id, factor


def normalize_flux(train, meta_train, n_jobs, is_training):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        factors = Parallel(n_jobs=n_jobs)(
            delayed(compute_factor)(specz, photoz, photoz_err, object_id,
                                    is_training)
            for specz, photoz, photoz_err, object_id in zip(
                meta_train['hostgal_specz'], meta_train['hostgal_photoz'],
                meta_train['hostgal_photoz_err'], meta_train['object_id']
            )
        )
    factor_array = np.empty(len(train))
    for object_id, factor in factors:
        flag = train['object_id'] == object_id
        factor_array[flag] = factor
    train['flux'] *= factor_array
    train['flux_err'] *= factor_array

    return train, meta_train


def make_feature_whole(train, use_flux, use_magnitude):
    """
    バンドの違いを気にせず全体を処理
    :param use_flux:
    :param use_magnitude:
    :param train:
    :return:
    """
    aggregators = {}
    if use_flux:
        aggregators.update({
            'flux': ['max', 'mean', 'median', 'std', 'skew'],
            'flux_ratio_sq': ['sum', 'skew'],
            'flux_by_flux_ratio_sq': ['sum', 'skew']
        })
    if use_magnitude:
        aggregators.update({
            'magnitude': ['max', 'mean', 'median', 'std', 'skew']
        })

    agg_train = train.groupby('object_id').agg(aggregators)
    new_columns = [
        k + '_' + agg for k in aggregators.keys() for agg in aggregators[k]
    ]
    agg_train.columns = new_columns

    # agg_train['mjd_diff'] = agg_train['mjd_max'] - agg_train['mjd_min']

    # agg_train['flux_dif1'] = agg_train['flux_max'] - agg_train['flux_min']
    # agg_train['flux_dif2'] = agg_train['flux_dif1'] / agg_train['flux_mean']
    if use_flux:
        agg_train['flux_w_mean'] = (
                agg_train['flux_by_flux_ratio_sq_sum'] /
                agg_train['flux_ratio_sq_sum']
        )
    # agg_train['flux_dif3'] = (
    #         agg_train['flux_dif1'] / agg_train['flux_w_mean']
    # )

    # agg_train['magnitude_dif1'] = (
    #         agg_train['magnitude_max'] - agg_train['magnitude_min']
    # )
    # agg_train['magnitude_dif2'] = (
    #         agg_train['magnitude_dif1'] / agg_train['magnitude_mean']
    # )

    # del agg_train['mjd_max'], agg_train['mjd_min']

    return agg_train


def make_feature_band(train, use_flux, use_magnitude):
    """
    バンドごとに処理
    :param use_magnitude:
    :param use_flux:
    :param train:
    :return:
    """
    aggregators = {}
    if use_flux:
        aggregators.update({
            'flux': ['max', 'mean', 'median', 'std', 'skew'],
            'flux_ratio_sq': ['sum', 'skew'],
            'flux_by_flux_ratio_sq': ['sum', 'skew']
        })
    if use_magnitude:
        aggregators.update({
            'magnitude': ['max', 'mean', 'median', 'std', 'skew']
        })

    agg_train = train.groupby(['object_id', 'passband']).agg(aggregators)
    new_columns = [
        k + '_' + agg for k in aggregators.keys() for agg in aggregators[k]
    ]
    agg_train.columns = new_columns

    # バンドの番号が末尾につくので、
    # agg_train['flux_dif1'] = agg_train['flux_max'] - agg_train['flux_min']
    # agg_train['flux_dif2'] = agg_train['flux_dif1'] / agg_train['flux_mean']
    if use_flux:
        agg_train['flux_w_mean'] = (
                agg_train['flux_by_flux_ratio_sq_sum'] /
                agg_train['flux_ratio_sq_sum']
        )
    # agg_train['flux_dif3'] = (
    #         agg_train['flux_dif1'] / agg_train['flux_w_mean']
    # )

    # agg_train['magnitude_dif1'] = (
    #         agg_train['magnitude_max'] - agg_train['magnitude_min']
    # )
    # agg_train['magnitude_dif2'] = (
    #         agg_train['magnitude_dif1'] / agg_train['magnitude_mean']
    # )

    # バンドについて縦に並んでいるのを横に並べ替え
    agg_train = agg_train.unstack()

    # MultiIndexになっているので、フラットなものに変更
    c = agg_train.columns
    # noinspection PyUnresolvedReferences
    new_columns = [
        '{}_{}'.format(c.levels[0][i], c.levels[1][j])
        for i, j in zip(*c.labels)
    ]
    agg_train.columns = new_columns

    return agg_train


def make_feature(train, meta_train, n_jobs, is_training, class_map,
                 use_flux, use_magnitude, use_soft_label):
    # redshiftが0.5の距離に揃える
    train, meta_train = normalize_flux(
        train=train, meta_train=meta_train, n_jobs=n_jobs,
        is_training=is_training
    )

    if use_flux:
        train['flux_ratio_sq'] = np.power(
            train['flux'] / train['flux_err'], 2.0
        )
        train['flux_by_flux_ratio_sq'] = train['flux'] * train['flux_ratio_sq']
    if use_magnitude:
        train['magnitude'] = np.arcsinh(train['flux'] * 0.5)

    agg_train = make_feature_whole(
        train=train, use_flux=use_flux, use_magnitude=use_magnitude
    )
    agg_train_band = make_feature_band(
        train=train, use_flux=use_flux, use_magnitude=use_magnitude
    )

    full_train = agg_train.reset_index().merge(
        right=meta_train,
        how='outer',
        on='object_id'
    )
    full_train = agg_train_band.reset_index().merge(
        right=full_train,
        how='outer',
        on='object_id'
    )
    # full_train = agg_train_band.reset_index().merge(
    #     right=meta_train, how='outer', on='object_id'
    # )

    if 'target' in full_train:
        y = full_train['target']
        del full_train['target']

        y_map = np.array([class_map[v] for v in y])
        if use_soft_label:
            soft_labels = np.array([[0.96, 0.01, 0.01, 0.01, 0.01],
                                    [0.01, 0.96, 0.01, 0.01, 0.01],
                                    [0.01, 0.01, 0.96, 0.01, 0.01],
                                    [0.01, 0.01, 0.01, 0.96, 0.01],
                                    [0.01, 0.01, 0.01, 0.01, 0.96]])
            soft_labels = soft_labels / np.sum(soft_labels,
                                               axis=1, keepdims=True)
            y = soft_labels[y_map]
        else:
            y = to_categorical(y_map, num_classes=len(class_map))
    else:
        y = None

    oof_df = full_train[['object_id']]
    if 'object_id' in full_train:
        del full_train['object_id'], full_train['distmod']
        del full_train['ra'], full_train['decl'], full_train['ddf']
        del full_train['gal_l'], full_train['gal_b']
        del full_train['hostgal_specz'], full_train['mwebv']
        del full_train['hostgal_photoz'], full_train['hostgal_photoz_err']

    train_mean = full_train.mean(axis=0)
    full_train.fillna(train_mean, inplace=True)

    # カラムの順序が一定ではないようなので、sort
    full_train.sort_index(axis=1, inplace=True)

    return full_train, y, oof_df


def make_training_data(data_dir, only_sn, only_extra, only_intra):
    # データ数が少ない順
    if only_sn:
        classes = (95, 52, 62, 42, 90)
    elif only_extra:
        classes = (95, 52, 67, 62, 15, 42, 90)

        only_extra = False
        only_sn = True
    else:
        classes = (53, 64, 6, 95, 52, 67, 92, 88, 62, 15, 16, 65, 42, 90)

    data_dir = Path(data_dir)
    if 'raw' in str(data_dir):
        train = pd.read_csv(data_dir / 'training_set.csv')
    else:
        train = pd.read_csv(
            data_dir / 'training_set.csv',
            names=['object_id', 'mjd', 'passband', 'flux', 'flux_err',
                   'detected', 'interpolation']
        )
    meta_train = pd.read_csv(data_dir / 'training_set_metadata.csv')

    if only_sn:
        # 対象のクラスのみに減らす
        flag = np.zeros(len(meta_train), dtype=np.bool)
        for c in classes:
            flag = np.logical_or(flag, meta_train['target'] == c)
        meta_train = meta_train.loc[flag]

        flag = np.zeros(len(train), dtype=np.bool)
        for i in meta_train['object_id']:
            flag = np.logical_or(flag, train['object_id'] == i)
        train = train[flag]
    elif only_extra:
        flag = meta_train['hostgal_photoz'] > 0
        meta_train = meta_train[flag]

        target_list = meta_train['target']
        unique_target, count = np.unique(target_list, return_counts=True)
        order = np.argsort(count)
        classes = tuple(unique_target[order])

        flag = np.zeros(len(train), dtype=np.bool)
        for i in meta_train['object_id']:
            flag = np.logical_or(flag, train['object_id'] == i)
        train = train[flag]
    elif only_intra:
        pass

    if only_extra or only_sn:
        train = train[train['flux'] > 0]

    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    class_weight = {
        c: 1 if c not in (64, 15) else 2 for c in classes
    }
    # for c in [64, 15]:
    #     class_weight[c] = 2

    print('Unique classes : ', classes)

    return train, meta_train, classes, class_weight


def make_test_data(data_dir, only_extra, only_intra):
    data_dir = Path(data_dir)
    test = pd.read_csv(data_dir / 'test_set.csv')
    meta_test = pd.read_csv(data_dir / 'test_set_metadata.csv')

    if only_extra:
        flag = meta_test['hostgal_photoz'] > 0
        meta_test = meta_test[flag]

        id_set = set(meta_test['object_id'].values)
        flag = np.array([i in id_set for i in test['object_id'].values],
                        dtype=np.bool)
        test = test[flag]

        test = test[test['flux'] > 0]
    elif only_intra:
        flag = meta_test['hostgal_photoz'] == 0
        meta_test = meta_test[flag]

        id_set = set(meta_test['object_id'].values)
        flag = np.array([i in id_set for i in test['object_id'].values],
                        dtype=np.bool)
        test = test[flag]

    return test, meta_test


def multi_weighted_log_loss(y_ohe, y_p, class_weight):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi log loss for PLAsTiCC challenge
    """
    # classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    # 15, 64は2、それ以外は1
    # class_weight = {
    #     6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1,
    #     88: 1, 90: 1, 92: 1, 95: 1
    # }
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([
        class_weight[k] for k in sorted(class_weight.keys())
    ])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


def build_model(input_size, classes, hidden_size, dropout_rate=0.25,
                activation='relu'):
    K.clear_session()

    start_neurons = hidden_size
    # create model
    model = Sequential()
    model.add(Dense(start_neurons, input_dim=input_size,
                    activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 2, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 4, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 8, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate / 2))

    model.add(Dense(len(classes), activation='softmax'))
    return model


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true=y_true, y_pred=y_pred, k=2)


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true=y_true, y_pred=y_pred, k=3)


def top_4_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true=y_true, y_pred=y_pred, k=4)


def plot_loss_acc(history, output_path):
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex='all')

    start = 50
    epoch = history.epoch[start:]

    ax0.plot(epoch, history.history['loss'][start:], label='train')
    ax0.plot(epoch, history.history['val_loss'][start:], label='validation')
    ax0.grid()
    ax0.set_ylabel('loss')
    ax0.set_xlabel('epoch')
    ax0.legend(loc='upper left')
    ax0.set_title('model loss')

    ax1.plot(epoch, history.history['acc'][start:], label='train')
    ax1.plot(epoch, history.history['val_acc'][start:], label='validation')
    ax1.grid()
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(loc='upper left')
    ax1.set_title('model accuracy')

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches='tight')
    plt.close()


def plot_top_accuracy(history, output_path):
    fig, ax = plt.subplots()

    palette = sns.color_palette()
    ax.plot(history.epoch, history.history['acc'], label='train 1', ls=':',
            color=palette[0])
    ax.plot(history.epoch, history.history['val_acc'], label='validation 1',
            color=palette[0])
    for i in range(2, 5):
        ax.plot(history.epoch, history.history['top_{}_accuracy'.format(i)],
                label='train {}'.format(i), ls=':', color=palette[i - 1])
        ax.plot(
            history.epoch, history.history['val_top_{}_accuracy'.format(i)],
            label='validation {}'.format(i), color=palette[i - 1]
        )
    ax.grid()
    ax.legend(loc='best', ncol=2)

    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, classes, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(12, 12))
    annotation = np.around(cm, 2)

    sns.heatmap(cm, xticklabels=classes, yticklabels=classes, cmap='Blues',
                annot=annotation, lw=0.5, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_aspect('equal')
    fig.tight_layout()

    return fig


# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
def weighted_log_loss(y_true, y_pred, weight_table):
    yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
    loss = -tf.reduce_mean(
        tf.reduce_mean(y_true * tf.log(yc), axis=0) / weight_table
    )
    return loss


@click.group()
def cmd():
    pass


class BaseData(object):
    def __init__(self, classes, class_weight):
        self.classes = classes
        self.class_weight = class_weight

        self.class_map = {c: i for i, c in enumerate(classes)}


class StaticFeatureData(BaseData):
    def __init__(self, data_dir, only_sn, only_extra, only_intra):
        train, meta_train, classes, class_weight = make_training_data(
            only_sn=only_sn, only_extra=only_extra, only_intra=only_intra,
            data_dir=data_dir
        )
        ss = StandardScaler()

        super().__init__(classes=classes, class_weight=class_weight)

        self.train = train
        self.meta_train = meta_train

        self.ss = ss

        self.y_map = np.array([
            self.class_map[t] for t in meta_train['target']
        ])

    def __len__(self):
        return len(self.meta_train)

    def make_dataset(self, index, batch_size, is_training, n_jobs,
                     shuffle_only, use_flux, use_magnitude, use_soft_label):
        dataset = DatasetStatic(
            df=self.train, meta=self.meta_train, index=index,
            standard_scaler=self.ss, batch_size=batch_size,
            is_training=is_training, n_jobs=n_jobs, class_map=self.class_map,
            shuffle_only=shuffle_only,
            use_flux=use_flux, use_magnitude=use_magnitude,
            use_soft_label=use_soft_label
        )
        return dataset

    @property
    def object_id(self):
        return self.meta_train['object_id']


class AstronomyFeatureData(BaseData):
    def __init__(self, data_dir, feature_size):
        classes = (95, 52, 62, 42, 90)
        class_weight = {
            c: 1 if c not in (64, 15) else 2 for c in classes
        }
        super().__init__(classes=classes, class_weight=class_weight)

        data_dir = Path(data_dir)
        name = ('training_set_bin1_sn_180days_gpr'
                '_feature{}_181214_1.csv').format(feature_size)
        train = pd.read_csv(data_dir / name, header=0, index_col=0)

        meta_train = pd.read_csv(
            data_dir / 'training_set_metadata.csv', header=0, index_col=0
        )
        y = meta_train.loc[train.index, 'target']

        ss = StandardScaler()

        self.train = train
        self.meta_train = meta_train
        self.object_id = train.index.values

        self.ss = ss

        self.y_map = np.array([self.class_map[t] for t in y])
        self.y = to_categorical(self.y_map, len(self.classes))

    def __len__(self):
        return len(self.y)

    def make_dataset(self, index, batch_size, is_training):
        dataset = DatasetAstronomy(
            df=self.train, target=self.y, index=index,
            standard_scaler=self.ss, batch_size=batch_size,
            is_training=is_training, class_map=self.class_map,
            object_id=self.object_id
        )
        return dataset


class AstronomyFeatureData2(BaseData):
    def __init__(self, data_dir, dataset_index):
        classes = (95, 52, 67, 62, 15, 42, 90)
        class_weight = {
            c: 1 if c not in (64, 15) else 2 for c in classes
        }
        super().__init__(classes=classes, class_weight=class_weight)

        data_dir = Path(data_dir)
        name = 'training_set_{}_181217_2.csv'.format(dataset_index)
        # インデックスはobject_id
        train = pd.read_csv(data_dir / name, header=0, index_col=0)

        meta_train = pd.read_csv(
            data_dir / 'training_set_metadata.csv', header=0, index_col=0
        )
        y = meta_train.loc[train.index, 'target']

        ss = StandardScaler()

        self.train = train
        self.meta_train = meta_train
        self.object_id = train.index.values

        self.ss = ss

        self.y_map = np.array([self.class_map[t] for t in y])
        self.y = to_categorical(self.y_map, len(self.classes))

    def __len__(self):
        return len(self.y)

    def make_dataset(self, index, batch_size, is_training):
        dataset = DatasetAstronomy(
            df=self.train, target=self.y, index=index,
            standard_scaler=self.ss, batch_size=batch_size,
            is_training=is_training, class_map=self.class_map,
            object_id=self.object_id
        )
        return dataset


class CombinedFeatureData(BaseData):
    def __init__(self, static_feature_path, astronomy_feature_path):
        classes = (95, 52, 67, 62, 15, 42, 90)
        class_weight = {
            c: 1 if c not in (64, 15) else 2 for c in classes
        }
        super().__init__(classes=classes, class_weight=class_weight)

        static_df = pd.read_csv(static_feature_path, header=0)
        astronomy_df = pd.read_csv(
            astronomy_feature_path, header=0, index_col=0
        )

        meta = pd.read_csv(
            Path(static_feature_path).with_name('training_set_metadata.csv'),
            header=0, index_col=0
        )
        meta = meta.loc[astronomy_df.index]

        tmp = meta.reset_index()
        static_df = self.convert_static_features(df=static_df, meta=tmp)

        # indexで二つの特徴を繋ぎ合わせる
        x = astronomy_df.join(static_df)
        y = meta.loc[x.index, 'target']

        ss = StandardScaler()

        self.x = x
        self.y_map = np.array([self.class_map[t] for t in y])
        self.y = to_categorical(self.y_map, len(self.classes))
        self.object_id = x.index.values

        self.ss = ss

    def __len__(self):
        return len(self.y)

    def make_dataset(self, index, batch_size, is_training):
        dataset = DatasetCombined(
            df=self.x, target=self.y, object_id=self.object_id, index=index,
            standard_scaler=self.ss, batch_size=batch_size,
            is_training=is_training, class_map=self.class_map
        )
        return dataset

    @staticmethod
    def convert_static_features(df, meta):
        # 不要な行を取り除く
        flag = np.zeros(len(df), dtype=np.bool)
        for i in meta['object_id']:
            flag = np.logical_or(flag, df['object_id'] == i)
        df = df[flag]

        # 特徴量に変換する
        df, meta = normalize_flux(
            train=df, meta_train=meta, n_jobs=1, is_training=False
        )

        df['magnitude'] = np.arcsinh(df['flux'] * 0.5)

        agg_train = make_feature_whole(
            train=df, use_flux=False, use_magnitude=True
        )
        agg_train_band = make_feature_band(
            train=df, use_flux=False, use_magnitude=True
        )

        full_train = agg_train.join(agg_train_band)

        train_mean = full_train.mean(axis=0)
        full_train.fillna(train_mean, inplace=True)

        # カラムの順序が一定ではないようなので、sort
        full_train.sort_index(axis=1, inplace=True)

        return full_train


def load_dataset(data_dir, astronomy_feature, static_feature,
                 only_sn, only_extra, only_intra, feature_size, dataset_index,
                 data_dir2):
    if data_dir2 is None:
        assert static_feature != astronomy_feature

        if astronomy_feature:
            if dataset_index is not None:
                data = AstronomyFeatureData2(
                    data_dir=data_dir, dataset_index=dataset_index
                )
            else:
                data = AstronomyFeatureData(
                    data_dir=data_dir, feature_size=feature_size
                )
        else:
            data = StaticFeatureData(
                data_dir=data_dir, only_sn=only_sn, only_extra=only_extra,
                only_intra=only_intra
            )
    else:
        data_dir2 = Path(data_dir2)
        name = 'training_set_{}_181217_1.csv'.format(dataset_index)

        data = CombinedFeatureData(
            static_feature_path=data_dir2 / 'training_set.csv',
            astronomy_feature_path=data_dir / name
        )
    return data


@cmd.command()
@click.option('--only-sn', is_flag=True)
@click.option('--only-extra', is_flag=True)
@click.option('--only-intra', is_flag=True)
@click.option('--model-dir', type=click.Path())
@click.option('--data-dir', type=click.Path(exists=True),
              default=r'C:/Users/imoto/Documents/plasticc/data/raw')
@click.option('--n-jobs', type=int, default=1)
@click.option('--hidden-size', type=int, default=512)
@click.option('--static-feature', is_flag=True)
@click.option('--astronomy-feature', is_flag=True)
@click.option('--shuffle-only', is_flag=True)
@click.option('--epochs', type=int, default=200)
@click.option('--use-flux', is_flag=True)
@click.option('--use-magnitude', is_flag=True)
@click.option('--use-soft-label', is_flag=True)
@click.option('--cv-index', type=int)
@click.option('--feature-size', type=int)
@click.option('--dataset-index', type=int)
@click.option('--data-dir2', type=click.Path())
def learn(only_sn, only_extra, only_intra, model_dir, data_dir, n_jobs,
          hidden_size, static_feature, astronomy_feature, shuffle_only, epochs,
          use_flux, use_magnitude, use_soft_label,
          cv_index, feature_size, dataset_index, data_dir2):
    # assert use_flux or use_magnitude

    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    data_dir = Path(data_dir)

    parameters = {
        'data': str(data_dir), 'data2': data_dir2,
        'hidden_size': hidden_size, 'static_feature': static_feature,
        'only_sn': only_sn, 'only_extra': only_extra, 'only_intra': only_intra,
        'shuffle_only': shuffle_only, 'use_flux': use_flux,
        'use_magnitude': use_magnitude, 'astronomy_feature': astronomy_feature,
        'use_soft_label': use_soft_label, 'feature_size:': feature_size,
        'dataset_index': dataset_index
    }
    with (model_dir / 'parameters.json').open('w') as f:
        json.dump(parameters, f, sort_keys=True, indent=4)

    data = load_dataset(
        data_dir=data_dir,
        only_sn=only_sn, only_extra=only_extra, only_intra=only_intra,
        astronomy_feature=astronomy_feature, static_feature=static_feature,
        feature_size=feature_size, dataset_index=dataset_index,
        data_dir2=data_dir2
    )

    y_map = data.y_map
    y_categorical = to_categorical(y_map)

    classes = data.classes
    y_count = Counter(y_map)
    weight_table = np.zeros_like(classes, dtype=np.float32)
    for i in range(len(classes)):
        weight_table[i] = y_count[i] / y_map.shape[0]

    oof_predictions = np.zeros((len(data), len(classes)))
    oof_index = None
    batch_size = 100
    loss_list = []

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):
        if cv_index is not None and cv_index != fold_:
            continue
        oof_index = val_

        model_path = str(model_dir / "keras{}.model".format(fold_))
        check_point = ModelCheckpoint(
            model_path,
            monitor='val_loss', mode='min', save_best_only=True, verbose=0
        )
        # progress_bar = ProgbarLogger(count_mode='steps')

        if astronomy_feature:
            dataset_train = data.make_dataset(
                index=trn_, batch_size=batch_size, is_training=True
            )
            dataset_valid = data.make_dataset(
                index=val_, batch_size=batch_size, is_training=False
            )
        else:
            dataset_train = data.make_dataset(
                index=trn_, batch_size=batch_size,
                is_training=not shuffle_only, n_jobs=n_jobs,
                shuffle_only=shuffle_only,
                use_flux=use_flux, use_magnitude=use_magnitude,
                use_soft_label=use_soft_label
            )
            dataset_valid = data.make_dataset(
                index=val_, batch_size=batch_size,
                is_training=False, n_jobs=n_jobs, shuffle_only=False,
                use_flux=use_flux, use_magnitude=use_magnitude,
                use_soft_label=use_soft_label
            )

        d = model_dir / str(fold_)
        if not d.exists():
            d.mkdir(parents=True)
        np.savez_compressed(
            str(d / 'train.pickle'), x=dataset_train.x, y=dataset_train.y,
            object_id=dataset_train.object_id
        )
        np.savez_compressed(
            str(d / 'validation.pickle'), x=dataset_valid.x,
            y=dataset_valid.y, object_id=dataset_valid.object_id
        )
        joblib.dump(data.ss, str(d / 'standard_scaler.pickle'))

        label = list(np.argmax(dataset_train.y, axis=-1))

        train_size = len(dataset_train.y)
        input_size = dataset_train.x.shape[1]
        print('input_size: {}'.format(input_size))

        train_y_count = Counter(label)
        train_weight_table = np.zeros_like(classes, dtype=np.float32)
        for i in range(len(classes)):
            train_weight_table[i] = train_y_count[i] / train_size

        model = build_model(
            input_size, classes, hidden_size=hidden_size,
            dropout_rate=0.5, activation='tanh'
        )
        model.compile(
            loss=lambda t, p: weighted_log_loss(
                y_true=t, y_pred=p, weight_table=train_weight_table
            ),
            optimizer='adam', metrics=['accuracy', top_2_accuracy,
                                       top_3_accuracy, top_4_accuracy]
        )
        history = model.fit_generator(
            generator=dataset_train, validation_data=dataset_valid,
            epochs=epochs, shuffle=False, verbose=1, workers=n_jobs,
            callbacks=[check_point]
        )

        plot_loss_acc(history, model_dir / 'result{}.png'.format(fold_))
        plot_top_accuracy(history, model_dir / 'accuracy{}.png'.format(fold_))

        print('Loading Best Model')
        model.load_weights(model_path)
        # Get predicted probabilities for each class
        if static_feature:
            predictions = model.predict_proba(
                dataset_valid.x, batch_size=batch_size
            )
        else:
            predictions = model.predict_generator(dataset_valid)
        oof_predictions[val_, :] = predictions
        log_loss = multi_weighted_log_loss(
            dataset_valid.y, predictions, class_weight=data.class_weight
        )
        loss_list.append(float(log_loss))
        print(log_loss)
        with (model_dir / 'log_loss{}.txt'.format(fold_)).open('w') as f:
            f.write(str(log_loss))

    with (model_dir / 'loss.json').open('w') as f:
        json.dump(loss_list, f)

    if cv_index is not None:
        oof_predictions = oof_predictions[oof_index]
        y_categorical = y_categorical[oof_index]
        np.savez_compressed(
            str(model_dir / 'predictions{}.npz'.format(cv_index)),
            prediction=oof_predictions, target=y_categorical
        )

        misc_path = model_dir / 'misc.pickle'
        if not misc_path.exists():
            d = dict(class_weight=data.class_weight,
                     y_map=y_map, classes=classes)
            joblib.dump(d, str(misc_path))
    else:
        print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_log_loss(
            y_categorical, oof_predictions, class_weight=data.class_weight))

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(
            y_map, np.argmax(oof_predictions, axis=-1)
        )
        np.savetxt(str(model_dir / 'confusion_matrix.txt'), cnf_matrix)

        fig = plot_confusion_matrix(cnf_matrix, classes=classes,
                                    normalize=True)
        fig.savefig(str(model_dir / 'confusion_matrix.png'),
                    bbox_inches='tight')
        plt.close()


@cmd.command()
@click.option('--model-dir', type=click.Path(exists=True))
def draw_confusion_matrix(model_dir):
    model_dir = Path(model_dir)

    predictions = []
    targets = []
    for i in range(5):
        tmp = np.load(str(model_dir / 'predictions{}.npz'.format(i)))
        predictions.append(tmp['prediction'])
        targets.append(tmp['target'])
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    misc_path = model_dir / 'misc.pickle'
    tmp = joblib.load(str(misc_path))
    class_weight = tmp['class_weight']
    y_map = tmp['y_map']
    classes = tmp['classes']

    print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_log_loss(
        targets, predictions, class_weight=class_weight))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_map, np.argmax(predictions, axis=-1))
    np.savetxt(str(model_dir / 'confusion_matrix.txt'), cnf_matrix)

    fig = plot_confusion_matrix(cnf_matrix, classes=classes,
                                normalize=True)
    fig.savefig(str(model_dir / 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()


@cmd.command()
@click.option('--model-dir', type=click.Path(exists=True))
@click.option('--n-jobs', type=int, default=1)
def predict_training(model_dir, n_jobs):
    model_dir = Path(model_dir)
    with (model_dir / 'parameters.json').open('r') as f:
        parameters = json.load(f)
    static_feature = parameters['static_feature']
    astronomy_feature = parameters.get('astronomy_feature', False)
    only_sn = parameters['only_sn']
    only_extra = parameters.get('only_extra', False)
    only_intra = parameters.get('only_intra', False)

    data_dir = Path(parameters['data'])
    data_dir2 = parameters.get('data2', None)
    dataset_index = parameters.get('dataset_index', None)

    feature_size = parameters.get('feature_size', 5)
    data = load_dataset(
        data_dir=data_dir,
        only_sn=only_sn, only_extra=only_extra, only_intra=only_intra,
        static_feature=static_feature, astronomy_feature=astronomy_feature,
        feature_size=feature_size,
        data_dir2=data_dir2, dataset_index=dataset_index
    )

    classes = data.classes
    y_map = data.y_map
    oof_predictions = np.zeros((len(data), len(classes)))

    hidden_size = parameters['hidden_size']

    use_flux = parameters['use_flux']
    use_magnitude = parameters['use_magnitude']

    batch_size = 1000
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):
        model_path = str(model_dir / "keras{}.model".format(fold_))

        data.ss = joblib.load(str(
            model_dir / '{}'.format(fold_) / 'standard_scaler.pickle'
        ))

        if astronomy_feature:
            dataset = data.make_dataset(
                index=val_, batch_size=batch_size, is_training=False
            )
        else:
            dataset = data.make_dataset(
                index=val_, batch_size=batch_size,
                is_training=False, n_jobs=n_jobs, shuffle_only=False,
                use_flux=use_flux, use_magnitude=use_magnitude
            )
        input_size = dataset.x.shape[1]

        print(input_size)
        model = build_model(
            input_size, classes, hidden_size=hidden_size,
            dropout_rate=0.5, activation='tanh'
        )
        model.load_weights(model_path)

        if static_feature:
            predictions = model.predict_proba(
                dataset.x, batch_size=batch_size
            )
        else:
            predictions = model.predict_generator(dataset)
        oof_predictions[val_, :] = predictions

    y_categorical = to_categorical(y_map)
    log_loss = multi_weighted_log_loss(
        y_categorical, oof_predictions, class_weight=data.class_weight
    )
    print(log_loss)

    df = pd.DataFrame(oof_predictions, index=data.object_id, columns=classes)
    df.to_csv(model_dir / 'predictions.csv')


@cmd.command()
@click.option('--model-dir', type=click.Path(exists=True))
@click.option('--data-dir', type=click.Path(exists=True),
              default=r'C:/Users/imoto/Documents/plasticc/data/raw')
@click.option('--n-jobs', type=int, default=1)
@click.option('--test-csv', type=click.Path(exists=True))
def predict_test(model_dir, data_dir, n_jobs, test_csv):
    model_dir = Path(model_dir)
    with (model_dir / 'parameters.json').open('r') as f:
        parameters = json.load(f)
    static_feature = parameters['static_feature']
    astronomy_feature = parameters.get('astronomy_feature', False)
    only_sn = parameters['only_sn']
    only_extra = parameters.get('only_extra', False)
    only_intra = parameters.get('only_intra', False)

    hidden_size = parameters['hidden_size']

    data_dir = parameters['data']
    data_dir2 = parameters.get('data2', None)
    dataset_index = parameters.get('dataset_index', None)

    use_flux = parameters['use_flux']
    use_magnitude = parameters['use_magnitude']

    # data_dir = Path(data_dir)
    # if only_extra:
    #     data_path = data_dir / 'test_extra.pickle'
    #     if data_path.exists():
    #         test = pd.read_pickle(data_path)
    #         meta_test = pd.read_pickle(data_dir / 'test_meta_extra.pickle')
    #     else:
    #         test, meta_test = make_test_data(
    #             data_dir=data_dir, only_extra=only_extra or only_sn,
    #             only_intra=only_intra
    #         )
    #
    #         test.to_pickle(data_path)
    #         meta_test.to_pickle(data_dir / 'test_meta_extra.pickle')
    # else:
    #     test, meta_test = make_test_data(
    #         data_dir=data_dir, only_extra=only_extra or only_sn,
    #         only_intra=only_intra
    #     )

    # df2 = pd.read_pickle('../data/processed/test_set_v2.pickle')
    # df1 = pd.read_csv(test_csv, index_col=0)
    # df = df1.join(df2)
    df = pd.read_csv(test_csv, index_col=0, header=0)

    input_size = df.shape[1]

    train_mean = df.mean(axis=0)
    df.fillna(train_mean, inplace=True)
    df[np.isinf(df)] = 0
    # print(df.shape)

    if only_sn:
        classes = (95, 52, 62, 42, 90)
    elif only_extra:
        classes = (95, 52, 67, 62, 15, 42, 90)
    elif only_intra:
        classes = (53, 6, 92, 16, 65)
    else:
        classes = (53, 64, 6, 95, 52, 67, 92, 88, 62, 15, 16, 65, 42, 90)
    classes = (95, 52, 67, 62, 15, 42, 90)

    static_feature = True
    batch_size = 5000
    for i in range(5):
        print(i)
        model_path = str(model_dir / "keras{}.model".format(i))
        # print(model_path)

        if static_feature or astronomy_feature:
            ss = joblib.load(str(
                model_dir / '{}'.format(i) / 'standard_scaler.pickle'
            ))
            # print(ss.mean_)

            # if astronomy_feature:
            #     dataset = None
            # else:
            #     dataset = DatasetTest(
            #         df=test, meta=meta_test, standard_scaler=ss,
            #         batch_size=batch_size, n_jobs=n_jobs, use_flux=use_flux,
            #         use_magnitude=use_magnitude
            #     )
            # input_size = dataset.x.shape[1]
            # input_size = df.shape[1]
            #
            # train_mean = df.mean(axis=0)
            # df.fillna(train_mean, inplace=True)
            # df[np.isinf(df)] = 0

            x = ss.transform(df.values)
        else:
            dataset = None
            input_size = 0

        # print(input_size)
        model = build_model(
            input_size, classes, hidden_size=hidden_size,
            dropout_rate=0.5, activation='tanh'
        )
        model.load_weights(model_path)

        # predictions = model.predict_generator(dataset)
        predictions = model.predict(x, batch_size=batch_size)

        # df = pd.DataFrame(data=predictions, index=meta_test['object_id'],
        #                   columns=['class_{}'.format(c) for c in classes])
        output = pd.DataFrame(data=predictions, index=df.index,
                              columns=['class_{}'.format(c) for c in classes])
        output.sort_index(axis=1, inplace=True)
        output.to_csv(model_dir / 'test_prediction{}.csv'.format(i))
        output.to_pickle(model_dir / 'test_prediction{}.pickle'.format(i))


@cmd.command()
@click.option('--only-sn', is_flag=True)
@click.option('--only-extra', is_flag=True)
@click.option('--only-intra', is_flag=True)
@click.option('--model-dir', type=click.Path())
@click.option('--data-dir', type=click.Path(exists=True),
              default=r'C:/Users/imoto/Documents/plasticc/data/raw')
@click.option('--n-jobs', type=int, default=1)
@click.option('--hidden-size', type=int, default=512)
@click.option('--static-feature', is_flag=True)
@click.option('--astronomy-feature', is_flag=True)
@click.option('--shuffle-only', is_flag=True)
@click.option('--epochs', type=int, default=200)
@click.option('--use-flux', is_flag=True)
@click.option('--use-magnitude', is_flag=True)
@click.option('--use-soft-label', is_flag=True)
@click.option('--feature-size', type=int)
def learn_all(only_sn, only_extra, only_intra, model_dir, data_dir, n_jobs,
              hidden_size, static_feature, astronomy_feature, shuffle_only,
              epochs, use_flux, use_magnitude, use_soft_label, feature_size):
    assert use_flux or use_magnitude

    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    data_dir = Path(data_dir)

    parameters = {
        'data': str(data_dir), 'hidden_size': hidden_size,
        'static_feature': static_feature,
        'only_sn': only_sn, 'only_extra': only_extra, 'only_intra': only_intra,
        'shuffle_only': shuffle_only, 'use_flux': use_flux,
        'use_magnitude': use_magnitude, 'astronomy_feature': astronomy_feature,
        'use_soft_label': use_soft_label, 'feature_size:': feature_size
    }
    with (model_dir / 'parameters.json').open('w') as f:
        json.dump(parameters, f, sort_keys=True, indent=4)

    data = load_dataset(
        data_dir=data_dir,
        only_sn=only_sn, only_extra=only_extra, only_intra=only_intra,
        astronomy_feature=astronomy_feature, static_feature=static_feature,
        feature_size=feature_size
    )

    y_map = data.y_map
    y_categorical = to_categorical(y_map)

    classes = data.classes
    y_count = Counter(y_map)
    weight_table = np.zeros_like(classes, dtype=np.float32)
    for i in range(len(classes)):
        weight_table[i] = y_count[i] / y_map.shape[0]

    batch_size = 100

    model_path = str(model_dir / "keras.model")
    check_point = ModelCheckpoint(
        model_path,
        monitor='val_loss', mode='min', save_best_only=True, verbose=0
    )
    progress_bar = ProgbarLogger(count_mode='steps')

    flag = np.ones(len(y_map), dtype=np.bool)
    if astronomy_feature:
        dataset = data.make_dataset(
            index=flag, batch_size=batch_size, is_training=True
        )
    else:
        dataset = data.make_dataset(
            index=flag, batch_size=batch_size,
            is_training=not shuffle_only, n_jobs=n_jobs,
            shuffle_only=shuffle_only,
            use_flux=use_flux, use_magnitude=use_magnitude,
            use_soft_label=use_soft_label
        )

    np.savez_compressed(
        str(model_dir / 'train.pickle'), x=dataset.x, y=dataset.y,
        object_id=dataset.object_id
    )
    joblib.dump(data.ss, str(model_dir / 'standard_scaler.pickle'))

    label = list(np.argmax(dataset.y, axis=-1))

    train_size = len(dataset.y)
    input_size = dataset.x.shape[1]
    print('input_size: {}'.format(input_size))

    train_y_count = Counter(label)
    train_weight_table = np.zeros_like(classes, dtype=np.float32)
    for i in range(len(classes)):
        train_weight_table[i] = train_y_count[i] / train_size

    model = build_model(
        input_size, classes, hidden_size=hidden_size,
        dropout_rate=0.5, activation='tanh'
    )
    model.compile(
        loss=lambda t, p: weighted_log_loss(
            y_true=t, y_pred=p, weight_table=train_weight_table
        ),
        optimizer='adam', metrics=['accuracy', top_2_accuracy,
                                   top_3_accuracy, top_4_accuracy]
    )
    history = model.fit_generator(
        generator=dataset,
        epochs=epochs, shuffle=False, verbose=1, workers=n_jobs,
        callbacks=[check_point]
    )

    plot_loss_acc(history, model_dir / 'result.png')
    plot_top_accuracy(history, model_dir / 'accuracy.png')


def main():
    cmd()


if __name__ == '__main__':
    main()
