#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import json
import os
from time import time
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from predict_real_bogus_model import compute_pauc

__author__ = 'Yasuhiro Imoto'
__date__ = '26/12/2017'

# 目的関数内での最適化の対象となるパラメータ以外を与えるために利用する
flags = tf.app.flags
flags.DEFINE_string('data_dir', '../../data/processed/real_bogus',
                    'データのディレクトリ')
flags.DEFINE_string('data_name', 'HSC-G',
                    'HSC-G, HSC-I2, HSC-R2, HSC-Y, HSC-Z')
flags.DEFINE_string('output_dir', '../../models/real_bogus/random_forest',
                    '計算結果の出力ディレクトリ')
flags.DEFINE_string('mode', 'random_forest', 'random_forest, xgboost')
flags.DEFINE_integer('n_iterations', 100, 'hyperoptで探索する回数(履歴を含む)')

flags.DEFINE_float('fpr', 0.05, 'スコアのpAUCを計算するときのFPR')
flags.DEFINE_integer('cv', 5, 'K-fold cv')
FLAGS = flags.FLAGS


def load_data():
    data_path = os.path.join(FLAGS.data_dir,
                             'param_{}.pickle'.format(FLAGS.data_name))
    df = pd.read_pickle(data_path)  # pd.DataFrame
    df = df.fillna(10)
    # 変換
    tmp = 1.0857 / df.loc[:, 'magerr'].values
    df.loc[:, 'magerr'] = tmp

    # 必要な列を抜き出す
    # 順序の固定を行う
    columns = ['magerr', 'elongation.norm', 'fwhm.norm',
               'significance.abs', 'residual', 'psffit.sigma.ratio',
               'psffit.peak.ratio', 'frac.det', 'density',
               'density.good', 'baPsf', 'sigmaPsf']
    data = df[columns].values
    label = df['real/bogus'].values

    return data, label


def score_function(estimator, X, y):
    p = estimator.predict_proba(X)
    # ラベルが1の結果のみを抽出
    p = p[:, 1]
    positive = p[y != 0]
    negative = p[y == 0]

    negative = -np.sort(-negative)
    fpr_list = (0.01, 0.05, 0.1, 0.5, 1.0)
    result = [compute_pauc(positive, negative, fpr) for fpr in fpr_list]
    df = pd.DataFrame(result, index=fpr_list, columns=['pAUC', 'TPR'])

    return df


def objective_random_forest(args):
    data, label = load_data()

    args['n_estimators'] = int(args['n_estimators'])
    clf = RandomForestClassifier(n_jobs=-1, **args)

    scores = []
    skf = StratifiedKFold(n_splits=FLAGS.cv)
    for train_index, test_index in skf.split(data, label):
        train_data, train_label = data[train_index], label[train_index]
        test_data, test_label = data[test_index], label[test_index]

        clf.fit(train_data, train_label)

        scores.append(score_function(clf, test_data, test_label))
    tmp_scores = [df.loc[FLAGS.fpr, 'pAUC'] for df in scores]
    mean = np.mean(tmp_scores)
    variance = np.var(tmp_scores)

    d = {'loss': -mean, 'status': STATUS_OK,
         'eval_time': time(),
         'loss_variance': variance,
         'attachments': {str(i): df.to_dict() for i, df in enumerate(scores)}}
    return d


def get_search_space_random_forest():
    space = {
        'n_estimators': hp.qloguniform('n_estimators',
                                       np.log(20), np.log(2000), 10),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_features': hp.quniform('max_features', 0.1, 1.0, 0.1)
    }
    return space


def main(_):
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.data_name, FLAGS.mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 履歴の読み込み
    trial_file = os.path.join(output_dir, 'trials.pickle')
    if os.path.exists(trial_file):
        trials = joblib.load(trial_file)  # type: Trials
    else:
        trials = Trials()

    if FLAGS.mode == 'random_forest':
        objective = objective_random_forest
        space = get_search_space_random_forest()
    else:
        objective = None
        space = None

    best = fmin(objective, space=space, algo=tpe.suggest,
                max_evals=FLAGS.n_iterations, trials=trials)

    # 履歴の保存
    joblib.dump(trials, trial_file)

    print('best score: ', best)
    print(trials.best_trial)
    with open(os.path.join(output_dir, 'best_parameter.json'), 'w') as f:
        json.dump(trials.best_trial, f, sort_keys=True, indent=4,
                  default=serialize_datetime)


def serialize_datetime(o):
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(repr(o) + ' is not JSON serializable')


if __name__ == '__main__':
    tf.app.run()
