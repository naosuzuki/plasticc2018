#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from hyperopt import fmin, tpe, hp, Trials
from sklearn.externals import joblib

from train_real_bogus_model import train_pauc_model
from predict_real_bogus_model import predict_pauc_model

__author__ = 'Yasuhiro Imoto'
__date__ = '25/12/2017'

# 目的関数内での最適化の対象となるパラメータ以外を与えるために利用する
flags = tf.app.flags
flags.DEFINE_string('data_dir', '../../data/processed/real_bogus',
                    'データのディレクトリ')
flags.DEFINE_string('data_name', 'HSC-G',
                    'HSC-G, HSC-I2, HSC-R2, HSC-Y, HSC-Z')
flags.DEFINE_string('output_dir', '../../models/real_bogus/pauc',
                    '計算結果の出力ディレクトリ')
flags.DEFINE_string('mode', 'relaxed', 'relaxed, exact')
flags.DEFINE_integer('n_iterations', 100, 'hyperoptで探索する回数(履歴を含む)')

flags.DEFINE_float('beta', 0.1, 'exactの場合のbeta')

flags.DEFINE_integer('epoch', 1000, 'pAUCの計算の反復回数')
flags.DEFINE_integer('batch_size', 1000, 'pAUCの計算時のバッチサイズ')
flags.DEFINE_integer('validation_frequency', 100,
                     'validation dataを評価する頻度')
flags.DEFINE_integer('seed', 0x5eed, 'データを分割するときのシード')
flags.DEFINE_string('split_ratio', '0.8,0.1,0.1', 'データを分割するときの比率')
FLAGS = flags.FLAGS

# 出力ディレクトリに連番を付ける
count = 0


def objective_relaxed(args):
    gamma = args['gamma']
    hidden_size = args['hidden_size']
    lambda1, lambda2 = args['lambda1'], args['lambda2']

    data_path = os.path.join(FLAGS.data_dir,
                             'param_{}.pickle'.format(FLAGS.data_name))
    global count
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.data_name, FLAGS.mode,
                              'trials/{0:03d}'.format(count))
    count += 1

    regularizer_weights = [lambda1, lambda2]
    split_ratio = convert_split_ratio()

    train_pauc_model(data_path=data_path, epoch=FLAGS.epoch, resume=False,
                     output_dir=output_dir, hidden_size=hidden_size,
                     regularizer_weights=regularizer_weights,
                     validation_frequency=FLAGS.validation_frequency,
                     seed=FLAGS.seed, split_ratio=split_ratio, mode='relaxed',
                     batch_size=FLAGS.batch_size, gamma=gamma)
    predict_pauc_model(data_path=data_path, output_dir=output_dir)
    # グラフの情報が残っているので、連続で呼ぶと次はエラーになる
    # グラフの情報を消す
    tf.reset_default_graph()

    # output_dirにtrain, validation, testについてのpAUCでの
    # 評価結果のファイルができている
    # FPR=0.05でのpAUCの値を最大化することにする
    df = pd.read_csv(os.path.join(output_dir, 'summary-test.csv'), index_col=0,
                     header=0)
    score = df.loc[0.05, 'pAUC']

    # 最小化問題に変換
    return -score


def objective_exact(args):
    beta = FLAGS.beta
    hidden_size = args['hidden_size']
    lambda1, lambda2 = args['lambda1'], args['lambda2']

    data_path = os.path.join(FLAGS.data_dir,
                             'param_{}.pickle'.format(FLAGS.data_name))
    global count
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.data_name, FLAGS.mode,
                              'trials/{0:03d}'.format(count))
    count += 1

    regularizer_weights = [lambda1, lambda2]
    split_ratio = convert_split_ratio()

    train_pauc_model(data_path=data_path, epoch=FLAGS.epoch, resume=False,
                     output_dir=output_dir, hidden_size=hidden_size,
                     regularizer_weights=regularizer_weights,
                     validation_frequency=FLAGS.validation_frequency,
                     seed=FLAGS.seed, split_ratio=split_ratio, mode='exact',
                     batch_size=FLAGS.batch_size, beta=beta)
    predict_pauc_model(data_path=data_path, output_dir=output_dir)
    # グラフの情報が残っているので、連続で呼ぶと次はエラーになる
    # グラフの情報を消す
    tf.reset_default_graph()

    # output_dirにtrain, validation, testについてのpAUCでの
    # 評価結果のファイルができている
    # FPR=0.05でのpAUCの値を最大化することにする
    df = pd.read_csv(os.path.join(output_dir, 'summary-test.csv'), index_col=0,
                     header=0)
    score = df.loc[0.05, 'pAUC']

    # 最小化問題に変換
    return -score


def convert_split_ratio():
    ratio = FLAGS.split_ratio   # type: str
    r = ratio.split(',')
    split_ratio = [float(v) for v in r]
    return split_ratio


def get_search_space_relaxed():
    space = {
        # (0, 1)の範囲になるように両端に少し隙間を与える
        'gamma': hp.uniform('gamma', 1e-3, 1 - 1e-3),
        # 大きい値では間隔が広がるようにする
        # 最小単位が10になるように設定
        'hidden_size': hp.qloguniform('hidden_size',
                                      np.log(10), np.log(1000), 10),
        'lambda1': hp.loguniform('lambda1', np.log(1e-3), np.log(1e2)),
        'lambda2': hp.loguniform('lambda2', np.log(1e-3), np.log(1e2))
    }
    return space


def get_search_space_exact():
    space = {
        # 大きい値では間隔が広がるようにする
        # 最小単位が10になるように設定
        'hidden_size': hp.qloguniform('hidden_size',
                                      np.log(10), np.log(1000), 10),
        'lambda1': hp.loguniform('lambda1', np.log(1e-3), np.log(1e2)),
        'lambda2': hp.loguniform('lambda2', np.log(1e-3), np.log(1e2))
    }
    return space


def main(_):
    global count

    output_dir = os.path.join(FLAGS.output_dir, FLAGS.data_name, FLAGS.mode)
    # 履歴の読み込み
    trial_file = os.path.join(output_dir, 'trials.pickle')
    if os.path.exists(trial_file):
        trials = joblib.load(trial_file)    # type: Trials
        count = len(trials.trials)
    else:
        trials = Trials()

    if FLAGS.mode == 'relaxed':
        objective = objective_relaxed
        space = get_search_space_relaxed()
    else:
        objective = objective_exact
        space = get_search_space_exact()

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
