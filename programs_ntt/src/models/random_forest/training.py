#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

__author__ = 'Yasuhiro Imoto'
__date__ = '15/12/2017'


def train_random_forest(train_data, n_estimators, max_depth, n_jobs, resume,
                        output_dir, mode, random_state, **kwargs):
    x = train_data['x']
    y = train_data['y']

    # 前回の結果から再学習できるように設定
    # 実際にあるのかはわからない
    if resume:
        clf = joblib.load(os.path.join(output_dir, 'model.pickle'))
    else:
        if mode == 'xgboost':
            # condaでインストールできたバージョンが少し古いのか
            # random_stateがまだ有効になっていないので代わりにseedを指定する
            clf = xgb.XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth, nthread=n_jobs,
                seed=random_state
            )
        else:
            # resumeのためにwarm_start=Trueを設定
            # warm_start=Trueは速度にペナルティがあるかも
            clf = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs,
                random_state=random_state,
                criterion=kwargs['criterion'], warm_start=True
            )

    clf.fit(x, y)

    # 学習済みのモデルを保存
    joblib.dump(clf, os.path.join(output_dir, 'model.pickle'))

