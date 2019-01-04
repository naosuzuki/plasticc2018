#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
light gbmが良かったのか入力特徴量が良かったのかを調べるために
light gbmで高橋様たちの天文的な特徴量を分類する
"""
import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import matplotlib.pyplot as plt
    import seaborn as sns

__author__ = 'Yasuhiro Imoto'
__date__ = '25/12/2018'


def load_data(data_path):
    data_path = Path(data_path)

    df = pd.read_csv(
        data_path,
        names=['object_id', 'mjd', 'passband', 'flux', 'flux_err',
               'detected', 'interpolation']
    )
    meta = pd.read_csv(
        data_path.with_name('training_set_metadata.csv'), header=0
    )
    tmp_meta = meta[['object_id', 'target']]