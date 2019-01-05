#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import csv
import os

import numpy as np
from sklearn.externals import joblib

__author__ = 'Yasuhiro Imoto'
__date__ = '15/12/2017'


def predict_random_forest(test_data, output_dir, file_name):
    clf = joblib.load(os.path.join(output_dir, 'model.pickle'))
    results = clf.predict_proba(test_data['x'])

    with open(os.path.join(output_dir, file_name), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        for score, label in zip(results[:, 1], test_data['y']):
            writer.writerow([score, label])
