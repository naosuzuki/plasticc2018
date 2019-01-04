#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import numpy as np
try:
    from tensorflow.python.data import Dataset
except (SystemError, ImportError):
    from tensorflow.contrib.data import Dataset

__author__ = 'Yasuhiro Imoto'
__date__ = '28/12/2017'


def get_data(data, batch_size):
    # positive dataはそのままモデルに入力
    # negative dataはtf.data.Datasetでバッチサイズごとに取り出す
    negative_data = data['negative'].astype(np.float32)
    dataset = Dataset.from_tensor_slices(negative_data)
    dataset = dataset.repeat(1).batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    positive_data = data['positive'].astype(np.float32)

    return positive_data, next_element, iterator
