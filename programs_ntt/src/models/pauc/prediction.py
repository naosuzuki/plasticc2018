#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import os
import csv

import numpy as np
import tensorflow as tf
try:
    from tensorflow.python.data import Dataset
except (ImportError, SystemError):
    from tensorflow.contrib.data import Dataset

__author__ = 'Yasuhiro Imoto'
__date__ = '14/12/2017'


def predict_pauc(test_data, output_dir, file_name, model, mean, std):
    """

    :param test_data: 変換済みのデータとラベル
    :param output_dir:
    :param file_name:
    :param model:
    :param mean:
    :param std:
    :return:
    """
    dataset = Dataset.from_tensor_slices(test_data['x'].astype(np.float32))
    dataset = dataset.map(lambda x: (x - mean) / std)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(1000)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    outputs = model(next_element)

    saver = tf.train.Saver()
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        checkpoint = tf.train.get_checkpoint_state(output_dir)
        if checkpoint:
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            assert False, 'model is not found.'

        results = []
        while True:
            try:
                tmp = sess.run(outputs)
                results.append(np.squeeze(tmp))
            except tf.errors.OutOfRangeError:
                break
        results = np.hstack(results)

        with open(os.path.join(output_dir, file_name), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=' ')
            for score, label in zip(results, test_data['y']):
                writer.writerow([score, label])
