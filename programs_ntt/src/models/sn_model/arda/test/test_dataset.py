#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import os
import unittest

import tensorflow as tf

from sn_model.arda.dataset import make_domain_dataset, load_data, compute_moments

__author__ = 'Yasuhiro Imoto'
__date__ = '21/9/2017'


class MyTestCaseDataset(unittest.TestCase):
    def setUp(self):
        root_path = r'\\opal\imoto\CREST3'
        date = 170810
        method = 'modified'
        directory_date = '{}_{}'.format(date, method)
        data_path = os.path.join(root_path, 'datalist', 'observation')
        artificial_path_train = os.path.join(data_path, 'train',
                                             directory_date,
                                             'dataset.classify.train.nc')
        test_path = os.path.join(data_path, 'test', directory_date,
                                 'dataset.classify.all.nc')

        self.test_path = test_path
        self.artificial_path = artificial_path_train

        self.artificial_data = load_data(artificial_path_train)
        mean, std = compute_moments(self.artificial_data[1])

        self.mean = mean
        self.std = std

    def test_data_size(self):
        dataset = make_domain_dataset(self.artificial_path, self.test_path,
                                      self.mean, self.std, batch_size=100,
                                      epochs=1, shuffle=False, training=False)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        n = 0
        with tf.Session() as sess:
            while True:
                try:
                    source_data, target_data = sess.run(next_element)
                    n += len(source_data[0])

                    # 不足分がNoneで埋められていないことを確認
                    self.assertIsNotNone(target_data[0][0])
                    self.assertIsNotNone(target_data[1][0])
                    self.assertIsNotNone(target_data[2][0])
                except tf.errors.OutOfRangeError:
                    break

        self.assertEqual(len(self.artificial_data[0]), n)


if __name__ == '__main__':
    unittest.main()
