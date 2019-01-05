#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf

from sn_model.arda import setup_model

__author__ = 'Yasuhiro Imoto'
__date__ = '04/9/2017'


class MyTestCaseClassifyWeight(unittest.TestCase):
    """
    クラス判別器の学習中にclassifyとfeature generatorの重みは変化し、
    criticの重みは不変であることを確認する
    """
    @classmethod
    def setUpClass(cls):
        input_size = 10
        output_size = 7
        tmp = setup_model(input_size, 30, output_size, 0.5, 0.1, 1.0,
                          training=True)
        cls.opt = tmp[0]
        cls.placeholder = tmp[1]

        cls.input_size = input_size
        cls.output_size = output_size

    def _make_input(self, n):
        x_source = np.random.randn(n, self.input_size)
        x_target = np.random.randn(n, self.input_size)
        y_category = np.random.choice(self.output_size, n)
        return [x_source, x_target], y_category

    def test_variable_count_feature(self):
        variables = tf.global_variables()

        count = np.sum([1 for v in variables if 'feature' in v.name])
        # denseが3個分、adam optimizerで3倍になる
        self.assertEqual(6 * 3, count, msg=[v.name for v in variables
                                            if 'feature' in v.name])

    def test_variable_count_classifier(self):
        variables = tf.global_variables()

        count = np.sum([1 for v in variables if 'classifier' in v.name])
        # denseが3個とbatch_normalizationが1個
        self.assertEqual(6 * 3 + 4 * 2, count,
                         msg=[v.name for v in variables
                              if 'classifier' in v.name])

    def test_update_feature(self):
        variables = {v.name: v for v in tf.global_variables()
                     if 'feature' in v.name}

        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            v0 = {key: sess.run(value) for key, value in variables.items()}

            for i in range(20):
                (source, target), label = self._make_input(100)
                feed_dict = {self.placeholder['source']: source,
                             self.placeholder['target']: target,
                             self.placeholder['label']: label}
                sess.run(self.opt['classifier'], feed_dict=feed_dict)

            v1 = {key: sess.run(value) for key, value in variables.items()}

        for key in v0:
            self.assertFalse(np.array_equal(v0[key], v1[key]), msg=key)

    def test_update_critic(self):
        variables = {v.name: v for v in tf.global_variables()
                     if 'critic' in v.name}

        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            v0 = {key: sess.run(value) for key, value in variables.items()}

            for i in range(20):
                (source, target), label = self._make_input(100)
                feed_dict = {self.placeholder['source']: source,
                             self.placeholder['target']: target,
                             self.placeholder['label']: label}
                sess.run(self.opt['classifier'], feed_dict=feed_dict)

            v1 = {key: sess.run(value) for key, value in variables.items()}

        for key in v0:
            self.assertTrue(np.array_equal(v0[key], v1[key]), msg=key)

    def test_update_classifier(self):
        variables = {v.name: v for v in tf.global_variables()
                     if 'classifier' in v.name}

        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            v0 = {key: sess.run(value) for key, value in variables.items()}

            for i in range(20):
                (source, target), label = self._make_input(100)
                feed_dict = {self.placeholder['source']: source,
                             self.placeholder['target']: target,
                             self.placeholder['label']: label}
                sess.run(self.opt['classifier'], feed_dict=feed_dict)

            v1 = {key: sess.run(value) for key, value in variables.items()}

        for key in v0:
            self.assertFalse(np.array_equal(v0[key], v1[key]), msg=key)

    def test_update_feature_resume(self):
        saver = tf.train.Saver()
        with TemporaryDirectory() as tmp:
            with tf.Session() as sess:
                variables = {v.name: v for v in tf.global_variables()
                             if 'feature' in v.name}

                sess.run(tf.group(tf.global_variables_initializer(),
                                  tf.local_variables_initializer()))
                for i in range(20):
                    (source, target), label = self._make_input(100)
                    feed_dict = {self.placeholder['source']: source,
                                 self.placeholder['target']: target,
                                 self.placeholder['label']: label}
                    sess.run(self.opt['classifier'], feed_dict=feed_dict)
                saver.save(sess, os.path.join(tmp, 'model'))

                v0 = {key: sess.run(value) for key, value in variables.items()}

            with tf.Session() as sess:
                sess.run(tf.local_variables_initializer())
                checkpoint = tf.train.get_checkpoint_state(tmp)
                saver.restore(sess, checkpoint.model_checkpoint_path)

                variables = {v.name: v for v in tf.global_variables()
                             if 'feature' in v.name}

                v1 = {key: sess.run(value) for key, value in variables.items()}

                for i in range(20):
                    (source, target), label = self._make_input(100)
                    feed_dict = {self.placeholder['source']: source,
                                 self.placeholder['target']: target,
                                 self.placeholder['label']: label}
                    sess.run(self.opt['classifier'], feed_dict=feed_dict)

                v2 = {key: sess.run(value) for key, value in variables.items()}

        for key in v0:
            self.assertTrue(np.array_equal(v0[key], v1[key]), msg=key)
            self.assertFalse(np.array_equal(v1[key], v2[key]), msg=key)

    def test_update_critic_resume(self):
        saver = tf.train.Saver()
        with TemporaryDirectory() as tmp:
            with tf.Session() as sess:
                variables = {v.name: v for v in tf.global_variables()
                             if 'critic' in v.name}

                sess.run(tf.group(tf.global_variables_initializer(),
                                  tf.local_variables_initializer()))
                for i in range(20):
                    (source, target), label = self._make_input(100)
                    feed_dict = {self.placeholder['source']: source,
                                 self.placeholder['target']: target,
                                 self.placeholder['label']: label}
                    sess.run(self.opt['classifier'], feed_dict=feed_dict)
                saver.save(sess, os.path.join(tmp, 'model'))

                v0 = {key: sess.run(value) for key, value in variables.items()}

            with tf.Session() as sess:
                sess.run(tf.local_variables_initializer())
                checkpoint = tf.train.get_checkpoint_state(tmp)
                saver.restore(sess, checkpoint.model_checkpoint_path)

                variables = {v.name: v for v in tf.global_variables()
                             if 'critic' in v.name}

                v1 = {key: sess.run(value) for key, value in variables.items()}

                for i in range(20):
                    (source, target), label = self._make_input(100)
                    feed_dict = {self.placeholder['source']: source,
                                 self.placeholder['target']: target,
                                 self.placeholder['label']: label}
                    sess.run(self.opt['classifier'], feed_dict=feed_dict)

                v2 = {key: sess.run(value) for key, value in variables.items()}

        for key in v0:
            self.assertTrue(np.array_equal(v0[key], v1[key]), msg=key)
            self.assertTrue(np.array_equal(v1[key], v2[key]), msg=key)

    def test_update_classifier_resume(self):
        saver = tf.train.Saver()
        with TemporaryDirectory() as tmp:
            with tf.Session() as sess:
                variables = {v.name: v for v in tf.global_variables()
                             if 'classifier' in v.name}

                sess.run(tf.group(tf.global_variables_initializer(),
                                  tf.local_variables_initializer()))
                for i in range(20):
                    (source, target), label = self._make_input(100)
                    feed_dict = {self.placeholder['source']: source,
                                 self.placeholder['target']: target,
                                 self.placeholder['label']: label}
                    sess.run(self.opt['classifier'], feed_dict=feed_dict)
                saver.save(sess, os.path.join(tmp, 'model'))

                v0 = {key: sess.run(value) for key, value in variables.items()}

            with tf.Session() as sess:
                sess.run(tf.local_variables_initializer())
                checkpoint = tf.train.get_checkpoint_state(tmp)
                saver.restore(sess, checkpoint.model_checkpoint_path)

                variables = {v.name: v for v in tf.global_variables()
                             if 'classifier' in v.name}

                v1 = {key: sess.run(value) for key, value in variables.items()}

                for i in range(20):
                    (source, target), label = self._make_input(100)
                    feed_dict = {self.placeholder['source']: source,
                                 self.placeholder['target']: target,
                                 self.placeholder['label']: label}
                    sess.run(self.opt['classifier'], feed_dict=feed_dict)

                v2 = {key: sess.run(value) for key, value in variables.items()}

        for key in v0:
            self.assertTrue(np.array_equal(v0[key], v1[key]), msg=key)
            self.assertFalse(np.array_equal(v1[key], v2[key]), msg=key)


if __name__ == '__main__':
    unittest.main()
