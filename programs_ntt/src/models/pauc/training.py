#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import os

import tensorflow as tf
from tqdm import tqdm

from .batch import relaxed, exact
from pauc.score import get_score_model

__author__ = 'Yasuhiro Imoto'
__date__ = '14/12/2017'


def train_pauc_relaxed(train_data, validation_data, epoch, resume, output_dir,
                       validation_frequency, batch_size,
                       lambda_constraints, gamma, patience, patience_increase,
                       improvement_threshold, model_type, **kwargs):
    model = get_score_model(
        model_type=model_type, positive_data=train_data['positive'],
        negative_data=train_data['negative'], **kwargs
    )

    train_operators = relaxed.make_objective_function(
        data=train_data, batch_size=batch_size, model=model, gamma=gamma,
        lambda_constraints=lambda_constraints, train=True
    )

    validation_operators = relaxed.make_objective_function(
        data=validation_data, batch_size=batch_size, model=model, gamma=gamma,
        lambda_constraints=lambda_constraints, train=False
    )

    train_pauc_helper(
        train_operators=train_operators,
        validation_operators=validation_operators, epoch=epoch, resume=resume,
        output_dir=output_dir, validation_frequency=validation_frequency,
        patience=patience, patience_increase=patience_increase,
        improvement_threshold=improvement_threshold
    )


def train_pauc_exact(train_data, validation_data, epoch, resume, output_dir,
                     validation_frequency, batch_size,
                     lambda_constraints, beta, patience, patience_increase,
                     improvement_threshold, model_type, **kwargs):
    model = get_score_model(
        model_type=model_type, positive_data=train_data['positive'],
        negative_data=train_data['negative'], **kwargs
    )

    train_operators = exact.make_objective_function(
        data=train_data, batch_size=batch_size, model=model, beta=beta,
        lambda_constraints=lambda_constraints, train=True
    )

    validation_operators = exact.make_objective_function(
        data=validation_data, batch_size=batch_size, model=model, beta=beta,
        lambda_constraints=lambda_constraints, train=False
    )

    train_pauc_helper(
        train_operators=train_operators,
        validation_operators=validation_operators, epoch=epoch, resume=resume,
        output_dir=output_dir, validation_frequency=validation_frequency,
        patience=patience, patience_increase=patience_increase,
        improvement_threshold=improvement_threshold
    )


def train_pauc_helper(train_operators, validation_operators, epoch, resume,
                      output_dir, validation_frequency, patience,
                      patience_increase, improvement_threshold):
    (train_iterator, train_update_op, apply_op, train_reset_op,
     train_summary_op1, train_summary_op2, step0_op) = train_operators

    (validation_iterator, validation_update_op, validation_score_op,
     validation_summary_op, validation_reset_op) = validation_operators

    global_step = tf.train.get_global_step()
    # pAUCなので最小値は0
    # 最小値以下で初期化
    best_validation_score = -100

    saver = tf.train.Saver()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        checkpoint = tf.train.get_checkpoint_state(output_dir)
        if resume or checkpoint:
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

        writer = tf.summary.FileWriter(output_dir)
        step = sess.run(global_step)

        if step >= epoch:
            return

        if step == 0:
            validation_score, validation_summary = evaluate_validation(
                sess, validation_iterator, validation_reset_op,
                validation_score_op, validation_update_op,
                validation_summary_op
            )
            writer.add_summary(validation_summary, global_step=step)

            sess.run(train_iterator.initializer)
            sess.run(train_reset_op)
            while True:
                try:
                    sess.run(step0_op)
                except tf.errors.OutOfRangeError:
                    break
            train_summary2 = sess.run(train_summary_op2)
            writer.add_summary(train_summary2, global_step=step)

        progress = tqdm(total=epoch - step)
        while True:
            sess.run(train_iterator.initializer)
            sess.run(train_reset_op)
            while True:
                try:
                    sess.run(train_update_op)
                except tf.errors.OutOfRangeError:
                    break
            sess.run(apply_op)

            step = sess.run(global_step)

            train_summary1 = sess.run(train_summary_op1)
            writer.add_summary(train_summary1, global_step=step)

            if step % validation_frequency == 0 or step > patience:
                validation_score, validation_summary = evaluate_validation(
                    sess, validation_iterator, validation_reset_op,
                    validation_score_op, validation_update_op,
                    validation_summary_op
                )
                writer.add_summary(validation_summary, global_step=step)

                train_summary2 = sess.run(train_summary_op2)
                writer.add_summary(train_summary2, global_step=step)

                if validation_score > best_validation_score:
                    if (validation_score * improvement_threshold >
                            best_validation_score):
                        # 大きく改善したので、学習を延長する
                        patience = max(patience, step * patience_increase)
                    best_validation_score = validation_score
                    # 保存
                    saver.save(sess, os.path.join(output_dir, 'model'),
                               global_step=global_step, write_meta_graph=False)

            progress.update()

            if step >= patience:
                print('early stopping (step: {}/{})'.format(step, epoch))
                break
            if step >= epoch:
                print('reached the max iterations '
                      '(step: {0}/{0})'.format(epoch))
                break

        if step % validation_frequency != 0:
            validation_score, validation_summary = evaluate_validation(
                sess, validation_iterator, validation_reset_op,
                validation_score_op, validation_update_op,
                validation_summary_op
            )
            writer.add_summary(validation_summary, global_step=step)

            if validation_score > best_validation_score:
                saver.save(sess, os.path.join(output_dir, 'model'),
                           global_step=global_step, write_meta_graph=False)


def evaluate_validation(sess, iterator, reset_op, score_op, update_op,
                        summary_op):
    sess.run(iterator.initializer)
    sess.run(reset_op)
    while True:
        try:
            sess.run(update_op)
        except tf.errors.OutOfRangeError:
            break
    score, summary = sess.run([score_op, summary_op])
    return score, summary
