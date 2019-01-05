#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import matplotlib.pyplot as plt
    import seaborn as sns

__author__ = 'Yasuhiro Imoto'
__date__ = '25/12/2018'


NUM_FOLDS = 5


def get_hostgal_range(hostgal_photoz):
    return np.clip(hostgal_photoz//0.2, 0, 6)


class SampleWeighter(object):
    # Example usage:
    # >> sw = SampleWeighter(train_exgal["hostgal_photoz"],
    #                        test_exgal["hostgal_photoz"])
    # >> train_exgal = calculate_weights(train_exgal, False)

    def __init__(self, train_exgal_hp, test_exgal_hp):
        train_exgal_hr = get_hostgal_range(train_exgal_hp)
        test_exgal_hr = get_hostgal_range(test_exgal_hp)

        # noinspection PyUnresolvedReferences
        train_hp_dist = (train_exgal_hr.value_counts() /
                         len(train_exgal_hr)).to_dict()
        # noinspection PyUnresolvedReferences
        test_hp_dist = (test_exgal_hr.value_counts() /
                        len(test_exgal_hr)).to_dict()
        self.weight_list = [
            test_hp_dist[i] / train_hp_dist[i]
            for i in range(int(train_exgal_hr.max()) + 1)
        ]

    def calculate_weights(self, df, is_galactic):
        # gives weights so that test set hostgal_photoz distribution
        # is represented
        if is_galactic:
            df["sample_weight"] = 1.0
        else:
            # noinspection PyUnresolvedReferences
            df["sample_weight"] = (
                get_hostgal_range(df["hostgal_photoz"]).apply(
                    lambda x: self.weight_list[int(x)]
                )
            )
            # df["sample_weight"] = 1.0

        # gives more weights to non-ddf
        # because they are more common in test set
        df["sample_weight"] *= (2 - df["ddf"])

        # normalizes the weights so that each class has total sum of 100
        # (effecting training equally)
        df["sample_weight"] *= (
                100 / df.groupby("target")["sample_weight"].transform("sum")
        )

        # doubles weights for class 15 and class 64
        df["sample_weight"] *= df["target"].apply(
            lambda x: 1 + (x in {15, 64})
        )
        return df


def map_classes(df):
    class_list = df["target"].value_counts(ascending=True).index
    class_dict = {}
    for i, c in enumerate(class_list):
        class_dict[c] = i

    df["target"] = df["target"].map(class_dict)
    return df, class_list.tolist()


def load_data(feature_data_dir, meta_data_dir):
    feature_data_dir = Path(feature_data_dir)
    meta_data_dir = Path(meta_data_dir)

    train_df = pd.read_csv(
        meta_data_dir / 'training_set_metadata.csv', header=0
    )
    test_df = pd.read_csv(
        meta_data_dir / 'test_set_metadata.csv', header=0
    )
    for feature_file in ["bazin_features", "features_v1", "features_v2"]:
        train_df = train_df.merge(
            pd.read_pickle(
                feature_data_dir /
                "training_set_{}.pickle".format(feature_file)
            ),
            on="object_id", how="left")
        test_df = test_df.merge(
            pd.read_pickle(
                feature_data_dir / "test_set_{}.pickle".format(feature_file)
            ),
            on="object_id", how="left")

    # hostgal_calc_df = pd.read_csv("features/hostgal_calc.csv")
    # train_df = train_df.merge(hostgal_calc_df, on="object_id", how="left")
    # test_df = test_df.merge(hostgal_calc_df, on="object_id", how="left")

    train_gal = train_df[train_df["hostgal_photoz"] == 0].copy()
    train_exgal = train_df[train_df["hostgal_photoz"] > 0].copy()
    test_gal = test_df[test_df["hostgal_photoz"] == 0].copy()
    test_exgal = test_df[test_df["hostgal_photoz"] > 0].copy()

    sw = SampleWeighter(train_exgal["hostgal_photoz"],
                        test_exgal["hostgal_photoz"])

    train_gal = sw.calculate_weights(train_gal, True)
    train_exgal = sw.calculate_weights(train_exgal, False)

    train_gal, gal_class_list = map_classes(train_gal)
    train_exgal, exgal_class_list = map_classes(train_exgal)

    return (train_gal, train_exgal, test_gal, test_exgal,
            gal_class_list, exgal_class_list,
            test_df[["object_id", "hostgal_photoz"]])


def load_data_v2(data_dir, meta_path):
    data_dir = Path(data_dir)

    df = pd.read_csv(meta_path, header=0)

    tmp_list = []
    for f in data_dir.glob('bazin*.pickle'):
        # noinspection PyTypeChecker
        tmp = pd.read_pickle(f)
        tmp_list.append(tmp)
    bazin = pd.concat(tmp_list, axis=0, ignore_index=True)
    df = df.merge(bazin, on='object_id', how='right')

    for name in ('features_v1', 'features_v2'):
        tmp_list = []
        for f in data_dir.glob('{}_*.pickle'.format(name)):
            # noinspection PyTypeChecker
            tmp = pd.read_pickle(f)
            tmp_list.append(tmp)
        features = pd.concat(tmp_list, axis=0, ignore_index=True)
        df = df.merge(features, on=['object_id', 'sub_id'], how='left')

    redshift_list = []
    for f in data_dir.glob('data*.h5'):
        with pd.HDFStore(f) as store:
            tmp = store['/redshift']
            tmp.index.name = 'object_id'
            tmp = pd.melt(
                tmp.reset_index(), id_vars=['object_id'],
                value_vars=tmp.columns, var_name='sub_id',
                value_name='hostgal_calc'
            )
            redshift_list.append(tmp)
    redshift = pd.concat(redshift_list, axis=0)
    redshift = redshift.astype({'sub_id': np.int})
    df = df.merge(redshift, on=['object_id', 'sub_id'], how='left')

    df.to_csv(data_dir / 'features.csv')

    df_galactic = df.copy()
    df_extra = df.copy()

    sw = SampleWeighter(df_galactic["hostgal_photoz"],
                        df_galactic["hostgal_photoz"])

    df_galactic = sw.calculate_weights(df_galactic, True)
    df_extra = sw.calculate_weights(df_extra, False)

    df_galactic, galactic_class_list = map_classes(df_galactic)
    df_extra, extra_class_list = map_classes(df_extra)

    return df_galactic, galactic_class_list, df_extra, extra_class_list


def train(df, features, parameters):
    oof_predictions = np.zeros((len(df), parameters['num_classes']))

    if 'sub_id' in df.columns:
        unique_id, index = np.unique(df['object_id'], return_index=True)
        target = df.iloc[index]['target']
    else:
        target = df['target'].values

    skf = StratifiedKFold(5, random_state=42)
    for train_index, validation_index in skf.split(target, target):
        if 'sub_id' in df.columns:
            flag_train = np.zeros(len(df), dtype=np.bool)
            # noinspection PyUnboundLocalVariable
            for i in unique_id[train_index]:
                flag_train = np.logical_or(flag_train, df['object_id'] == i)
            dev = df[flag_train]

            flag_validation = np.zeros(len(df), dtype=np.bool)
            for i in unique_id[validation_index]:
                flag_validation = np.logical_or(flag_validation,
                                                df['object_id'] == i)
            val = df[flag_validation]
        else:
            dev, val = df.iloc[train_index], df.iloc[validation_index]

        lgb_train = lgb.Dataset(dev[features], dev['target'],
                                weight=dev['sample_weight'])
        lgb_validation = lgb.Dataset(val[features], val['target'],
                                     weight=val['sample_weight'])

        model = lgb.train(
            parameters, lgb_train, num_boost_round=200,
            valid_sets=[lgb_train, lgb_validation], early_stopping_rounds=10,
            verbose_eval=50
        )
        if 'sub_id' in df.columns:
            # noinspection PyUnboundLocalVariable
            oof_predictions[flag_validation, :] = model.predict(val[features])
        else:
            oof_predictions[validation_index, :] = model.predict(val[features])

    return oof_predictions


def train_and_predict(train_df, test_df, features, params):
    oof_preds = np.zeros((len(train_df), params["num_class"]))
    test_preds = np.zeros((len(test_df), params["num_class"]))

    skf = StratifiedKFold(NUM_FOLDS, random_state=4)

    for train_index, val_index in skf.split(train_df, train_df["target"]):
        dev_df, val_df = train_df.iloc[train_index], train_df.iloc[val_index]
        lgb_train = lgb.Dataset(
            dev_df[features], dev_df["target"], weight=dev_df["sample_weight"]
        )
        lgb_val = lgb.Dataset(
            val_df[features], val_df["target"], weight=val_df["sample_weight"]
        )

        model = lgb.train(
            params, lgb_train, num_boost_round=200,
            valid_sets=[lgb_train, lgb_val], early_stopping_rounds=10,
            verbose_eval=50
        )
        oof_preds[val_index, :] = model.predict(val_df[features])

        test_preds += model.predict(test_df[features]) / NUM_FOLDS

    return oof_preds, test_preds


def get_lgb_predictions(train_gal, train_exgal, test_gal, test_exgal,
                        gal_class_list, exgal_class_list, model_dir):
    bazin = ["A", "B", "tfall", "trise", "cc", "fit_error", "t0_shift"]
    f_flux = (["flux_sn" + str(i) for i in range(6)] +
              ["sn" + str(i) for i in range(6)])
    f_skew = ["skew" + str(i) for i in range(6)]
    f_f = ["f" + str(i) for i in range(6)]
    f_d = ["d" + str(i) for i in range(6)]
    f_dd = ["dd" + str(i) for i in range(6)]
    v3_features = ['first', 'last', 'peak', 'deep', 'till_peak', 'after_peak',
                   'deep_peak', 'peak_32']
    # peak_time = ["peak_time" + str(i) for i in [0, 1, 4, 5]]

    features_gal = (
            ['mwebv', 'flux', 'flux_err', 'fake_flux', 'total_detected',
             'ratio_detected', 'observation_count', 'std_flux', 'min_flux',
             'max_flux', 'delta_flux', 'detected_flux', 'time_diff_pos',
             'time_diff_neg'] +
            f_flux + f_skew + f_f + f_d + f_dd + v3_features + bazin
    )

    # features_exgal = (
    #         ['hostgal_photoz', 'hostgal_photoz_err', 'hostgal_calc', 'mwebv',
    #          'fake_flux', 'time_diff_pos', 'time_diff_neg'] +
    #         f_flux + f_skew + f_f + f_d + v3_features + bazin + peak_time
    # )
    features_exgal = (
            ['hostgal_photoz', 'hostgal_photoz_err', 'hostgal_photoz', 'mwebv',
             'fake_flux', 'time_diff_pos', 'time_diff_neg'] +
            f_flux + f_skew + f_f + f_d + v3_features + bazin
    )

    params_gal = {"objective": "multiclass",
                  "num_class": len(gal_class_list),
                  "min_data_in_leaf": 200,
                  "num_leaves": 5,
                  "feature_fraction": 0.7
                  }

    params_exgal = {"objective": "multiclass",
                    "num_class": len(exgal_class_list),
                    "min_data_in_leaf": 200,
                    "num_leaves": 5,
                    "feature_fraction": 0.7
                    }

    print("GALACTIC MODEL")
    oof_preds_gal, test_preds_gal = train_and_predict(
        train_gal, test_gal, features_gal, params_gal
    )
    print("EXTRAGALACTIC MODEL")
    oof_preds_exgal, test_preds_exgal = train_and_predict(
        train_exgal, test_exgal, features_exgal, params_exgal
    )

    evaluate(train_gal, train_exgal, oof_preds_gal, oof_preds_exgal, model_dir)

    return oof_preds_gal, oof_preds_exgal, test_preds_gal, test_preds_exgal


def get_lgb_predictions_augmented(df_galactic, galactic_class_list,
                                  df_extra, extra_class_list):
    bazin = ["A", "B", "tfall", "trise", "cc", "fit_error", "t0_shift"]
    f_flux = ["flux_sn" + str(i) for i in range(6)] + ["sn" + str(i) for i in
                                                       range(6)]
    f_skew = ["skew" + str(i) for i in range(6)]
    f_f = ["f" + str(i) for i in range(6)]
    f_d = ["d" + str(i) for i in range(6)]
    f_dd = ["dd" + str(i) for i in range(6)]
    v3_features = ['first', 'last', 'peak', 'deep', 'till_peak', 'after_peak',
                   'deep_peak', 'peak_32']
    # peak_time = ["peak_time" + str(i) for i in [0, 1, 4, 5]]

    features_gal = (
            ['mwebv', 'flux', 'flux_err', 'fake_flux', 'total_detected',
             'ratio_detected', 'observation_count', 'std_flux', 'min_flux',
             'max_flux', 'delta_flux', 'detected_flux', 'time_diff_pos',
             'time_diff_neg'] +
            f_flux + f_skew + f_f + f_d + f_dd + v3_features + bazin
    )
    print('feature size (galactic): {}'.format(len(features_gal)))

    # features_exgal = (
    #         ['hostgal_photoz', 'hostgal_photoz_err', 'hostgal_calc', 'mwebv',
    #          'fake_flux', 'time_diff_pos', 'time_diff_neg'] +
    #         f_flux + f_skew + f_f + f_d + v3_features + bazin + peak_time
    # )
    features_exgal = (
            ['hostgal_photoz', 'hostgal_photoz_err', 'hostgal_photoz', 'mwebv',
             'fake_flux', 'time_diff_pos', 'time_diff_neg'] +
            f_flux + f_skew + f_f + f_d + v3_features + bazin
    )
    print('feature size (extra galactic): {}'.format(len(features_exgal)))

    params_gal = {
        "objective": "multiclass", "num_classes": len(galactic_class_list),
        "min_data_in_leaf": 200, "num_leaves": 5, "feature_fraction": 0.7
    }

    params_exgal = {
        "objective": "multiclass", "num_classes": len(extra_class_list),
        "min_data_in_leaf": 200, "num_leaves": 5, "feature_fraction": 0.7
    }

    print("GALACTIC MODEL")
    oof_preds_gal = train(df_galactic, features_gal, params_gal)
    print("EXTRAGALACTIC MODEL")
    oof_preds_exgal = train(df_extra, features_exgal, params_exgal)

    if 'sub_id' in df_extra.columns:
        oof_preds_gal = pd.DataFrame(
            oof_preds_gal,
            index=df_galactic.set_index(['object_id', 'sub_id']).index,
            columns=galactic_class_list
        )
        oof_preds_exgal = pd.DataFrame(
            oof_preds_exgal,
            index=df_extra.set_index(['object_id', 'sub_id']).index,
            columns=extra_class_list
        )
    else:
        oof_preds_gal = pd.DataFrame(
            oof_preds_gal,
            index=df_galactic.set_index('object_id').index,
            columns=galactic_class_list
        )
        oof_preds_exgal = pd.DataFrame(
            oof_preds_exgal,
            index=df_extra.set_index('object_id').index,
            columns=extra_class_list
        )
    return oof_preds_gal, oof_preds_exgal


def evaluate(train_gal, train_exgal, oof_preds_gal, oof_preds_exgal,
             model_dir):
    gal_loss = log_loss(train_gal["target"], np.round(oof_preds_gal, 4),
                        sample_weight=train_gal["sample_weight"])
    exgal_loss = log_loss(train_exgal["target"], np.round(oof_preds_exgal, 4),
                          sample_weight=train_exgal["sample_weight"])
    print("Galactic CV: {}".format(gal_loss))
    print("Extragalactic CV: {}".format(exgal_loss))
    print("Overall CV: {}".format((5 / (14 + 2)) * gal_loss +
                                  ((9 + 2) / (14 + 2)) * exgal_loss))

    d = {'galactic_log_loss': float(gal_loss),
         'extra_galactic_log_loss': float(exgal_loss),
         'log_loss': float((5 / (14 + 2)) * gal_loss +
                           ((9 + 2) / (14 + 2)) * exgal_loss)}
    with (model_dir / 'result.json').open('w') as f:
        json.dump(d, f, indent=4, sort_keys=True)


# Example usage: test_preds_exgal = get_meta_preds(train_exgal,
# oof_preds_exgal, test_preds_exgal, 0.2)
def get_meta_preds(train_df, oof_preds, test_preds, C):
    lr = LogisticRegression(
        C=C, intercept_scaling=0.1, multi_class="multinomial", solver="lbfgs"
    )
    lr.fit(safe_log(oof_preds), train_df["target"],
           sample_weight=train_df["sample_weight"])
    return lr.predict_proba(safe_log(test_preds))


def safe_log(x):
    return np.log(np.clip(x, 1e-4, None))


def submit(test_df, test_preds_gal, test_preds_exgal, gal_class_list,
           exgal_class_list, sub_file):
    all_classes = [c for c in gal_class_list] + [c for c in exgal_class_list]

    gal_indices = np.where(test_df["hostgal_photoz"] == 0)[0]
    exgal_indices = np.where(test_df["hostgal_photoz"] > 0)[0]

    test_preds = np.zeros((test_df.shape[0], len(all_classes)))
    test_preds[gal_indices, :] = np.hstack((
        np.clip(test_preds_gal, 1e-4, None),
        np.zeros((test_preds_gal.shape[0], len(exgal_class_list)))
    ))
    test_preds[exgal_indices, :] = np.hstack((
        np.zeros((test_preds_exgal.shape[0], len(gal_class_list))),
        np.clip(test_preds_exgal, 1e-4, None)
    ))

    estimated99 = get_class99_proba(test_df, test_preds, all_classes)

    sub_df = pd.DataFrame(
        index=test_df['object_id'],
        data=np.round(test_preds * (1 - estimated99), 4),
        columns=['class_%d' % i for i in all_classes]
    )
    sub_df["class_99"] = estimated99

    sub_df.to_csv(sub_file)


def get_class99_proba(test_df, test_preds, all_classes):
    base = 0.02

    high99 = (get_hostgal_range(test_df["hostgal_photoz"]) == 0)

    low99 = is_labeled_as(test_preds, all_classes, 15)
    for label in [64, 67, 88, 90]:
        low99 = low99 | is_labeled_as(test_preds, all_classes, label)
    class99 = 0.22 - 0.18 * low99 + 0.13 * high99 - base

    not_sure = (test_preds.max(axis=1) < 0.9)
    filt = (test_df["hostgal_photoz"] > 0) & not_sure

    return (base + (class99 * filt).values).reshape(-1, 1)


def is_labeled_as(preds, class_list, label):
    return preds.argmax(axis=1) == np.where(np.array(class_list) == label)[0]


def draw_confusion_matrix(target, prediction, class_list, path):
    y_pred = np.argmax(prediction, axis=1)
    cm = confusion_matrix(y_true=target, y_pred=y_pred)

    cm = cm / np.sum(cm, axis=1, keepdims=True)
    annotation = np.around(cm, 2)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(cm, xticklabels=class_list, yticklabels=class_list,
                cmap='Blues', annot=annotation, lw=0.5, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_aspect('equal')
    fig.tight_layout()

    fig.savefig(path, bbox_inches='tight')
    plt.close()


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--feature-dir', type=click.Path(exists=True),
              default='../data/processed/4th')
@click.option('--meta-dir', type=click.Path(exists=True),
              default='../data/raw')
@click.option('--model-dir', type=click.Path())
def raw(feature_dir, meta_dir, model_dir):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    (train_gal, train_exgal, test_gal, test_exgal,
     gal_class_list, exgal_class_list, test_df) = load_data(
        feature_data_dir=feature_dir, meta_data_dir=meta_dir
    )
    (oof_preds_gal, oof_preds_exgal,
     test_preds_gal, test_preds_exgal) = get_lgb_predictions(
        train_gal, train_exgal, test_gal, test_exgal,
        gal_class_list=gal_class_list, exgal_class_list=exgal_class_list,
        model_dir=model_dir
    )

    test_preds_gal = get_meta_preds(
        train_gal, oof_preds_gal, test_preds_gal, 0.2
    )
    test_preds_exgal = get_meta_preds(
        train_exgal, oof_preds_exgal, test_preds_exgal, 0.2
    )

    submit(test_df, test_preds_gal, test_preds_exgal,
           gal_class_list, exgal_class_list,
           str(model_dir / "submission_lgb.csv"))

    tmp = np.copy(train_gal['target'].values)

    draw_confusion_matrix(
        target=tmp, prediction=oof_preds_gal,
        class_list=gal_class_list,
        path=model_dir / 'confusion_matrix_galactic.png'
    )
    draw_confusion_matrix(
        target=train_exgal['target'].values, prediction=oof_preds_exgal,
        class_list=exgal_class_list,
        path=model_dir / 'confusion_matrix_extra_galactic.png'
    )


@click.command()
@click.option('--data-dir', type=click.Path(exists=True),
              default='../data/interim/gp2d')
@click.option('--model-dir', type=click.Path(),
              default='../models/gp2d-feature')
@click.option('--v2', is_flag=True)
def augmented(data_dir, model_dir, v2):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    if v2:
        (df_galactic, galactic_class_list,
         df_extra, extra_class_list) = load_data_v2(
            data_dir=data_dir,
            meta_path='../data/raw/training_set_metadata.csv'
        )
    else:
        (df_galactic, galactic_class_list,
         df_extra, extra_class_list) = load_data(data_dir=data_dir)

    oof_preds_gal, oof_preds_exgal = get_lgb_predictions(
        df_galactic=df_galactic, galactic_class_list=galactic_class_list,
        df_extra=df_extra, extra_class_list=extra_class_list)

    evaluate(df_galactic, df_extra, oof_preds_gal, oof_preds_exgal, model_dir)
    tmp = np.copy(df_galactic['target'].values)
    draw_confusion_matrix(
        target=tmp, prediction=oof_preds_gal.values,
        class_list=galactic_class_list,
        path=model_dir / 'confusion_matrix_galactic.png'
    )
    draw_confusion_matrix(
        target=df_extra['target'].values, prediction=oof_preds_exgal.values,
        class_list=extra_class_list,
        path=model_dir / 'confusion_matrix_extra_galactic.png'
    )

    oof_preds_gal.to_pickle(model_dir / 'predictions_galactic.pickle')
    oof_preds_exgal.to_pickle(model_dir / 'predictions_extra_galactic.pickle')


def main():
    cmd()


if __name__ == '__main__':
    main()
