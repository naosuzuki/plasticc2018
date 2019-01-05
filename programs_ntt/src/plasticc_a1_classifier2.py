#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
random forestとかで文不意してみる
"""

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm

from plasticc_a1_classifier1 import convert_data, Dataset

__author__ = 'Yasuhiro Imoto'
__date__ = '20/11/2018'


def load_train_data(train_data_path, use_hostgal=True):
    dataset, (label_map, count) = convert_data(train_data_path)

    tmp = train_test_split(
        *list(dataset), test_size=0.2, random_state=0, stratify=dataset.target
    )

    # noinspection PyProtectedMember
    train_values = Dataset._make(tmp[0::2])
    # noinspection PyProtectedMember
    test_values = Dataset._make(tmp[1::2])

    np.random.seed(42)

    if use_hostgal:
        keys = ('specz', 'photoz', 'photoz_err')
        train_additional_inputs = {
            key: getattr(train_values, key) for key in keys
        }
        test_additional_inputs = {
            key: getattr(test_values, key) for key in keys
        }
    else:
        train_additional_inputs = None
        test_additional_inputs = None

    train_data = make_data(
        flux=train_values.flux, flux_err=train_values.flux_err,
        target=train_values.target, repeat=4,
        additional_inputs=train_additional_inputs
    )
    test_data = make_data(
        flux=test_values.flux, flux_err=test_values.flux_err,
        target=test_values.target, repeat=0,
        additional_inputs=test_additional_inputs
    )

    return train_data, test_data, (label_map, count)


def make_data(flux, flux_err, target, repeat, additional_inputs=None):
    if additional_inputs is None:
        x = np.arcsinh(flux * 0.5).astype(np.float32)
    else:
        specz = additional_inputs['specz']
        r = np.random.normal(size=additional_inputs['photoz_err'].shape)
        photoz = (additional_inputs['photoz'] +
                  additional_inputs['photoz_err'] * r)

        x = np.concatenate(
            [np.arcsinh(flux * 0.5), specz, photoz], axis=1
        ).astype(np.float32)

    if repeat == 0:
        return x, target

    x, y = [x], [target]
    for i in range(repeat):
        tmp = flux + flux_err * np.random.normal(size=flux_err.shape)
        mag = np.arcsinh(tmp * 0.5)

        if additional_inputs is None:
            x.append(mag)
        else:
            r = np.random.normal(size=additional_inputs['photoz_err'].shape)
            photoz = (additional_inputs['photoz'] +
                      additional_inputs['photoz_err'] * r)

            # noinspection PyUnboundLocalVariable
            x.append(np.concatenate([mag, specz, photoz], axis=1))
        y.append(target)

    x = np.concatenate(x, axis=0).astype(np.float32)
    y = np.concatenate(y, axis=0)

    return x, y


def split_data(dataset, train_index, test_index, use_hostgal, repeat):
    if use_hostgal:
        keys = ('specz', 'photoz', 'photoz_err')
        train_additional_inputs = {
            key: getattr(dataset, key)[train_index] for key in keys
        }
        test_additional_inputs = {
            key: getattr(dataset, key)[test_index] for key in keys
        }
    else:
        train_additional_inputs = None
        test_additional_inputs = None

    train_x, train_y = make_data(
        flux=dataset.flux[train_index],
        flux_err=dataset.flux_err[train_index],
        target=dataset.target[train_index],
        repeat=repeat,
        additional_inputs=train_additional_inputs
    )
    test_x, test_y = make_data(
        flux=dataset.flux[test_index],
        flux_err=dataset.flux_err[test_index],
        target=dataset.target[test_index], repeat=0,
        additional_inputs=test_additional_inputs
    )

    return (train_x, train_y), (test_x, test_y)


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--data-path', type=click.Path(exists=True))
@click.option('--max-depth', type=int, default=None)
@click.option('--n-estimators', type=int, default=1000)
@click.option('--criterion', type=click.Choice(['gini', 'entropy']),
              default='gini')
@click.option('--seed', type=int, default=0)
@click.option('--model-dir', type=click.Path())
@click.option('--hostgal', is_flag=True)
@click.option('--cv', type=int, default=0)
@click.option('--augmented', type=int, default=9)
def random_forest(data_path, max_depth, n_estimators, criterion, seed,
                  model_dir, hostgal, cv, augmented):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    parameters = {
        'data': {'path': data_path, 'hostgal': hostgal,
                 'augmented': augmented},
        'max_depth': max_depth, 'n_estimators': n_estimators,
        'seed': seed, 'cv': cv
    }
    with (model_dir / 'parameters.json').open('w') as f:
        json.dump(parameters, f, sort_keys=True, indent=4)

    if cv > 0:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        dataset, (label, count) = convert_data(data_path=data_path)

        print(label)
        print(count)

        train_score_list, test_score_list = [], []
        for i, (train_index, test_index) in enumerate(
                tqdm(skf.split(dataset.target, dataset.target), total=cv)):
            (train_x, train_y), (test_x, test_y) = split_data(
                dataset=dataset, train_index=train_index,
                test_index=test_index, use_hostgal=hostgal, repeat=augmented
            )

            train_score, test_score = random_forest_cv(
                train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
                max_depth=max_depth, n_estimators=n_estimators, seed=seed,
                model_dir=model_dir, cv_index=i, criterion=criterion
            )
            train_score_list.append(train_score)
            test_score_list.append(test_score)

        with (model_dir / 'score.json').open('w') as f:
            json.dump({'train': train_score_list, 'test': test_score_list}, f,
                      indent=4, sort_keys=True)

        return

    model = RandomForestClassifier(
        n_estimators=n_estimators, criterion=criterion, random_state=seed,
        n_jobs=8, max_depth=max_depth
    )

    # noinspection PyTypeChecker
    train_model(model=model, data_path=data_path, use_hostgal=hostgal,
                model_dir=model_dir)


def random_forest_cv(train_x, train_y, test_x, test_y, max_depth, criterion,
                     n_estimators, seed, model_dir, cv_index):
    model = RandomForestClassifier(
        n_estimators=n_estimators, criterion=criterion, random_state=seed,
        n_jobs=8, max_depth=max_depth
    )

    # noinspection PyTypeChecker
    train_score, test_score = train_model_cv(
        model=model, train_x=train_x, train_y=train_y,
        test_x=test_x, test_y=test_y, model_dir=model_dir, cv_index=cv_index
    )

    return train_score, test_score


def train_model_cv(model, train_x, train_y, test_x, test_y, model_dir,
                   cv_index):
    model.fit(train_x, train_y)
    train_score = model.score(train_x, train_y)
    test_score = model.score(test_x, test_y)

    train_prediction = model.predict(train_x)
    train_cm = confusion_matrix(y_true=train_y, y_pred=train_prediction)

    test_prediction = model.predict(test_x)
    test_cm = confusion_matrix(y_true=test_y, y_pred=test_prediction)

    np.savetxt(str(model_dir / 'train_matrix{}.txt'.format(cv_index)),
               train_cm, fmt='%3d')
    np.savetxt(str(model_dir / 'test_matrix{}.txt'.format(cv_index)),
               test_cm, fmt='%3d')

    joblib.dump(model, str(model_dir / 'model{}.pickle'.format(cv_index)))

    return train_score, test_score


def train_model(model, data_path, use_hostgal, model_dir):
    train_data, test_data, (label, count) = load_train_data(
        data_path, use_hostgal=use_hostgal
    )
    print(label)
    print(count)

    model.fit(train_data[0], train_data[1])
    train_score = model.score(train_data[0], train_data[1])
    test_score = model.score(test_data[0], test_data[1])

    with (model_dir / 'score.json').open('w') as f:
        json.dump({'train': train_score, 'test': test_score}, f,
                  sort_keys=True, indent=4)

    train_prediction = model.predict(train_data[0])
    train_cm = confusion_matrix(y_true=train_data[1], y_pred=train_prediction)

    test_prediction = model.predict(test_data[0])
    test_cm = confusion_matrix(y_true=test_data[1], y_pred=test_prediction)

    np.savetxt(str(model_dir / 'train_matrix.txt'), train_cm, fmt='%3d')
    np.savetxt(str(model_dir / 'test_matrix.txt'), test_cm, fmt='%3d')

    joblib.dump(model, str(model_dir / 'model.pickle'))

    train_prediction = model.predict_proba(train_data[0])
    test_prediction = model.predict_proba(test_data[0])
    np.savez_compressed(str(model_dir / 'prediction.npz'),
                        train=train_prediction, test=test_prediction,
                        train_label=train_data[1], test_label=test_data[1])


@cmd.command()
@click.option('--data-path', type=click.Path(exists=True))
@click.option('--max-depth', type=int, default=3)
@click.option('--n-estimators', type=int, default=1000)
@click.option('--seed', type=int, default=0)
@click.option('--model-dir', type=click.Path())
@click.option('--hostgal', is_flag=True)
@click.option('--cv', type=int, default=0)
@click.option('--max-delta-step', type=int, default=0)
@click.option('--augmented', type=int, default=9)
def xgboost(data_path, max_depth, n_estimators, seed, model_dir, hostgal, cv,
            max_delta_step, augmented):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    parameters = {
        'data': {'path': data_path, 'hostgal': hostgal,
                 'augmented': augmented},
        'max_depth': max_depth, 'n_estimators': n_estimators,
        'seed': seed, 'cv': cv, 'max_delta_step': max_delta_step
    }
    with (model_dir / 'parameters.json').open('w') as f:
        json.dump(parameters, f, sort_keys=True, indent=4)

    if cv > 0:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        dataset, (label, count) = convert_data(data_path=data_path)

        print(label)
        print(count)

        train_score_list, test_score_list = [], []
        for i, (train_index, test_index) in enumerate(
                tqdm(skf.split(dataset.target, dataset.target), total=cv)):
            (train_x, train_y), (test_x, test_y) = split_data(
                dataset=dataset, train_index=train_index,
                test_index=test_index, use_hostgal=hostgal, repeat=augmented
            )

            train_score, test_score = xgboost_cv(
                train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
                max_depth=max_depth, n_estimators=n_estimators, seed=seed,
                model_dir=model_dir, cv_index=i, max_delta_step=max_delta_step
            )
            train_score_list.append(train_score)
            test_score_list.append(test_score)

        with (model_dir / 'score.json').open('w') as f:
            json.dump({'train': train_score_list, 'test': test_score_list}, f,
                      indent=4, sort_keys=True)

        return

    model = xgb.XGBClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=8,
        max_depth=max_depth
    )

    train_model(model=model, data_path=data_path, use_hostgal=hostgal,
                model_dir=model_dir)


def xgboost_cv(train_x, train_y, test_x, test_y, max_depth, n_estimators,
               seed, model_dir, cv_index, max_delta_step):
    model = xgb.XGBClassifier(
        n_estimators=n_estimators, random_state=seed + cv_index, n_jobs=8,
        max_depth=max_depth, max_delta_step=max_delta_step
    )

    train_score, test_score = train_model_cv(
        model=model, train_x=train_x, train_y=train_y,
        test_x=test_x, test_y=test_y, model_dir=model_dir, cv_index=cv_index
    )

    return train_score, test_score


@cmd.command()
@click.option('--data-path', type=click.Path(exists=True))
@click.option('--model-dir', type=click.Path(exists=True))
def xgboost_prediction(data_path, model_dir):
    model_dir = Path(model_dir)

    with (model_dir / 'parameters.json').open('r') as f:
        parameters = json.load(f)

    dataset, _ = convert_data(data_path)
    if parameters['data']['hostgal']:
        keys = ('specz', 'photoz', 'photoz_err')
        additional_inputs = {
            key: getattr(dataset, key) for key in keys
        }
    else:
        additional_inputs = None
    x, _ = make_data(
        flux=dataset.flux, flux_err=dataset.flux_err, target=dataset.target,
        repeat=0, additional_inputs=additional_inputs
    )

    cv = parameters['cv']
    if cv == 0:
        model = joblib.load(str(model_dir / 'model.pickle'))

        prediction = model.predict_proba(x)

        df = pd.DataFrame(data=prediction)
        df.to_csv(model_dir / 'prediction.csv')
    else:
        for i in range(cv):
            model = joblib.load(str(model_dir / 'model{}.pickle'.format(i)))

            prediction = model.predict_proba(x)

            df = pd.DataFrame(data=prediction)
            df.to_csv(model_dir / 'prediction{}.csv'.format(i))


def main():
    cmd()


if __name__ == '__main__':
    main()
