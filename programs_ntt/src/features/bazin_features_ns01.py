#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import numba
import click

__author__ = 'Yasuhiro Imoto'
__date__ = '26/12/2018'


NUM_PARTITIONS = 500
LOW_PASSBAND_LIMIT = 3
FEATURES = ["A", "B", "t0", "tfall", "trise", "cc", "fit_error", "status",
            "t0_shift"]


# bazin, errorfunc and fit_scipy are developed using:
# https://github.com/COINtoolbox/ActSNClass/blob/master/examples/1_fit_LC/fit_lc_parametric.py
@numba.jit(nopython=True)
def bazin(time, low_passband, A, B, t0, tfall, trise, cc):
    X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))
    return (A * X + B) * (1 - cc * low_passband)


# @numba.jit(nopython=True)
def errfunc(params, time, low_passband, flux, weights):
    return abs(flux - bazin(time, low_passband, *params)) * weights


def fit_scipy(time, low_passband, flux, flux_err):
    time -= time[0]
    sn = np.power(flux / flux_err, 2)
    start_point = (sn * flux).argmax()

    t0_init = time[start_point] - time[0]
    amp_init = flux[start_point]
    weights = 1 / (1 + flux_err)
    weights = weights / weights.sum()
    guess = [0, amp_init, t0_init, 40, -5, 0.5]

    result = least_squares(
        errfunc, guess, args=(time, low_passband, flux, weights), method='lm'
    )
    # noinspection PyUnresolvedReferences
    result.t_shift = t0_init - result.x[2]

    return result


def yield_data_augmented(store):
    for key in store.keys():
        if key == '/redshift':
            continue
        value = store[key]
        object_id = value['object_id'].values[0]
        sub_id = value['sub_id'].values[0]
        yield object_id, sub_id, value


def get_params_augmented(object_id_list, lc_df, result_queue):
    results = {}
    for object_id in object_id_list:
        light_df = lc_df[lc_df["object_id"] == object_id]
        sub_id = light_df['sub_id'].values[0]
        try:
            result = fit_scipy(
                light_df["mjd"].values, light_df["low_passband"].values,
                light_df["flux"].values, light_df["flux_err"].values
            )
            # noinspection PyUnresolvedReferences
            results[(object_id, sub_id)] = np.append(
                result.x, [result.cost, result.status, result.t_shift]
            )
        except Exception as e:
            print(e)
            results[object_id] = None
    result_queue.put(results)


def parallelize_augmented(hdf5_path):
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size)

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    n = 0
    with pd.HDFStore(hdf5_path) as store:
        for object_id, sub_id, df in yield_data_augmented(store):
            pool.apply_async(
                get_params_augmented, (object_id, sub_id, df, result_queue)
            )
            n += 1

        pool.close()
        pool.join()

    tmp = [result_queue.get() for _ in range(n)]
    data = np.vstack(tmp)
    df = pd.DataFrame(data, columns=FEATURES + ['sub_id'])
    return df


def yield_data(meta_df, lc_df):
    cols = ["object_id", "mjd", "flux", "flux_err", "low_passband"]
    for i in range(NUM_PARTITIONS):
        yield meta_df[(meta_df["object_id"] % NUM_PARTITIONS) == i]["object_id"].values, \
              lc_df[(lc_df["object_id"] % NUM_PARTITIONS) == i][cols]


def get_params(object_id_list, lc_df, result_queue):
    results = {}
    for object_id in object_id_list:
        light_df = lc_df[lc_df["object_id"] == object_id]
        try:
            result = fit_scipy(
                light_df["mjd"].values, light_df["low_passband"].values,
                light_df["flux"].values, light_df["flux_err"].values
            )
            # noinspection PyUnresolvedReferences
            results[object_id] = np.append(
                result.x, [result.cost, result.status, result.t_shift]
            )
        except Exception as e:
            print(e)
            results[object_id] = None
    result_queue.put(results)


def parallelize(meta_df, df):
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size)

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    for m, d in yield_data(meta_df, df):
        pool.apply_async(get_params, (m, d, result_queue))

    pool.close()
    pool.join()

    return [result_queue.get() for _ in range(NUM_PARTITIONS)]


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--data-path', type=click.Path(exists=True),
              default='../../data/raw/test_set.csv')
@click.option('--meta-path', type=click.Path(exists=True),
              default='../../data/raw/test_set_metadata.csv')
@click.option('--output-path', type=click.Path(),
              default='../../data/processed/4th/test_set_bazin_features.pickle'
              )
def raw(data_path, meta_path, output_path):
    output_path = Path(output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    meta_df = pd.read_csv(meta_path, header=0,\
              dtype={'object_id':np.uint32,'ra':np.float16,'decl':np.float16,\
                     'gal_l':np.float16,'gal_b':np.float16,\
                     'hostgal_specz':np.float16,'hostgal_photoz':np.float16,'hostgal_photoz_err':np.float16,\
                     'distmod':np.float16,'mwebv':np.float16,'target':np.uint8})
    df = pd.read_csv(data_path, header=0,dtype={'objec_id':np.uint32,'mjd':np.float16,\
                      'passband':np.uint8,'flux':np.float16,'flux_err':np.float16,'detected':np.uint8})

    df['low_passband'] = (df['passband'] < LOW_PASSBAND_LIMIT).astype(int)

    result_list = parallelize(meta_df=meta_df, df=df)
    final_result = {}
    for res in result_list:
        final_result.update(res)

    for index, col in enumerate(FEATURES):
        meta_df[col] = meta_df["object_id"].apply(
            lambda x: final_result[x][index])

    # meta_df[["object_id"] + FEATURES].to_csv(output_path, index=False)
    out_df = meta_df[["object_id"] + FEATURES]
    out_df.to_pickle(output_path)


@cmd.command()
@click.option('--data-dir', type=click.Path(exists=True),
              default='../../data/interim/augmented')
@click.option('--target', type=click.Choice([
    '15', '42', '52', '62', '64', '67', '88', '90', '95'
]))
def augmented(data_dir, target):
    data_dir = Path(data_dir)
    target = int(target)

    df = parallelize_augmented(data_dir / 'data{}_00.h5'.format(target))
    df.to_pickle(data_dir / 'bazin{}_00.pickle'.format(target))


def main():
    cmd()


if __name__ == '__main__':
    main()
