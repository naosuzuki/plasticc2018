#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product
from time import sleep
from subprocess import call

__author__ = 'Yasuhiro Imoto'
__date__ = '15/2/2018'


def write_job(output_size, method, redshift, dataset, file_name):
    if redshift:
        tmp = 'with'
    else:
        tmp = 'without'

    data_dir = '/home/imoto/crest_auto/data/processed/180206'
    data_fmt = (
        '{data_dir}/{dataset}/train/dataset.{{}}-{output_size}classes.nc'
    ).format(data_dir=data_dir, dataset=dataset, output_size=output_size)
    model_dir = ('/home/imoto/crest_auto/models/180206/{}/hyperopt/'
                 '{}/{}classes/{}_redshift').format(dataset, method,
                                                    output_size, tmp)
    parameter_path = os.path.join(model_dir, 'best_parameter.json')

    train_options = [
        '--output_size={}'.format(output_size),
        '--epoch=500',
        r'--band_data=\{\"i\":3,\"z\":3\}',
        '--method={}'.format(method),
        '--train_data_path={}'.format(data_fmt.format('tr')),
        '--validation_data_path={}'.format(data_fmt.format('va')),
        '--test_data_path={}'.format(data_fmt.format('te')),
        '--output_dir={}'.format(model_dir),
        '--parameter_path={}'.format(parameter_path)
    ]
    if redshift:
        train_options.append('--use_redshift')

    real_data_path = os.path.join(
        data_dir, 'test', 'dataset.test.all-{}classes.nc'.format(output_size)
    )
    test_options = [
        '--train_data_path={}'.format(data_fmt.format('tr')),
        '--validation_data_path={}'.format(data_fmt.format('va')),
        '--test_data_path={}'.format(data_fmt.format('te')),
        '--real_data_path={}'.format(real_data_path),
        '--output_dir={}'.format(model_dir)
    ]

    slurm_name = (
        "/home/imoto/crest_auto/src/models/slurm/optimization"
        "/%j-{dataset}-{method}-{output_size}-{redshift}.out"
    ).format(dataset=dataset, method=method, output_size=output_size,
             redshift=redshift)

    job = """#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o "{slurm}"

hostname
source activate py3.5
export http_proxy=http://proxy-u.ecl.ntt.co.jp:8080/
export https_proxy=http://proxy-u.ecl.ntt.co.jp:8080/
export PYTHONPATH='/home/imoto/crest_auto/src/models'
echo {train_options}
python train_model.py optimize {train_options}
echo {test_options}
python predict_model.py {test_options}
""".format(train_options=' '.join(train_options), slurm=slurm_name,
           test_options=' '.join(test_options))

    with open(file_name, 'w') as f:
        f.write(job)


def main():
    method_list = ['modified', 'traditional']
    redshift_flag = [True, False]
    output_size_list = [2, 6]
    dataset_list = ['dataset_all', 'dataset_selected']

    if not os.path.exists('jobs/optimization'):
        os.makedirs('jobs/optimization')
    if not os.path.exists('slurm/optimization'):
        os.makedirs('slurm/optimization')

    for method, redshift, output_size, dataset in product(
            method_list, redshift_flag, output_size_list, dataset_list):
        file_name = 'jobs/optimization/{}_{}_{}_{}.sh'.format(
            dataset, method, redshift, output_size)

        write_job(output_size=output_size, method=method, redshift=redshift,
                  dataset=dataset, file_name=file_name)
        call(['sbatch', file_name])
        sleep(1)


if __name__ == '__main__':
    main()
