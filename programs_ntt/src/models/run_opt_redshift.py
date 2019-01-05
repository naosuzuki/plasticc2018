#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product
from time import sleep
from subprocess import call

__author__ = 'Yasuhiro Imoto'
__date__ = '22/2/2018'


def write_job(method, dataset, file_name):
    data_dir = '/home/imoto/crest_auto/data/processed/180206'
    data_fmt = (
        '{data_dir}/{dataset}/train/dataset.{{}}-2classes.nc'
    ).format(data_dir=data_dir, dataset=dataset)
    model_dir = ('/home/imoto/crest_auto/models/180206/{}/hyperopt/'
                 '{}/2classes/without_redshift').format(dataset, method)
    parameter_path = os.path.join(model_dir, 'best_parameter.json')

    train_options = [
        '--epoch=500',
        r'--band_data=\{\"i\":3,\"z\":3\}',
        '--method={}'.format(method),
        '--train_data_path={}'.format(data_fmt.format('tr')),
        '--validation_data_path={}'.format(data_fmt.format('va')),
        '--test_data_path={}'.format(data_fmt.format('te')),
        '--output_dir={}'.format(model_dir),
        '--parameter_path={}'.format(parameter_path)
    ]

    real_data_path = os.path.join(
        data_dir, 'test', 'dataset.test.all-2classes.nc'
    )
    test_options = [
        '--train_data_path={}'.format(data_fmt.format('tr')),
        '--validation_data_path={}'.format(data_fmt.format('va')),
        '--test_data_path={}'.format(data_fmt.format('te')),
        '--real_data_path={}'.format(real_data_path),
        '--output_dir={}'.format(model_dir)
    ]

    slurm_name = (
        "/home/imoto/crest_auto/src/models/slurm/optimization/redshift"
        "/%j-{dataset}-{method}.out"
    ).format(dataset=dataset, method=method)

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
    dataset_list = ['dataset_all', 'dataset_selected']

    if not os.path.exists('jobs/optimization/redshift'):
        os.makedirs('jobs/optimization/redshift')
    if not os.path.exists('slurm/optimization/redshift'):
        os.makedirs('slurm/optimization/redshift')

    for method, dataset in product(method_list, dataset_list):
        file_name = 'jobs/optimization/redshift/{}_{}.sh'.format(
            dataset, method)

        write_job(method=method, dataset=dataset, file_name=file_name)
        call(['sbatch', file_name])
        sleep(1)


if __name__ == '__main__':
    main()
