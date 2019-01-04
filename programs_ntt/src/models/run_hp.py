#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product
from time import sleep
from subprocess import call

__author__ = 'Yasuhiro Imoto'
__date__ = '12/1/2018'


def write_job(output_size, method, redshift, dataset, file_name):
    if redshift:
        tmp = 'with'
    else:
        tmp = 'without'

    data_fmt = ('/home/imoto/crest_auto/data/processed/180206/{}/train/'
                'dataset.{{}}-{}classes.nc').format(dataset, output_size)
    model_dir = ('/home/imoto/crest_auto/models/180206/{}/hyperopt/'
                 '{}/{}classes/{}_redshift').format(dataset, method,
                                                    output_size, tmp)
    options = [
        '--output_size={}'.format(output_size),
        '--epoch=500',
        '--n_iterations=100',
        r'--band_data=\{\"i\":3,\"z\":3\}',
        '--method={}'.format(method),
        '--train_data_path={}'.format(data_fmt.format('tr')),
        '--validation_data_path={}'.format(data_fmt.format('va')),
        '--test_data_path={}'.format(data_fmt.format('te')),
        '--output_dir={}'.format(model_dir)
    ]
    if redshift:
        options.append('--use_redshift')

    slurm_name = (
        "/home/imoto/crest_auto/src/models/slurm/hyperopt"
        "/%j-{dataset}-{method}-{output_size}-{redshift}.out"
    ).format(dataset=dataset, method=method, output_size=output_size,
             redshift=redshift)

    job = """#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -o "{slurm}"

hostname
source activate py3.5
echo {options}
export http_proxy=http://proxy-u.ecl.ntt.co.jp:8080/
export https_proxy=http://proxy-u.ecl.ntt.co.jp:8080/
export PYTHONPATH='/home/imoto/crest_auto/src/models'
python train_model.py search_parallel {options}
""".format(options=' '.join(options), slurm=slurm_name)

    with open(file_name, 'w') as f:
        f.write(job)


def main():
    method_list = ['modified', 'traditional']
    redshift_flag = [True, False]
    output_size_list = [2, 6]
    dataset_list = ['dataset_all', 'dataset_selected']

    if not os.path.exists('jobs'):
        os.makedirs('jobs')
    if not os.path.exists('slurm/hyperopt'):
        os.makedirs('slurm/hyperopt')

    for method, redshift, output_size, dataset in product(
            method_list, redshift_flag, output_size_list, dataset_list):
        file_name = 'jobs/{}_{}_{}_{}.sh'.format(
            dataset, method, redshift, output_size)

        write_job(output_size=output_size, method=method, redshift=redshift,
                  dataset=dataset, file_name=file_name)
        call(['sbatch', file_name])
        sleep(1)


if __name__ == '__main__':
    main()
