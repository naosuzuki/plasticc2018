#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product
from time import sleep
from subprocess import call

__author__ = 'Yasuhiro Imoto'
__date__ = '22/2/2018'


def write_job(method, dataset, file_name):
    data_fmt = ('/home/imoto/crest_auto/data/processed/180206/{}/train/'
                'dataset.{{}}-2classes.nc').format(dataset)
    model_dir = ('/home/imoto/crest_auto/models/180206/redshift/{}/hyperopt/'
                 '{}/2classes/without_redshift').format(dataset, method)
    options = [
        '--epoch=500',
        '--n_iterations=101',
        r'--band_data=\{\"i\":3,\"z\":3\}',
        '--method={}'.format(method),
        '--train_data_path={}'.format(data_fmt.format('tr')),
        '--validation_data_path={}'.format(data_fmt.format('va')),
        '--test_data_path={}'.format(data_fmt.format('te')),
        '--output_dir={}'.format(model_dir),
        '--hostname=ks000',
        '--port=1234',
        '--db_name=db20180206'
    ]

    slurm_name = (
        "/home/imoto/crest_auto/src/models/slurm/hyperopt/redshift"
        "/%j-{dataset}-{method}.out"
    ).format(dataset=dataset, method=method)

    job = """#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -o "{slurm}"

hostname
source activate py3.5
echo {options}
export http_proxy=http://proxy-u.ecl.ntt.co.jp:8080/
export https_proxy=http://proxy-u.ecl.ntt.co.jp:8080/
export PYTHONPATH='/home/imoto/crest_auto/src/models'
python train_model.py redshift search_parallel {options}
""".format(options=' '.join(options), slurm=slurm_name)

    with open(file_name, 'w') as f:
        f.write(job)


def main():
    method_list = ['modified', 'traditional']
    dataset_list = ['dataset_all', 'dataset_selected']

    job_dir = 'jobs/redshift'
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)
    slurm_dir = 'slurm/hyperopt/redshift'
    if not os.path.exists(slurm_dir):
        os.makedirs(slurm_dir)

    for method, dataset in product(method_list, dataset_list):
        file_name = os.path.join(job_dir, '{}_{}.sh'.format(dataset, method))

        write_job(method=method, dataset=dataset, file_name=file_name)
        call(['sbatch', file_name])
        sleep(1)


if __name__ == '__main__':
    main()
