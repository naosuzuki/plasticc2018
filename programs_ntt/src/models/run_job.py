#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product
from time import sleep
from subprocess import call

__author__ = 'Yasuhiro Imoto'
__date__ = '14/12/2017'


def write_job(band, mode, file_name):
    options = [
        '--mode={}'.format(mode),
        '--data_name={}'.format(band)
    ]
    if mode == 'exact':
        options.append('--beta=0.05')

    job = """#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source activate py3.5
echo Gpu device: $CUDA_VISIBLE_DEVICES
echo {options}
export PYTHONPATH='/home/imoto/crest-auto/src/models'
python search_pauc_parameter.py {options}
""".format(options=' '.join(options))

    with open(file_name, 'w') as f:
        f.write(job)


def main():
    band_list = ['HSC-G', 'HSC-I2', 'HSC-R2', 'HSC-Y', 'HSC-Z']
    mode_list = ['relaxed', 'exact']

    if not os.path.exists('jobs'):
        os.makedirs('jobs')

    for band, mode in product(band_list, mode_list):
        file_name = 'jobs/{}_{}.sh'.format(band, mode)

        write_job(band, mode, file_name)
        call(['sbatch', file_name])
        sleep(1)


if __name__ == '__main__':
    main()
