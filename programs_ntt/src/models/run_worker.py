#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product
from time import sleep
from subprocess import call

__author__ = 'Yasuhiro Imoto'
__date__ = '13/2/2018'


def write_job(file_name):
    base_name = os.path.basename(file_name)
    name, _ = os.path.splitext(base_name)

    job = """#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o "/home/imoto/crest_auto/src/models/slurm/worker/%j-{name}.out"

hostname
source activate py3.5
echo Gpu device: $CUDA_VISIBLE_DEVICES
export http_proxy=http://proxy-u.ecl.ntt.co.jp:8080/
export https_proxy=http://proxy-u.ecl.ntt.co.jp:8080/
export PYTHONPATH='/home/imoto/crest_auto/src/models'
hyperopt-mongo-worker --mongo=ks000:1234/db20180206 --poll-interval=10
""".format(name=name)

    with open(file_name, 'w') as f:
        f.write(job)


def main():
    if not os.path.exists('jobs'):
        os.makedirs('jobs')
    if not os.path.exists('slurm/worker'):
        os.makedirs('slurm/worker')

    for i in range(4):
        file_name = 'jobs/worker{0:02d}.sh'.format(i)
        write_job(file_name=file_name)
        call(['sbatch', file_name])
        sleep(1)


if __name__ == '__main__':
    main()
