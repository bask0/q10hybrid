#!/usr/bin/env bash

source /Net/Groups/BGI/people/bkraft/anaconda3/etc/profile.d/conda.sh
conda activate q10hybrid
cd /Net/Groups/BGI/people/bkraft/git/q10hybrid/

CUDA_VISIBLE_DEVICES="$1" python experiments/experiment_03.py "${@:2}"
