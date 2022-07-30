# This file downloads the run data of the specified job ids for both settings
# It then computes mean and std for each setting over all seeds and saves 
# the results in a json file.

import wandb
import json
import os
import sys
sys.path.append('../')

from analysis.utils import fetch_loss_list_per_setting, compute_mean_std_per_setting

api = wandb.Api()
ENT = "TomFrederik"
NUM_SEEDS = 5
NUM_SETTINGS = 21
LOSS = "value_prefix_loss"

DREAMER_PROJ = "MT-ToyTask-Dreamer"
OURS_PROJ = "MT-ToyTask-Ours"

DREAMER_JOB_ID = "9801838" #NOTE <-- change this for new runs
DENSE_JOB_ID = "9770135" #NOTE <-- change this for new runs
SPARSE_JOB_ID = "9807092" #NOTE <-- change this for new runs

SPARSE_SKIP = [(20, 3), (13, 3), (6, 0), (6, 1), (4, 0), (4, 1)]
DENSE_SKIP = [(0, 2), (1, 2), (2, 1), (3, 2), (4, 0), (5, 2)]

dreamer_dir = f"./data/dreamer/{DREAMER_JOB_ID}/{LOSS}"
dense_dir = f"./data/dense/{DENSE_JOB_ID}/{LOSS}"
sparse_dir = f"./data/sparse/{SPARSE_JOB_ID}/{LOSS}"

print('Loading dreamer data...')
if not os.path.exists(dreamer_dir):
    os.makedirs(dreamer_dir, exist_ok=True)
    dreamer_mean_std_per_setting = compute_mean_std_per_setting(fetch_loss_list_per_setting(api, DREAMER_JOB_ID, NUM_SETTINGS, NUM_SEEDS, DREAMER_PROJ, LOSS, ENT))
    with open(f"{dreamer_dir}/mean_std_per_setting.json", "w") as f:
        json.dump(dreamer_mean_std_per_setting, f)
else:
    print('Directory already exists -> skipping...')


print('Loading dense data...')
if not os.path.exists(dense_dir):
    os.makedirs(dense_dir, exist_ok=True)
    ours_mean_std_per_setting = compute_mean_std_per_setting(fetch_loss_list_per_setting(api, DENSE_JOB_ID, NUM_SETTINGS, NUM_SEEDS, OURS_PROJ, LOSS, ENT, skip_tuples=DENSE_SKIP))
    with open(f"{dense_dir}/mean_std_per_setting.json", "w") as f:
        json.dump(ours_mean_std_per_setting, f)
else:
    print('Directory already exists -> skipping...')
    

print('Loading sparse data...')
if not os.path.exists(sparse_dir):
    os.makedirs(sparse_dir, exist_ok=True)
    ours_mean_std_per_setting = compute_mean_std_per_setting(fetch_loss_list_per_setting(api, SPARSE_JOB_ID, NUM_SETTINGS, NUM_SEEDS, OURS_PROJ, LOSS, ENT, skip_tuples=SPARSE_SKIP))
    with open(f"{sparse_dir}/mean_std_per_setting.json", "w") as f:
        json.dump(ours_mean_std_per_setting, f)
else:
    print('Directory already exists -> skipping...')