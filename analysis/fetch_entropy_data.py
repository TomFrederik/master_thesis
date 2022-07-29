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

OURS_PROJ = "MT-ToyTask-Ours"
DENSE_JOB_ID = "9770135" #NOTE <-- change this for new runs
DENSE_SKIP = [(0, 2), (1, 2), (2, 1), (3, 2), (4, 0), (5, 2)]

LOSS = "prior_entropy"
ours_dir = f"./data/dense/{DENSE_JOB_ID}/{LOSS}"

print(f'Loading {LOSS} data...')
if not os.path.exists(ours_dir):
    os.makedirs(ours_dir, exist_ok=True)
    ours_mean_std_per_setting = compute_mean_std_per_setting(fetch_loss_list_per_setting(api, DENSE_JOB_ID, NUM_SETTINGS, NUM_SEEDS, OURS_PROJ, LOSS, ENT, skip_tuples=DENSE_SKIP))
    with open(f"{ours_dir}/mean_std_per_setting.json", "w") as f:
        json.dump(ours_mean_std_per_setting, f)
else:
    print('Directory already exists -> skipping...')
    
    
LOSS = "posterior_entropy"
ours_dir = f"./data/dense/{DENSE_JOB_ID}/{LOSS}"
print(f'Loading {LOSS} data...')
if not os.path.exists(ours_dir):
    os.makedirs(ours_dir, exist_ok=True)
    ours_mean_std_per_setting = compute_mean_std_per_setting(fetch_loss_list_per_setting(api, DENSE_JOB_ID, NUM_SETTINGS, NUM_SEEDS, OURS_PROJ, LOSS, ENT, skip_tuples=DENSE_SKIP))
    with open(f"{ours_dir}/mean_std_per_setting.json", "w") as f:
        json.dump(ours_mean_std_per_setting, f)
else:
    print('Directory already exists -> skipping...')