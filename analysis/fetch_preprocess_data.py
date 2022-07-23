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
# LOSS = "tuning_loss"
LOSS = "value_prefix_loss"

DREAMER_PROJ = "MT-ToyTask-Dreamer"
OURS_PROJ = "MT-ToyTask-Ours"

DREAMER_JOB_ID = "9749161" #NOTE <-- change this for new runs
OURS_JOB_ID = "9770135" #NOTE <-- change this for new runs

dreamer_dir = f"./data/dreamer/{DREAMER_JOB_ID}/{LOSS}"
ours_dir = f"./data/ours/{OURS_JOB_ID}/{LOSS}"

# print('Loading dreamer data...')
# if not os.path.exists(dreamer_dir):
#     os.makedirs(dreamer_dir, exist_ok=True)
#     dreamer_mean_std_per_setting = compute_mean_std_per_setting(fetch_loss_list_per_setting(api, DREAMER_JOB_ID, NUM_SETTINGS, NUM_SEEDS, DREAMER_PROJ, LOSS, ENT))
#     with open(f"{dreamer_dir}/mean_std_per_setting.json", "w") as f:
#         json.dump(dreamer_mean_std_per_setting, f)
# else:
#     print('Directory already exists -> skipping...')


print('Loading ours data...')
if not os.path.exists(ours_dir):
    os.makedirs(ours_dir, exist_ok=True)
    ours_mean_std_per_setting = compute_mean_std_per_setting(fetch_loss_list_per_setting(api, OURS_JOB_ID, NUM_SETTINGS, NUM_SEEDS, OURS_PROJ, LOSS, ENT))
    with open(f"{ours_dir}/mean_std_per_setting.json", "w") as f:
        json.dump(ours_mean_std_per_setting, f)
else:
    print('Directory already exists -> skipping...')