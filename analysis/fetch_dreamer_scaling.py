# This file downloads the run data of the specified job ids for both settings
# It then computes mean and std for each setting over all seeds and saves 
# the results in a json file.

import wandb
import json
import os
import sys
sys.path.append('../')

from analysis.utils import download_and_save

api = wandb.Api()
ENT = "TomFrederik"
NUM_SEEDS = 5
NUM_SETTINGS = [4, 8, 12, 16, 20, 24, 28]

PROJ = "MT-ToyTask-Dreamer"

JOB_ID = "9820228" #NOTE <-- change this for new runs


for loss in ["tuning_loss", "value_prefix_loss", "kl_loss"]:
    print(f'Loading {loss} data...')
    target_dir = f"./data/dreamer/{JOB_ID}/{loss}"
    download_and_save(
        api,
        ENT,
        target_dir, 
        JOB_ID, 
        NUM_SETTINGS,
        NUM_SEEDS,
        PROJ,
        loss,
        prefix="latent_scale",
)