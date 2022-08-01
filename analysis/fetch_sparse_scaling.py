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
NUM_SETTINGS = list(range(10,17))

OURS_PROJ = "MT-ToyTask-Ours"

SPARSE_JOB_ID = "9820227" #NOTE <-- change this for new runs

SPARSE_SKIP = [(16, 4), (15, 3), (15, 0), (14, 2)]

for loss in ["tuning_loss", "value_prefix_loss", "dyn_loss"]:
    print(f'Loading {loss} data...')
    sparse_dir = f"./data/sparse/{SPARSE_JOB_ID}/{loss}"
    download_and_save(
        api,
        ENT,
        sparse_dir, 
        SPARSE_JOB_ID, 
        NUM_SETTINGS,
        NUM_SEEDS,
        OURS_PROJ,
        loss,
        SPARSE_SKIP,
        prefix="sparse",
        setting_suffix="_128",
)