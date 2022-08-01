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
NUM_SETTINGS = 21
LOSS = "dyn_loss"

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
download_and_save(
    api,
    ENT,
    dreamer_dir, 
    DREAMER_JOB_ID, 
    NUM_SETTINGS,
    NUM_SEEDS,
    DREAMER_PROJ,
    "kl_loss",
)

print('Loading dense data...')
download_and_save(
    api,
    ENT,
    dense_dir, 
    DENSE_JOB_ID, 
    NUM_SETTINGS,
    NUM_SEEDS,
    OURS_PROJ,
    LOSS,
    DENSE_SKIP
)

print('Loading sparse data...')
download_and_save(
    api,
    ENT,
    sparse_dir, 
    SPARSE_JOB_ID, 
    NUM_SETTINGS,
    NUM_SEEDS,
    OURS_PROJ,
    LOSS,
    SPARSE_SKIP
)