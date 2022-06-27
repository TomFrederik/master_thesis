import json
import os
from argparse import ArgumentParser
import time

from train_hmm import main
from parsers import create_train_parser

# load default train args
train_parser: ArgumentParser = create_train_parser()
train_args = vars(train_parser.parse_args())

# parse args for experiment
parser = ArgumentParser()
parser.add_argument('--hparam_dir', type=str, default='./hparam_files/hmm/perc_dropout')
parser.add_argument('--best_hparam_file', type=str, default='./hparam_files/hmm/best_hparams_dense.json')
args = vars(parser.parse_args())

print("hparam_dir:", args['hparam_dir'])
print("Loading best hparams from {}".format(args['best_hparam_file']))

# update with args from best hparam file
with open(args['best_hparam_file'], 'r') as f:
    best_hparams = json.load(f)
train_args.update(best_hparams)

os.environ["WANDB_MODE"] = "offline"

# perform runs
for i, file in enumerate(os.listdir(args['hparam_dir'])):
    with open(os.path.join(args['hparam_dir'], file), 'r') as f:
        hparams = json.load(f)
    if not file == "2.json":
        continue    
    seed = 1
    run_args = train_args.copy()

    # update with args from specific experiment
    run_args.update(hparams)
    
    run_args.update({"seed": seed})
    try:
        main(**run_args)
    except Exception as e:
        print(f"Error running experiment: {e}")