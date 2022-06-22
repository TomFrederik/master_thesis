import json
import os
from argparse import ArgumentParser

from train_hmm import main
from parsers import create_train_parser

# load default train args
train_parser: ArgumentParser = create_train_parser()
train_args = vars(train_parser.parse_args())

# parse args for experiment
parser = ArgumentParser()
parser.add_argument('--hparam_dir', type=str, default='./hparam_files/hmm/perc_dropout')
parser.add_argument('--best_hparam_file', type=str, default='./hparam_files/hmm/best_hparams_dense.json')
parser.add_argument('--num_seeds_per_run', type=int, default=3)
args = vars(parser.parse_args())

print("hparam_dir:", args['hparam_dir'])
print("Loading best hparams from {}".format(args['best_hparam_file']))

# update with args from best hparam file
with open(args['best_hparam_file'], 'r') as f:
    best_hparams = json.load(f)
train_args.update(best_hparams)
train_args.update({"wandb_group": "dense_perc_dropout"})

# perform runs
for file in os.listdir(args['hparam_dir']):
    with open(os.path.join(args['hparam_dir'], file), 'r') as f:
        hparams = json.load(f)
    
    run_args = train_args.copy()

    # update with args from specific experiment
    run_args.update(hparams)
    
    main(**run_args)