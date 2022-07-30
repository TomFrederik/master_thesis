import json
import os
from argparse import ArgumentParser
import time
import sys
sys.path.append('../../')
from train_dreamer_world_model import main as main_dreamer
from src.dreamer.parsers import create_train_parser

def main(train_args):
    
    args = {
        "hparam_file": train_args['hparam_file'],
        "best_hparam_file": train_args['best_hparam_file'],
        "job_id": train_args['job_id'],
    }

    for param in args:
        del train_args[param]
        assert param not in train_args, f"{param} is in both args and train_args"

    print("Loading best hparams from {}".format(args['best_hparam_file']))

    # update with args from best hparam file
    with open(args['best_hparam_file'], 'r') as f:
        best_hparams = json.load(f)
    for key in best_hparams:
        if key != 'num_variables':
            train_args[key] = best_hparams[key]
    train_args.update({"wandb_group": "dense_perc_dropout"})

    if args['job_id'] is None:
        job_id = int(time.time())
    else:
        job_id = args['job_id']
        
    # perform runs
    setting_id = args['hparam_file'].split('/')[-1].split('.')[0]

    with open(args["hparam_file"], 'r') as f:
        hparams = json.load(f)

    run_args = train_args.copy()

    # update with args from specific experiment
    run_args.update(hparams)
    run_args.update({"wandb_id": f"latent_scale_{run_args['num_variables']}_seed_{run_args['seed']}_job_{job_id}"})
    
    print(f"Setting ID = {setting_id}, Running with hparams: {hparams}, seed: {run_args['seed']}")
    main_dreamer(**run_args)

# create parser and add arguments
train_parser: ArgumentParser = create_train_parser()
train_parser.add_argument('--hparam_file', type=str, default='../hparam_files/perc_dropout/0.json')
train_parser.add_argument('--best_hparam_file', type=str, default='../hparam_files/dreamer_best_hparams.json')
train_parser.add_argument('--job_id', type=int, default=None)
train_args = vars(train_parser.parse_args())

main(train_args)
