import json
import os
from argparse import ArgumentParser
import time

from train_dreamer_world_model import main as main_dreamer
from parsers import create_train_parser

def main(train_args):
    
    args = {
        "hparam_file": train_args['hparam_file'],
        "best_hparam_file": train_args['best_hparam_file'],
        "num_seeds_per_run": train_args['num_seeds_per_run'],
        "job_id": train_args['job_id'],
        "reduced_volume": train_args['reduced_volume'],
    }

    for param in args:
        del train_args[param]
        assert param not in train_args, f"{param} is in both args and train_args"

    print("Loading best hparams from {}".format(args['best_hparam_file']))

    # update with args from best hparam file
    with open(args['best_hparam_file'], 'r') as f:
        best_hparams = json.load(f)
    train_args.update(best_hparams)
    train_args.update({"wandb_group": "dense_perc_dropout"})

    if args['job_id'] is None:
        job_id = int(time.time())
    else:
        job_id = args['job_id']
        
    # perform runs
    setting_id = args["hparam_file"].split('.')[0]

    if args["reduced_volume"] and setting_id not in ["7", "8", "10", "12"]:
        print(f"\nSkipping {setting_id} because of reduced_volume!")
        return

    with open(os.path.join(args['hparam_dir'], args["hparam_file"]), 'r') as f:
        hparams = json.load(f)

    for seed in range(args['num_seeds_per_run']):
        run_args = train_args.copy()

        # update with args from specific experiment
        run_args.update(hparams)
        
        run_args.update({"seed": seed, "wandb_id": f"setting_{setting_id}_seed_{seed}_job_{job_id}"})
        
        print(f"Setting ID = {setting_id}, Running with hparams: {hparams}, seed: {seed}")
        try:
            main_dreamer(**run_args)
        except Exception as e:
            print(f"Error running experiment: {e}")

# create parser and add arguments
train_parser: ArgumentParser = create_train_parser()
train_parser.add_argument('--hparam_file', type=str, default='./hparam_files/dreamer/perc_dropout/0.json')
train_parser.add_argument('--best_hparam_file', type=str, default='./hparam_files/dreamer/best_hparams.json')
train_parser.add_argument('--num_seeds_per_run', type=int, default=3)
train_parser.add_argument('--reduced_volume', action='store_true', help="Reduce num settings to 1")
train_parser.add_argument('--job_id', type=int, default=None)
train_args = vars(train_parser.parse_args())

main(train_args)