from wandb import Api

from typing import List, Dict, Sequence, Tuple, Union
from tqdm import tqdm
import numpy as np
import logging
import os
import json

def gen_run_id_list(
    job_id: str, 
    num_settings: int, 
    num_seeds: int
) -> List[str]:
    my_list = []
    for setting_num in range(num_settings):
        for seed_num in range(num_seeds):
            my_list.append(f"setting_{setting_num}_seed_{seed_num}_job_{job_id}")
    return my_list

def fetch_last_loss(
    api: Api, 
    run_id: str, 
    project: str, 
    loss_name: str = "tuning_loss", 
    entity: str = 'TomFrederik'
) -> float:
    run = api.run(f"{entity}/{project}/{run_id}")
    print(loss_name)
    out = run.history(keys=[f'Validation/{loss_name}'])
    print(out)
    print(out.iloc[-1])
    return run.history(keys=[f'Validation/{loss_name}']).iloc[-1][f'Validation/{loss_name}']


def fetch_loss_list_per_setting(
    api: Api,
    job_id: str,
    num_settings: Union[int, Sequence[int]],
    num_seeds: int,
    project: str,
    loss_name: str = "tuning_loss", 
    entity: str = 'TomFrederik',
    reduced_volume: bool = False,
    skip_tuples: Sequence[Tuple[int, int]] = None,
    prefix: str = "setting",
    setting_suffix: str = "",
) -> Dict:
    if skip_tuples is None:
        skip_tuples = []
    out_dict = dict()

    if isinstance(num_settings, Sequence):
        settings = num_settings
    elif isinstance(num_settings, int):
        settings = list(range(num_settings))
    else:
        raise TypeError(f"Expected int or Sequence[int] for type(num_settings) but got {type(num_settings)}")
    
    for setting_num in tqdm(settings):
        loss_list = []
        if reduced_volume and setting_num not in [7, 10, 12]:
            continue
        for seed_num in range(num_seeds):
            if (setting_num, seed_num) in skip_tuples:
                print(f"WARNING: skipping setting {setting_num}, seed {seed_num} because of NaNs")
                continue
            run_id = f"{prefix}_{setting_num}{setting_suffix}_seed_{seed_num}_job_{job_id}"
            print(run_id)
            loss = fetch_last_loss(api, run_id, project, loss_name, entity)
            loss_list.append(loss)
        out_dict[setting_num] = loss_list
    return out_dict

def compute_mean_std_per_setting(loss_dict: Dict) -> Dict:
    out_dict = dict()
    for key, val in loss_dict.items():
        mean = np.mean(val)
        std = np.std(val, ddof=1)
        out_dict[key] = (mean, std)
    return out_dict



def download_and_save(
    api,
    entity,
    target_dir, 
    job_id, 
    num_settings,
    num_seeds,
    proj_name,
    loss_name,
    skip_tuples: Sequence[Tuple[str, str]] = None,
    prefix: str = "setting",
    setting_suffix: str = "",
):
    if os.path.exists(target_dir):
        logging.warning("Directory already exists")
        response = str(input("Force download? (y/n)"))
        while response not in ["y", "n"]:
            response = str(input("Force download? (y/n)"))
    else:
        response = "y"
        
    if response == "y":
        os.makedirs(target_dir, exist_ok=True)
        mean_std_per_setting = compute_mean_std_per_setting(
            fetch_loss_list_per_setting(
                api, 
                job_id, 
                num_settings, 
                num_seeds, 
                proj_name, 
                loss_name, 
                entity, 
                skip_tuples=skip_tuples, 
                prefix=prefix,
                setting_suffix=setting_suffix,
            ))
        with open(f"{target_dir}/mean_std_per_setting.json", "w") as f:
            json.dump(mean_std_per_setting, f)
    elif response == "n":
        print('Directory already exists -> skipping...')
    else:
        raise ValueError("Unexpected response")