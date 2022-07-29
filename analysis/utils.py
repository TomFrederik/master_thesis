from wandb import Api

from typing import List, Dict, Sequence, Tuple
from tqdm import tqdm
import numpy as np

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
    return run.history(keys=[f'Validation/{loss_name}']).iloc[-1][f'Validation/{loss_name}']


def fetch_loss_list_per_setting(
    api: Api,
    job_id: str,
    num_settings: int,
    num_seeds: int,
    project: str,
    loss_name: str = "tuning_loss", 
    entity: str = 'TomFrederik',
    reduced_volume: bool = False,
    skip_tuples: Sequence[Tuple[int, int]] = None,
) -> Dict:
    if skip_tuples is None:
        skip_tuples = []
    out_dict = dict()
    
    for setting_num in tqdm(range(num_settings)):
        loss_list = []
        if reduced_volume and setting_num not in [7, 10, 12]:
            continue
        for seed_num in range(num_seeds):
            if (setting_num, seed_num) in skip_tuples:
                print(f"WARNING: skipping setting {setting_num}, seed {seed_num} because of NaNs")
                continue
            run_id = f"setting_{setting_num}_seed_{seed_num}_job_{job_id}"
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