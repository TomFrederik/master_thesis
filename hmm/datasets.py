import numpy as np
from torch.utils.data import Dataset
import torch
import sys
sys.path.insert(0, '../')
from crossing.wrappers import FunctionalMVW


def _split_trajs(dones, actions, obs, max_len=20):
    stop_idcs = (np.where(dones == 1)[0] + 1).tolist()
    start_idcs = np.concatenate(([0], stop_idcs[:-1])).tolist()
    action = []
    
    # remove trajectories that are too long
    skip_idcs = set({})
    for i in range(len(stop_idcs)):
        if stop_idcs[i]-start_idcs[i] > max_len:
            skip_idcs.add(i)
    
    for i, (start, stop) in enumerate(zip(start_idcs, stop_idcs)):
        if i in skip_idcs:
            continue
        action.append(actions[start-i:stop-i-1])
    actions = np.array(action, dtype=object)
    obs = np.array([obs[start:stop] for i, (start, stop) in enumerate(zip(start_idcs, stop_idcs)) if i not in skip_idcs], dtype=object)
    return obs, actions

def load_data(data_path, multiview=False, train_val_split=0.9, max_datapoints=None, max_len=20, **kwargs):
    data = np.load(data_path)
    dones = data['done']
    obs = data['obs']
    action = data['action']
        
    sigma = np.std(obs)
    mu = np.mean(obs)
    obs, actions = _split_trajs(dones, action, obs, max_len)
    if max_datapoints is not None:
        obs = obs[:max_datapoints]
        actions = actions[:max_datapoints]
    multiview = multiview
    if multiview:
        multiview_wrapper = FunctionalMVW(kwargs['percentages'], kwargs['dropout'], kwargs['null_value'])
    else:
        multiview_wrapper = None
    all_idcs = np.random.permutation(np.arange(len(obs)))
    train_idcs = all_idcs[:int(train_val_split*len(obs))]
    val_idcs = all_idcs[int(train_val_split*len(obs)):]
    
    return (obs[train_idcs], actions[train_idcs]), (obs[val_idcs], actions[val_idcs]), sigma, mu, multiview_wrapper

def construct_train_val_data(data_path, multiview=False, train_val_split=0.9, test_only_dropout=False, max_datapoints=None, max_len=20, **kwargs):
    (train_obs, train_actions), (val_obs, val_actions), sigma, mu, mvwrapper = load_data(data_path, multiview, train_val_split, max_datapoints, max_len, **kwargs)
    train_data = SingleTrajToyData(train_obs, train_actions, sigma, mu, mvwrapper, drop=not test_only_dropout)
    val_data = SingleTrajToyData(val_obs, val_actions, sigma, mu, mvwrapper, drop=True)
    return train_data, val_data

class SingleTrajToyData(Dataset):
    def __init__(self, obs, actions, sigma, mu, mvwrapper, drop):
        self.obs = obs
        self.actions = actions
        self.sigma = sigma
        self.mu = mu
        self.drop = drop
        self.mvwrapper = mvwrapper
        
    def set_drop(self, drop: bool):
        self.drop = drop
    
    def __len__(self):
        return len(self.actions)
    
    def get_no_drop(self, idx):
        return self.__getitem__(idx, force_no_drop=True)
    
    def __getitem__(self, idx, force_no_drop=False):
        obs = self.obs[idx]
        if self.mvwrapper: # stack views along channel dimension
            output = self.mvwrapper.observation(obs, (force_no_drop or not self.drop)) 
            obs = output['views']
            dropped = output['dropped']
            obs = np.stack([o for key, o in obs.items() if key.startswith('view')], axis=1) 
        else:
            obs = obs[:,None]
            dropped = np.zeros((obs.shape[0], 1))
        
        player_pos = np.zeros_like(obs)
        player_pos[obs == 0] = 1
        
        # center and normalize
        obs = (obs - self.mu) / self.sigma
        
        # add null action since we did not record action at last step
        action = self.actions[idx]
        action = np.append(action, np.zeros_like(action[-1]))

        terms = np.zeros_like(action)
        terms[-1] = 1
        
        value_prefixes = 0.9 ** np.arange(len(action))[::-1]
        
        return (
            obs.astype(np.float32),
            action.astype(np.int64),
            value_prefixes.astype(np.float32),
            terms.astype(np.int64), #terms
            dropped.astype(np.float32),
            player_pos.astype(np.float32),
        )

class BatchTrajToyData(Dataset):
    def __init__(self, data_path, seq_len=12, **kwargs):
        data = np.load(data_path)
        dones = data['done']
        self.obs = data['obs']
        self.action = data['action']
        self.sigma = np.std(self.obs)
        self.mu = np.mean(self.obs)
        self._split_trajs(dones)
        self.seq_len = seq_len
        
    def _split_trajs(self, dones):
        stop_idcs = np.where(dones == 1)[0] + 1
        start_idcs = np.concatenate(([0], stop_idcs[:-1]))
        action = []
        for i, (start, stop) in enumerate(zip(start_idcs, stop_idcs)):
            action.append(self.action[start-i:stop-i-1])
        self.action = action
        self.obs = [self.obs[start:stop] for start, stop in zip(start_idcs, stop_idcs)]
        
    def __len__(self):
        return len(self.action)
    
    def __getitem__(self, idx):
        obs = (self.obs[idx] - self.mu) / self.sigma
        return (
            obs[:,None].astype(np.float32), # add channel dimension
            self.action[idx].astype(np.int64),
            np.zeros_like(self.action[idx]), #rewards
        )

    def truncate_sequence(self, sequence):
        if self.seq_len >= len(sequence):
            return torch.from_numpy(sequence)
        else:
            return torch.from_numpy(sequence[:self.seq_len]) # only use first T frames #TODO: make this better

    def truncate_tuple(self, tpl):
        return tuple(map(self.truncate_sequence, tpl))

    def collate_fn(self, batch_list):
        tpl = tuple(map(self.truncate_tuple, zip(*batch_list)))
        return tuple(map(torch.stack, tpl))
