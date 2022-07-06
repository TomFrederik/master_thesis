import numpy as np
from torch.utils.data import Dataset
import torch
import sys
sys.path.insert(0, '../')
from crossing.wrappers import FunctionalMVW
from typing import TypeVar
from dataclasses import dataclass
import h5py

Tensor = TypeVar('Tensor', torch.Tensor, np.ndarray)

def _split_trajs(dones, actions, obs):
    stop_idcs = (np.where(dones == 1)[0] + 1).tolist()
    start_idcs = np.concatenate(([0], stop_idcs[:-1])).tolist()
    action = []
    
    # remove trajectories that are too long
    skip_idcs = set({})
    for i, (start, stop) in enumerate(zip(start_idcs, stop_idcs)):
        action.append(actions[start-i:stop-i-1])
    actions = np.array(action, dtype=object)
    obs = np.array([obs[start:stop] for i, (start, stop) in enumerate(zip(start_idcs, stop_idcs)) if i not in skip_idcs], dtype=object)
    return obs, actions

@dataclass
class Transition:
    state: Tensor
    action: Tensor
    next_state: Tensor
    value_prefix: Tensor
    dropped: Tensor = None


def _extract_transition_tuples(dones, action, obs):
    transitions = []
    ctr = 0
    for i in range(len(action)-1,-1,-1):
        if dones[i] == 1:
            ctr = 0
            continue
        vp = 0.99 ** ctr
        ctr += 1
        transitions.append(Transition(obs[i], action[i], obs[i+1], vp))
    return np.array(transitions)

# def load_transition_data(data_path, multiview=False, train_val_split=0.9, max_datapoints=None, max_len=20, **kwargs):
#     data = np.load(data_path)
#     dones = data['done']
#     obs = data['obs']
#     action = data['action']
        
#     sigma = np.std(obs)
#     mu = np.mean(obs)
#     transitions = _extract_transition_tuples(dones, action, obs)
#     if max_datapoints is not None:
#         transitions = transitions[:max_datapoints]
#     multiview = multiview
#     if multiview:
#         multiview_wrapper = FunctionalMVW(kwargs['percentages'], kwargs['dropout'], kwargs['null_value'])
#     else:
#         multiview_wrapper = None
#     all_idcs = np.random.permutation(np.arange(len(transitions)))
#     train_idcs = all_idcs[:int(train_val_split*len(transitions))]
#     val_idcs = all_idcs[int(train_val_split*len(transitions)):]
    
#     return transitions[train_idcs], transitions[val_idcs], sigma, mu, multiview_wrapper

# def construct_train_val_transition_data(data_path, multiview=False, train_val_split=0.9, test_only_dropout=False, max_datapoints=None, max_len=20, **kwargs):
    # train_transitions, val_transitions, sigma, mu, mvwrapper = load_data(data_path, multiview, train_val_split, max_datapoints, max_len, **kwargs)
    # train_data = TransitionData(train_transitions, sigma, mu, mvwrapper, drop=not test_only_dropout)
    # val_data = TransitionData(val_transitions, sigma, mu, mvwrapper, drop=True)
    # return train_data, val_data

def load_data_h5py(data_path, multiview=False, train_val_split=0.9, **kwargs):
    f = h5py.File(data_path, 'r')
    
    std = f['obs'].attrs.get('std')
    mean = f['obs'].attrs.get('mean')
    
    if multiview:
        multiview_wrapper = FunctionalMVW(kwargs['percentages'], kwargs['dropout'], kwargs['null_value'])
    else:
        multiview_wrapper = None
        
    all_idcs = np.random.permutation(np.arange(len(f['obs'])))
    train_idcs = all_idcs[:int(train_val_split*len(f['obs']))]
    val_idcs = all_idcs[int(train_val_split*len(f['obs'])):]

    f.close()
    
    return train_idcs, val_idcs, std, mean, multiview_wrapper

def load_data(data_path, multiview=False, train_val_split=0.9, **kwargs):
    data = np.load(data_path)
    dones = data['done']
    obs = data['obs']
    action = data['action']
    sigma = np.std(obs)
    mu = np.mean(obs)
    obs, actions = _split_trajs(dones, action, obs)
    if multiview:
        multiview_wrapper = FunctionalMVW(kwargs['percentages'], kwargs['dropout'], kwargs['null_value'])
    else:
        multiview_wrapper = None
    all_idcs = np.random.permutation(np.arange(len(obs)))
    train_idcs = all_idcs[:int(train_val_split*len(obs))]
    val_idcs = all_idcs[int(train_val_split*len(obs)):]
    
    return (obs[train_idcs], actions[train_idcs]), (obs[val_idcs], actions[val_idcs]), sigma, mu, multiview_wrapper

def construct_pong_train_val_data(data_path, multiview=False, train_val_split=0.9, test_only_dropout=False, scale=1.0, **kwargs):
    train_idcs, val_idcs, sigma, mu, mvwrapper = load_data_h5py(data_path, multiview, train_val_split, **kwargs)
    train_data = PongBatchTrajToyData(data_path, train_idcs, sigma, mu, mvwrapper, drop=not test_only_dropout, max_len=kwargs['max_len'], scale=scale)
    val_data = PongBatchTrajToyData(data_path, val_idcs, sigma, mu, mvwrapper, drop=True, max_len=kwargs['max_len'], scale=scale)
    return train_data, val_data



def construct_toy_train_val_data(data_path, multiview=False, train_val_split=0.9, test_only_dropout=False, scale=1.0, **kwargs):
    (train_obs, train_actions), (val_obs, val_actions), sigma, mu, mvwrapper = load_data(data_path, multiview, train_val_split, **kwargs)
    train_data = ToyBatchTrajToyData(train_obs, train_actions, sigma, mu, mvwrapper, drop=not test_only_dropout, max_len=kwargs['max_len'], scale=scale)
    val_data = ToyBatchTrajToyData(val_obs, val_actions, sigma, mu, mvwrapper, drop=True, max_len=kwargs['max_len'], scale=scale)
    return train_data, val_data


class TransitionData(Dataset):
    def __init__(self, transitions, sigma, mu, mvwrapper, drop) -> None:
        super().__init__()
        self.transitions = transitions
        self.sigma = sigma
        self.mu = mu
        self.mvwrapper = mvwrapper
        self.drop = drop
    
    def set_drop(self, drop: bool):
        self.drop = drop
    
    def __len__(self):
        return len(self.transitions)
    
    def get_no_drop(self, idx):
        return self.__getitem__(idx, force_no_drop=True)
    
    def __getitem__(self, idx, force_no_drop=False):
        transition: Transition = self.transitions[idx]
        
        print(transition)
        print(transition.state.shape)
        if self.mvwrapper: # stack views along channel dimension
            output = self.mvwrapper.observation(transition.state, (force_no_drop or not self.drop)) 
            state = output['views']
            dropped = output['dropped']
            print(dropped)
            state = np.stack([o for key, o in state.items() if key.startswith('view')], axis=1) 
        else:
            state = state[:,None]
            dropped = np.zeros(1)
        
        # center and normalize
        state = (state - self.mu) / self.sigma
        next_state = (transition.next_state - self.mu) / self.sigma
        
        action = transition.action
        
        vp = transition.value_prefix

        return Transition(
            state.astype(np.float32),
            action.astype(np.int64),
            next_state.astype(np.float32),
            float(vp),
            float(dropped),
        )
    
    @staticmethod
    def collate_fn(batch):
        state = torch.stack([torch.from_numpy(item[0]) for item in batch])
        actions = torch.stack([torch.from_numpy(item[1]) for item in batch])
        next_state = torch.stack([torch.from_numpy(item[2]) for item in batch])
        vp = torch.stack([torch.from_numpy(item[3]) for item in batch])
        dropped = torch.stack([torch.from_numpy(item[4]) for item in batch])
        return Transition(state, actions, next_state, vp, dropped)
    

class ToyBatchTrajToyData(Dataset):
    def __init__(self, obs, actions, sigma, mu, mvwrapper, drop, max_len=-1, scale=1):
        self.obs = obs
        self.actions = actions
        self.sigma = sigma
        self.mu = mu
        self.drop = drop
        self.mvwrapper = mvwrapper
        self.max_len = max_len
        self.scale = scale
        self.img_shape = self.obs[0].shape[-2:]
    
    def set_drop(self, drop: bool):
        self.drop = drop
    
    def __len__(self):
        return len(self.actions)
    
    def get_no_drop(self, idx):
        return self.__getitem__(idx, force_no_drop=True)
    
    def __getitem__(self, idx, force_no_drop=False):
        
        obs = self.obs[idx]
        action = self.actions[idx]
        
        # compute traj length and pad length
        traj_length = len(self.actions[idx])
    
        if self.max_len == -1:
            max_len = traj_length
            start_idx = 0
        else:
            max_len = self.max_len
            start_idx = np.random.choice(traj_length)   
        
        pad_length = max(0, max_len + start_idx - traj_length)
        
        # retrieve and pad actions
        action = action[start_idx:start_idx+max_len]
        action = np.append(action, np.zeros(pad_length))
        # add null action since we did not record action at last step
        action = np.append(action, np.zeros_like(action[-1]))

        # retrieve and pad observations
        obs = obs[start_idx:start_idx+max_len+1]
        obs = np.concatenate([obs, np.zeros((pad_length, *obs.shape[1:]))], axis=0)
        if self.mvwrapper: # stack views along channel dimension
            output = self.mvwrapper.observation(obs, (force_no_drop or not self.drop)) 
            obs = output['views']
            dropped = output['dropped']
            obs = np.stack([o for key, o in obs.items() if key.startswith('view')], axis=1) 
        else:
            obs = obs[:,None]
            dropped = np.zeros((obs.shape[0], 1))
        # scale up
        obs = scale_obs(obs, self.scale)
        
        player_pos = np.zeros_like(obs)
        player_pos[obs == 0] = 1
        
        # center and normalize
        obs = (obs - self.mu) / self.sigma
        
        nonterms = np.ones_like(action)
        if pad_length > 0:
            nonterms[-pad_length:] = 0

        value_prefixes = np.zeros_like(action)
        if max_len + start_idx - traj_length >= 0:
            value_prefixes[-(max_len + start_idx - traj_length + 1):] = 1
        return (
            obs.astype(np.float32),
            action.astype(np.int64),
            value_prefixes.astype(np.float32),
            nonterms.astype(np.int64), #terms
            dropped.astype(np.float32),
            player_pos.astype(np.float32),
        )
        
class PongBatchTrajToyData(Dataset):
    def __init__(self, data_path, idcs, sigma, mu, mvwrapper, drop, max_len=-1, scale=1):
        self.data_path = data_path
        self.idcs = idcs
        self.sigma = sigma
        self.mu = mu
        self.drop = drop
        self.mvwrapper = mvwrapper
        self.max_len = max_len
        self.scale = scale
        with h5py.File(data_path, 'r') as f:
            self.img_shape = f['obs'][f'traj_{self.idcs[0]}'].shape[-2:]
        
    def set_drop(self, drop: bool):
        self.drop = drop
    
    def __len__(self):
        return len(self.idcs)
    
    def get_no_drop(self, idx):
        return self.__getitem__(idx, force_no_drop=True)
    
    def __getitem__(self, idx, force_no_drop=False):
        # compute traj length and pad length
        with h5py.File(self.data_path, 'r') as f:
        
            action = f['action'][f'traj_{self.idcs[idx]}']
            obs = f['obs'][f'traj_{self.idcs[idx]}']
            reward = f['reward'][f'traj_{self.idcs[idx]}'] 
            done = f['done'][f'traj_{self.idcs[idx]}']
        
            traj_length = len(action)
        
            if self.max_len == -1:
                max_len = traj_length
                start_idx = 0
            else:
                max_len = self.max_len
                start_idx = np.random.choice(traj_length)   
            
            pad_length = max(0, max_len + start_idx - traj_length)
            
            # retrieve and pad actions
            action = action[start_idx:start_idx+max_len]
            action = np.append(action, np.zeros(pad_length))
            # add null action since we did not record action at last step
            action = np.append(action, np.zeros_like(action[-1]))

            # retrieve and pad observations
            obs = obs[start_idx:start_idx+max_len+1]
            obs = np.concatenate([obs, np.zeros((pad_length, *obs.shape[1:]))], axis=0)
            if self.mvwrapper: # stack views along channel dimension
                output = self.mvwrapper.observation(obs, (force_no_drop or not self.drop)) 
                obs = output['views']
                dropped = output['dropped']
                obs = np.stack([o for key, o in obs.items() if key.startswith('view')], axis=1) 
            else:
                obs = obs[:,None]
                dropped = np.zeros((obs.shape[0], 1))
            # scale up
            obs = scale_obs(obs, self.scale)
            
            player_pos = np.zeros_like(obs)
            player_pos[obs == 0] = 1
            
            # center and normalize
            obs = (obs - self.mu) / self.sigma
            
            nonterms = np.ones_like(action)
            if pad_length > 0:
                nonterms[-pad_length:] = 0

            value_prefixes = reward[start_idx:start_idx+max_len+1,0]
            # value_prefixes = np.zeros_like(action)
            # if max_len + start_idx - traj_length >= 0:
            #     value_prefixes[-(max_len + start_idx - traj_length + 1):] = 1
        return (
            obs.astype(np.float32),
            action.astype(np.int64),
            value_prefixes.astype(np.float32),
            nonterms.astype(np.int64), #terms
            dropped.astype(np.float32),
            player_pos.astype(np.float32),
        )
        
def scale_obs(obs, scale):
    h, w = obs.shape[-2:]
    new_h, new_w = int(h * scale), int(w * scale)
    new_obs = np.zeros((*obs.shape[:-2], new_h, new_w))
    # fill new obs appropriately
    for i in range(new_h):
        for j in range(new_w):
            new_obs[..., i, j] = obs[..., int(i / scale), int(j / scale)]
    return new_obs