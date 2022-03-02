import numpy as np
from torch.utils.data import Dataset


class SingleTrajToyData(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        dones = data['done']
        self.obs = data['obs']
        self.action = data['action']
        self.sigma = np.std(self.obs)
        self.mu = np.mean(self.obs)
        self._split_trajs(dones)
    
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
        