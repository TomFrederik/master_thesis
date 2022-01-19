from math import ceil
import random
from typing import Dict, Optional, List, Union, Iterable

import einops
import gym
from gym_minigrid.wrappers import FullyObsWrapper
import matplotlib.pyplot as plt
import numpy as np

class MultiViewWrapper(gym.ObservationWrapper):
    def __init__(
        self, 
        env:gym.Env, 
        obj_idx: Optional[int] = 9, 
        percentages: Optional[List[float]] = None,
        null_value: Optional[int] = 1,
):
        if percentages is None:
            percentages = [0.7, 0.7]
        
        # check input
        if any(p < 0 for p in percentages) or any(p > 1 for p in percentages):
            raise ValueError("percentages must not contain values outside [0,1]")
        if sum(percentages) < 1:
            raise ValueError(f"percentages should sum at least to 1, but {sum(percentages) = }")

        self.percentages = percentages
        self.obj_idx = obj_idx
        self.null_value = null_value
        self.initialized = False
        self.env = FullyObsWrapper(env) # TODO move this outside?
    
    @property
    def num_views(self):
        return len(self.percentages)

    def observation(self, observation:Dict) -> Dict:
        
        env = self.unwrapped

        observation = observation['image'][...,0] # get object idx only array

        if not self.initialized: #TODO currently only works for immobile objects!!
            self.view_to_obj = self._init_views(observation)

        views = self._construct_views(observation)

        return views

    def _init_views(self, observation:Dict) -> Dict:
        # find all objects that match the desired index
        self.object_locations = np.argwhere(observation == self.obj_idx)

        # verify that there is at least one such object
        if len(self.object_locations) == 0:
            raise ValueError(f"Could not find any objects with index {self.obj_idx} in the environment!")
        
        # Make sure that every object appears at least once
        not_taken = list(range(len(self.object_locations)))
        random.shuffle(not_taken)
        not_taken = set(not_taken)

        cur_view = 0
        view_to_obj = dict()
        while len(not_taken) > 0:
            if cur_view not in view_to_obj:
                view_to_obj[cur_view] = set()
            if ceil(self.percentages[cur_view] * len(self.object_locations)) <= len(view_to_obj[cur_view]):
                cur_view += 1
                continue
            else:
                view_to_obj[cur_view].add(not_taken.pop())

        # now that every location appears at least once, we randomly sample to fill up the remaining
        # object slots for all views
        for view in range(len(self.percentages)):
            if view not in view_to_obj:
                view_to_obj[view] = set()
            max_obj = ceil(self.percentages[cur_view] * len(self.object_locations)) 
            if max_obj <= len(view_to_obj[view]):
                continue # this view is already complete
            else:
                remaining = max_obj - len(view_to_obj[view])
                space = list(set(range(len(self.object_locations))).difference(set(view_to_obj[view])))
                view_to_obj[view] = view_to_obj[view].union(set(list(np.random.choice(space, replace=False, size=remaining))))

        return view_to_obj

    def _construct_views(self, observation:np.ndarray) -> Dict:
        
        ground_truth_view = observation
        
        views = dict(full=ground_truth_view)

        for view, obj_idcs in self.view_to_obj.items():
            view_image = ground_truth_view.copy()
            
            for i, location in enumerate(self.object_locations):
                if i not in obj_idcs:
                    view_image[..., location[0], location[1]] = self.null_value
            
            views[f'view_{view}'] = view_image
        
        return views


class DropoutWrapper(gym.ObservationWrapper):
    def __init__(
        self, 
        env, 
        dropout_p:Optional[Union[Iterable[float], float]] = 0.1,
        ignore_view:Optional[str] = 'full',
    ):
        # check inputs
        if isinstance(dropout_p, Iterable):
            if any(p > 1 for p in dropout_p) or any(p < 0 for p in dropout_p):
                raise ValueError(f"Illegal dropout_p value detected. Values should be in range [0,1].")
            if len(dropout_p) != env.num_views:
                raise ValueError(f"dropout_p length ({len(dropout_p)}) should be same as num_views ({env.num_views})")
        else:
            if dropout_p > 1 or dropout_p < 0:
                raise ValueError(f"Illegal dropout_p value detected. Value should be in range [0,1].")
            dropout_p = [dropout_p] * env.num_views
        
        self.env = env
        self.dropout_p = dropout_p
        self.ignore_view = ignore_view

    def observation(self, observation:Dict) -> Dict:
        cur = 0
        for view in observation.keys():
            if view == self.ignore_view:
                continue
            
            if random.random() < self.dropout_p[cur]:
                observation[view] = None
            cur += 1
        return observation

class FlatWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        self.env = env

    def observation(self, observation:Dict) -> Dict:
        for key in observation:
            observation[key] = einops.rearrange(observation[key], 'H W -> (H W)')
        return observation