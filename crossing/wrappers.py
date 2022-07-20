from math import ceil
import random
from typing import Dict, Optional, List, Union, Iterable, Tuple
import itertools
import einops
import gym
from gym_minigrid.wrappers import FullyObsWrapper
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv

## NOTE maybe use this for transition tuples instead?
# class FunctionalMVW: # used to map recorded obs to multiple views, without needing an env object
#     def __init__(
#         self,
#         percentages: Optional[List[float]] = None,
#         dropout: Optional[float] = 0.0,
#         null_value: Optional[int] = 1,
#     ):
#         if percentages is None:
#             percentages = [0.7, 0.7]
        
#         # check input
#         if any(p < 0 for p in percentages) or any(p > 1 for p in percentages):
#             raise ValueError("percentages must be values in [0,1]")
#         if sum(percentages) < 1:
#             raise ValueError("percentages must sum to at least 1")
#         if dropout < 0 or dropout >= 1:
#             raise ValueError("dropout must be a value in [0,1)")
        
#         self.percentages = percentages
#         self.null_value = null_value
#         self.initialized = False
#         self.dropout = dropout
    
#     @property
#     def num_views(self):
#         return len(self.percentages)

#     def observation(self, observation, force_no_drop: Optional[bool] = False) -> Dict:
#         if not self.initialized:
#             self.view_to_loc = self._init_views(observation)

#         output = self._construct_views(observation, force_no_drop)

#         # ### TESTING HALF-HALF Split
#         # view_1 = observation.copy()
#         # view_2 = observation.copy()
#         # view_1[:, :3] = self.null_value
#         # view_2[:, 3:] = self.null_value
        
#         # views = dict(
#         #     view_1=view_1,
#         #     view_2=view_2,
#         # )
#         # print(view_1[0])
#         # print(view_2[0])
        
#         return output

#     def _init_views(self, observation) -> Dict:
#         self.locations = list(itertools.product(range(observation.shape[0]), range(observation.shape[1])))

#         # Make sure that every location appears at least once
#         shuffled_idcs = np.random.permutation(np.arange(len(self.locations)))
#         not_taken = set(self.locations[idx] for idx in shuffled_idcs)

#         # populate each view with objects so that each object appears at least once
#         cur_view = 0
#         view_to_loc = dict()
#         while len(not_taken) > 0:
#             if cur_view not in view_to_loc: # init view
#                 view_to_loc[cur_view] = set()
#             if ceil(self.percentages[cur_view] * len(self.locations)) <= len(view_to_loc[cur_view]): # view is full
#                 cur_view += 1
#                 continue
#             else: # add an object to the view
#                 view_to_loc[cur_view].add(not_taken.pop())
        
#         # now that every location appears at least once, we randomly sample to fill up the remaining
#         # object slots for all views
#         for view in range(len(self.percentages)):
#             if view not in view_to_loc:
#                 view_to_loc[view] = set()
#             max_obj = ceil(self.percentages[cur_view] * len(self.locations)) 
#             if max_obj <= len(view_to_loc[view]):
#                 continue # this view is already complete
#             else:
#                 free_space_in_view = max_obj - len(view_to_loc[view])
                
#                 # set up sample space
#                 sample_space = list(range(len(self.locations)))
#                 for i in sample_space:
#                     if self.locations[i] in view_to_loc[view]:
#                         sample_space.remove(i)
                
#                 # sample without replacement
#                 new_objects = random.sample(sample_space, free_space_in_view)
#                 new_objects = {self.locations[i] for i in new_objects}
                
#                 # merge new objects into view
#                 view_to_loc[view] = view_to_loc[view].union(new_objects)

#         return view_to_loc

#     def _construct_views(self, observation:np.ndarray, force_no_drop: Optional[bool] = False) -> Dict:
        
#         views = dict(full=observation)
#         dropped = np.zeros((self.num_views))
        
#         for view, obj_idcs in self.view_to_loc.items():
#             view_image = observation.copy()
            
#             for i, location in enumerate(self.locations):
#                 if location not in obj_idcs:
#                     view_image[..., location[0], location[1]] = self.null_value
            
            
#             for t in range(len(observation)):
#                 if force_no_drop:
#                     dropped[t, view] = 0
#                 elif self.dropout > 0 and random.random() < self.dropout:
#                     dropped[t, view] = 1
#                     view_image[t] = self.null_value
            
#             views[f'view_{view}'] = view_image
#         output = dict(
#             views=views,
#             dropped=dropped,
#         )
        
#         return output






class FunctionalMVW: # used to map recorded obs to multiple views, without needing an env object
    def __init__(
        self,
        percentage: Optional[float] = None,
        num_views: int = 1,
        dropout: Optional[float] = 0.0,
        view_null_value: Optional[int] = 0,
        drop_null_value: Optional[int] = 1,
    ):
        if percentage not in [0.5, 0.75, 1]:
            raise ValueError("percentage must be 0.5, 0.75, or 1")
        
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be a value in [0,1)")

        if num_views not in [1,2]:
            raise ValueError(f"num_views must be in [1,2] but is {num_views}")
        
        if num_views == 1 and percentage != 1:
            raise ValueError(f"percentage must be 1 when num_views is 1")
        
        self.percentage = percentage
        self.num_views = num_views
        self.view_null_value = view_null_value
        self.drop_null_value = drop_null_value
        self.initialized = False
        self.dropout = dropout
    
    def observation(self, observation, force_no_drop: Optional[bool] = False) -> Dict:
        if observation.shape[-2:] == (84,84):
            observation = tv.transforms.Resize((64,64))(torch.from_numpy(observation)).numpy()
        
        
        if not self.initialized:
            self._init_views(observation[0]) # only use the first observation of the trajectory
            self.initialized = True
        
        output = self._construct_views(observation, force_no_drop)

        # ### TESTING HALF-HALF Split
        # view_1 = observation.copy()
        # view_2 = observation.copy()
        # view_1[:, :3] = self.null_value
        # view_2[:, 3:] = self.null_value
        
        # views = dict(
        #     view_1=view_1,
        #     view_2=view_2,
        # )
        # print(view_1[0])
        # print(view_2[0])
        
        return output

    def _init_views(self, observation) -> Dict:
        # assign each location to a quadrant
        # upper-left: 0
        # upper-right: 1
        # lower-left: 2
        # lower-right: 3
        
        
        self.null_mask = np.zeros(observation.shape) + self.view_null_value
        
        if self.num_views == 1:
            self.view_masks = {0: np.ones_like(observation).astype(np.float32)}
            return
        
        half_height = observation.shape[0] // 2
        half_width = observation.shape[1] // 2
        
        
        quadrant_masks = {}
        
        for i in range(4):
            quadrant_masks[i] = np.zeros_like(observation)
            if i < 2:
                if i == 0:
                    quadrant_masks[i][:half_height, :half_width] = 1
                elif i == 1:
                    quadrant_masks[i][:half_height, half_width:] = 1
            else:
                if i == 2:
                    quadrant_masks[i][half_height:, :half_width] = 1
                elif i == 3:
                    quadrant_masks[i][half_height:, half_width:] = 1
            
        view_to_quadrants = dict()
        
        shuffled_idcs = np.random.permutation(np.arange(4))
        if self.percentage == 0.5:
            view_to_quadrants[0] = shuffled_idcs[:2]
            view_to_quadrants[1] = shuffled_idcs[2:]
        elif self.percentage == 0.75:
            view_to_quadrants[0] = shuffled_idcs[:3]
            view_to_quadrants[1] = shuffled_idcs[1:]
        elif self.percentage == 1:
            view_to_quadrants[0] = shuffled_idcs.copy()
            view_to_quadrants[1] = shuffled_idcs.copy()

        self.view_masks = {
            i: np.sum(np.stack([quadrant_masks[x] for x in view_to_quadrants[i]], axis=0), axis=0).astype(np.float32) for i in range(self.num_views)
        }
        
        
    def _construct_views(self, observation:np.ndarray, force_no_drop: Optional[bool] = False) -> Dict:
        
        views = dict(full=observation)
        dropped = np.zeros((len(observation), self.num_views))
        
        for view in range(self.num_views):
            view_image = observation * self.view_masks[view][None] + self.null_mask * (1 - self.view_masks[view][None])
            if force_no_drop:
                dropped[:, view] = 0
            elif self.dropout > 0:
                for t in range(observation.shape[0]):
                    if random.random() < self.dropout:
                        dropped[t, view] = 1
                        view_image[t] = self.drop_null_value
            views[f'view_{view}'] = view_image
        output = dict(
            views=views,
            dropped=dropped,
        )
        
        return output


class MultiViewWrapper(gym.ObservationWrapper):
    def __init__(
        self, 
        env:gym.Env, 
        obj_idx: Optional[int] = 2, 
        percentages: Optional[List[float]] = None,
        null_value: Optional[int] = 1,
):
        env = FullyObsWrapper(env) # TODO move this outside?
        super().__init__(env)
        self.env = env

        self.observation_space = gym.spaces.Box(0, 10, (7,7,), dtype=np.int32)

        if percentages is None:
            percentages = [0.7, 0.7]
        
        # check input
        if any(p < 0 for p in percentages) or any(p > 1 for p in percentages):
            raise ValueError("percentages must be values in [0,1]")

        self.percentages = percentages
        self.obj_idx = obj_idx
        self.null_value = null_value
        self.initialized = False
    
    @property
    def num_views(self):
        return len(self.percentages)

    def observation(self, observation:Dict) -> Dict:
        observation = observation['image'][1:-1,1:-1,0] # get object idx only array

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

        # populate each view with objects so that each object appears at least once
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
        super().__init__(env)
        
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

class MyFullFlatWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        env = FullyObsWrapper(env) #TODO refactor this
        super().__init__(env)
        self.env = env

        self.observation_space = gym.spaces.Box(0, 10, (49,), dtype=np.int32)
    
    def observation(self, observation):
        obs = observation['image'][1:-1,1:-1,0]
        obs = einops.rearrange(obs, 'h w -> (h w)').astype(np.int32)

        return obs

class MyFullWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.observation_space = gym.spaces.Box(0, 3, (7,7,), dtype=np.int64)
    
    def observation(self, observation):
        obs = observation['image'][1:-1,1:-1,0].astype(np.int64)
        obs[obs == 10] = 1
        # experimental
        obs[-1,-1] = 3  
        obs[self.env.agent_pos[0]-1, self.env.agent_pos[1]-1] = 0
        return obs

class NormalizeObservations(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(-1, 1, (1,7,7,), dtype=np.float32)
    
    def observation(self, obs):
        obs = (obs - 1.5) / 1.5
        return obs[None]

class FlatWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(0, 10, (49,), dtype=np.int32)
        self.env = env

    def observation(self, observation:Dict) -> Dict:
        for key in observation:
            observation[key] = einops.rearrange(observation[key], 'H W -> (H W)')
        return observation

class NumSeedsEnvWrappper(gym.Wrapper):
    def __init__(self, env, num_seeds=None):
        super().__init__(env)
        
        self.env = env
        self.num_seeds = num_seeds
        
        if self.num_seeds is not None:
            self.env.seed(np.random.choice(self.num_seeds))
        
    def reset(self):
        if self.num_seeds is not None:
            self.env.seed(np.random.choice(self.num_seeds))
        return self.env.reset()

class StepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(4)
    
    def step(self, action):
        self.env.step_count += 1

        reward = 0
        done = False


        # get hypothetical next cell
        next_cell = list(self.env.agent_pos)
        if action == 0: # down
            next_cell[0] += 1
        elif action == 1: # up
            next_cell[0] -= 1
        elif action == 2: # right
            next_cell[1] += 1
        elif action == 3: # left
            next_cell[1] -= 1
        else:
            assert False, "unknown action"
        next_cell = tuple(next_cell)

        # Get the contents of the next cell
        next_content = self.env.grid.get(*next_cell)
        # Move forward
        if next_content is None:
            self.env.agent_pos = next_cell
        elif next_content.type == 'goal':
            done = True
            self.env.agent_pos = next_cell
            # self.env.step_count = 0
            reward = 1
        elif next_content.type == 'lava':
            done = True

        if self.env.step_count >= self.env.max_steps:
            done = True

        obs = self.env.gen_obs()

        return obs, reward, done, {}

    def reset(self, **kwargs):
        print('\nCalling reset')
        print(kwargs)
        print(self.env.agent_pos)
        print(type(self.env))
        obs = self.env.reset(**kwargs)
        print(self.env.step_count)
        print(self.env.agent_pos)
        print()
        return obs

class OneHotActionToIndex(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def step(self, action):
        return self.env.step(np.argmax(action))

class OneHotObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(0, 1, (4, *env.observation_space.shape), dtype=np.float32)
    
    def observation(self, observation):
        observation = torch.nn.functional.one_hot(torch.from_numpy(observation), 4)

        observation = einops.rearrange(observation, 'h w c -> c h w').numpy()
        
        return observation