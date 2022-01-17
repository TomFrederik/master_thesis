from typing import Dict, Optional
from math import ceil
import random

import gym
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, RGBImgObsWrapper, ImgObsWrapper, FullyObsWrapper
import matplotlib.pyplot as plt
import numpy as np

class PartialWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        self.env = env
    
    def observation(self, observation):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=8
        )

        return {
            'image1': rgb_img[:32],
            'image2': rgb_img[32:]
        }



class MultiViewWrapper(gym.ObservationWrapper):
    def __init__(
        self, 
        env:gym.Env, 
        obj_idx:int, 
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
    
    def observation(self, observation:Dict) -> Dict:
        
        env = self.unwrapped

        observation = observation['image'][...,0] # get object idx only array

        if not self.initialized: #TODO currently only works for immobile objects!!
            self._init_views(observation)

        views = self._apply_filter(observation)

        return views

    def _init_views(self, observation:Dict) -> None:
            # find all objects that match the desired index
            self.object_locations = np.argwhere(observation == self.obj_idx)
            
            # Make sure that every object appears at least once
            not_taken = list(len(self.object_locations))
            random.shuffle(not_taken)
            not_taken = set(not_taken)

            cur_view = 0
            view_to_obj = dict()
            while len(not_taken) > 0:
                if cur_view not in view_to_obj:
                    view_to_obj[cur_view] = []
                if ceil(self.percentages[cur_view] * len(self.object_locations)) <= len(view_to_obj[cur_view]):
                    cur_view += 1
                    continue
                else:
                    view_to_obj[cur_view].append(not_taken.pop())

            # now that every location appears at least once, we randomly sample to fill up the remaining
            # object slots for all views
            for view in range(len(self.percentages)):
                max_obj = ceil(self.percentages[cur_view] * len(self.object_locations)) 
                if max_obj <= len(view_to_obj[view]):
                    continue # this view is already complete
                else:
                    remaining = max_obj - len(view_to_obj[view])
                    space = list(set(range(len(self.object_locations))).difference(set(view_to_obj[view])))
                    view_to_obj.expand(list(np.random.choice(space, replace=False, size=remaining)))

            

    def _construct_views(self, observation:Dict) -> Dict:
        return {'view_1': view_1, 'view_2': view_2}    


# env = gym.make("MiniGrid-MultiRoom-N2-S4-v0")
env = gym.make('MiniGrid-LavaCrossingS9N2-v0')
env = FullyObsWrapper(env)
# env = RGBImgPartialObsWrapper(env) # Get pixel observations
# env = RGBImgObsWrapper(env) # Get pixel observations
# env = PartialWrapper(env) # Get pixel observations
# env = ImgObsWrapper(env) # Get rid of the 'mission' field
obs = env.reset()['image'][...,0].transpose(1,0) # This now produces an RGB tensor only
print(obs)
plt.imshow(obs.astype(np.float32)/10)
plt.show()


rgb_img = env.render(
    mode='rgb_array',
    highlight=False,
    tile_size=8
)

plt.imshow(rgb_img)
plt.show()
# while True:
#     action = env.action_space.sample()
#     obs = env.step(action)[0]
#     print(obs)
#     plt.figure()
#     plt.imshow(obs['image1'])
#     plt.show()
#     plt.figure()
#     plt.imshow(obs['image2'])
#     plt.show()
#     plt.close()
