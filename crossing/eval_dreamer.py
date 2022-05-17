import wandb
import argparse
import os
import torch
import numpy as np
import gym

import gym_minigrid as mini

from wrappers import MyFullWrapper, StepWrapper, DeterministicEnvWrappper, OneHotObs, OneHotActionToIndex, NormalizeObservations

from dreamerv2.utils.wrapper import GymMinAtar, OneHotAction
from dreamerv2.training.config import MinAtarConfig
from dreamerv2.training.trainer import Trainer
from dreamerv2.training.evaluator import Evaluator

def main(args):

    # os.environ["WANDB_MODE"] = "offline"
    exp_id = args.id
    
    '''make dir for saving results'''
    result_dir = os.path.join('results', '{}_{}'.format("simple_crossing", exp_id))
    model_dir = os.path.join(result_dir, 'models')                                                  #dir to save learnt models
    load_path = os.path.join(model_dir, 'models_best.pth')
    
    np.random.seed(args.seed)
    if torch.cuda.is_available() and args.device == "cuda":
        device = 'cuda'
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    torch.manual_seed(args.seed)
    # print('using :', device)  
    
    env = gym.make("MiniGrid-SimpleCrossingS9N1-v0")
    env = StepWrapper(env)
    env = mini.wrappers.FullyObsWrapper(env)
    env = MyFullWrapper(env)
    env = NormalizeObservations(env)
    if args.constant_env: #TODO
        env = DeterministicEnvWrappper(env)
    env = OneHotActionToIndex(env)

    obs_shape = env.observation_space.shape
    action_size = env.action_space.n
    obs_dtype = np.float32
    action_dtype = np.float32
    batch_size = args.batch_size
    seq_len = args.seq_len
    print(f"{obs_shape = }")

    config = MinAtarConfig( #TODO
        env="MiniGrid-SimpleCrossingS9N1-v0",
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype = obs_dtype,
        action_dtype = action_dtype,
        seq_len = seq_len,
        batch_size = batch_size,
        model_dir=model_dir, 
    )

    config_dict = config.__dict__
    config_dict['constant_env'] = args.constant_env
    
    evaluator = Evaluator(config, device)

    evaluator.eval_saved_agent(env, load_path)


if __name__ == "__main__":

    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add it here"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--id', default=0, type=int)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence Length (chunk length)')
    parser.add_argument('--constant_env', action='store_true', help="Whether to change environment layout on reset")
    args = parser.parse_args()
    main(args)
