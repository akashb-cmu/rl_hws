#!/usr/bin/env python
"""Run Atari Environment with DQN."""

import argparse
import os
import random

import numpy as np

import gym

#import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.log_utils import Name
#from deeprl_hw2.preprocessors import *
#from deeprl_hw2.core import ReplayMemory

import pdb
import keras.models as M
import keras.layers as L


class LinearModel():
    def __init__(self, window, input_shape, num_actions):
        self.model = model = M.Sequential()
        model.add(L.Flatten(input_shape=tuple([window]+list(input_shape))))
        model.add(L.Dense(num_actions))
        model.output_shape


    def __call__(self):
        return self.model

    
class ConvModel():
    def __init__(self, window, input_shape, num_actions):
        self.model = model = M.Sequential()
        model.add(L.Conv2D(32,8,strides=4, activation='relu', input_shape=tuple([window]+list(input_shape)),data_format='channels_first'))
        model.add(L.Conv2D(64,4,strides=2, activation='relu', data_format='channels_first'))
        model.add(L.Conv2D(64,3,strides=1, activation='relu', data_format='channels_first'))
        model.add(L.Flatten())
        model.add(L.Dense(512))
        model.add(L.Dense(num_actions))
        model.output_shape

    def __call__(self):
        return self.model

# class MLP():
#     def __init__(self, window, input_shape, num_actions):
#         super(MLP, self).__init__()
#         self.input_shape = reduce(lambda x,y: x*y, input_shape)
#         self.num_actions = num_actions

#         self.affine1 = torch.nn.Linear(self.input_shape, 16)
#         self.affine2 = torch.nn.Linear(16, num_actions)

#     def forward(self, x):
# #        x = x.view(-1)
#         x = self.affine1(x)
# #        x = self.affine2(x)
#         return x

def create_model(window, input_shape, num_actions,
                 model_name='linear'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.
      Options: linear, deep

    Returns
    -------
    torch.nn.Module
      The Q-model.
    """
    if (model_name == 'linear'):
        model = LinearModel(window, input_shape, num_actions)
        return model()
    elif (model_name == 'conv'):
        model = ConvModel(window, input_shape, num_actions) 
        return model()
    elif model_name == 'mlp':
        model = MLP(window, input_shape, num_actions) 
        return model()
    else:
        assert 1, 'Model not found'
    

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('-m','--model', default='linear', type=str, help='type of model')
    parser.add_argument('--input_shape', default=[84,84], type=int, nargs='+', help='input shape')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning Rate')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--burn_in', default=50000, type=int, help='Burn In Time')
    parser.add_argument('--eval_freq', default=40000, type=int, help='Evaluation Frequency')
    parser.add_argument('--target_update_freq',default=10000,type=int, help='Target Update Frequency')
    parser.add_argument('--tot_frames',default=5000000,type=int, help='Total Number of Frames')
    
    args = parser.parse_args()
    name = Name(args, 'output', 'model')
#    args.output = get_output_folder(args.output, args.env)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.
    ## Initialize environment
    print(args)
    train_env = gym.make(args.env)
    train_env.reset()
    env = gym.make(args.env)
    env.reset()
    ## Arguments

    window = 4
    input_shape = args.input_shape

    num_actions = train_env.action_space.n

    gamma = 0.99
    tot_frames  = args.tot_frames
    target_update_freq = args.target_update_freq
    num_burn_in = args.burn_in
    train_freq = 1
    eval_freq = args.eval_freq
    batch_size = args.batch_size
    epsilon = 1 ## make a decreasing epsilon

    ## Initialise Q Model
    Q = create_model(window, input_shape, num_actions, args.model)
    Q_cap = create_model(window, input_shape, num_actions, args.model)
    Q_cap.set_weights(Q.get_weights())
    
    ## Preprocessor
#    preprocessor = AtariPreprocessor(input_shape, window)

    ## Memory
#    memory = ReplayMemory(num_burn_in, window_length=window)
    
    agent = DQNAgent(q_network = [Q, Q_cap],
                     gamma = gamma,
                     target_update_freq = target_update_freq,
                     num_burn_in = num_burn_in,
                     train_freq = train_freq,
                     eval_freq = eval_freq,
                     batch_size = batch_size,
                     epsilon = epsilon,
                     num_actions = num_actions,
                     name=name,
                     folder=args.output,
                     mode='train')

    agent.compile('rmsprop', 'huber_loss', args.lr)
    
    agent.fit_akash(train_env,env,
                    tot_frames=tot_frames,
                    burn_in_time=num_burn_in,
                    eval_plot_period=eval_freq,
                    target_fix_freq=target_update_freq,
                    batch_size=batch_size)
    pdb.set_trace()

if __name__ == '__main__':
    main()
