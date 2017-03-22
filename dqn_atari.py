#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

import gym

#import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.preprocessors import *
from deeprl_hw2.core import ReplayMemory

import pdb


class LinearModel(torch.nn.Module):
    def __init__(self, window, input_shape, num_actions):
        super(LinearModel, self).__init__()
        self.window = window
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.input_size = window*np.multiply(*input_shape)

        self.affine = torch.nn.Linear(self.input_size, num_actions)

    def forward(self, x):
        x = x.resize(x.size(0), self.input_size)
        x = self.affine(x)
        return x

class ConvModel(torch.nn.Module):
    def __init__(self, window, input_shape, num_actions):
        super(ConvModel, self).__init__()
        self.window = window
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.conv_out = 64*7*7

        self.conv1 = torch.nn.Conv2d(window, 32, (8,8), 4)
        self.conv2 = torch.nn.Conv2d(32, 64, (4,4), 2)
        self.conv3 = torch.nn.Conv2d(64, 64, (3,3), 1)
        self.affine1 = torch.nn.Linear(self.conv_out, 512)
        self.affine2 = torch.nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.resize(x.size(0), self.conv_out)
        x = F.relu(self.affine1(x))
        x = self.affine2(x)
        return x


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
        return LinearModel(window, input_shape, num_actions)
    elif (model_name == 'conv'):
        return ConvModel(window, input_shape, num_actions)
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
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('-m','--model', default='linear', type=str, help='type of model')
    parser.add_argument('--gpu', default=0, type=int, help='gpu (0 or more) vs cpu (-1)')

    args = parser.parse_args()
#    args.input_shape = tuple(args.input_shape)

#    args.output = get_output_folder(args.output, args.env)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

    ## Initialize environment
    print(args)
    env = gym.make(args.env)
    env.reset()
    
    ## Arguments
    window = 4
    input_shape = (84,84)
    num_actions = env.action_space.n

    gamma = 0.99
    target_update_freq = 100
    num_burn_in = 100000
    train_freq = 1
    batch_size = 32
    epsilon = 1 ## make a decreasing epsilon

    ## Initialise Q Model
    Q = create_model(window, input_shape, num_actions, args.model)
    Q_cap = create_model(window, input_shape, num_actions, args.model)
    Q_cap.load_state_dict(Q.state_dict())

    if args.gpu >=0:
        Q.cuda()
        Q_cap.cuda()

    ## Preprocessor
    preprocessor = AtariPreprocessor(input_shape, window)

    ## Memory
    memory = ReplayMemory(num_burn_in, window_length=window)
    
    agent = DQNAgent(q_network = [Q, Q_cap],
                     preprocessor = preprocessor,
                     memory = memory,
                     gamma = gamma,
                     target_update_freq = target_update_freq,
                     num_burn_in = num_burn_in,
                     train_freq = train_freq,
                     batch_size = batch_size,
                     epsilon = epsilon,
                     mode='train')
    
    agent.compile('adam', 'huber_loss')
    agent.fit(env,1000,1000)
    pdb.set_trace()

if __name__ == '__main__':
    main()
