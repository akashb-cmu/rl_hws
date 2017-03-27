import gym
import cPickle
#import torch
#import torch.autograd as A
#import torch.nn as NN
#import torch.nn.functional as F
#import torch.optim as O
from core_akash import Preprocessor, ReplayMemory
from preprocessors_akash import HistoryPreprocessor, AtariPreprocessor, PreprocessorSequence
import matplotlib.pyplot as plt
#from Q_Networks import DQN_Mnih, DQN_Mnih_BNorm
#from objectives import *
from policy_akash import *
import numpy as np
import skvideo.io

def get_action_mask(batch_q_values, actions):
    action_mask = np.zeros(batch_q_values.size(),dtype=np.float32)
    for index, action in enumerate(actions):
        action_mask[index][action] = 1.
    return action_mask

def get_action_values(batch_q_values, actions):
    q_vals = []
    for ind, action in enumerate(actions):
        q_vals.append(batch_q_values[ind][action])
    return torch.cat(q_vals)


def get_next_sample(env, preproc, action, prev_sample=None, render=False, ret_reward=False):
    observation, reward, is_terminal, debug_info = env.step(action)
    if render:
        env.render()
    next_sample = preproc.process_state_for_memory(env_output=observation,
                                                   curr_state_id=prev_sample.next_ip_state_id if prev_sample
                                                                                                 is not None else -1,
                                                   # for first sample, history is all 0s
                                                   action=action, reward=reward, is_terminal=is_terminal)
    if not ret_reward:
        return next_sample
    else:
        return next_sample, reward, observation

def get_first_state(env, preproc, render=False, ret_reward=False):
    env.reset()
    preproc.reset()
    if render:
        env.render()
    action = env.action_space.sample()
    return get_next_sample(env, preproc=preproc, action=action, prev_sample=None, render=render, ret_reward=ret_reward)

def get_action(preproc, prev_sample, policy, q_net, use_gpu, is_train=True):
    curr_state, next_state, action, reward, is_terminal = preproc.process_state_for_network(prev_sample)
    # prev_q_input = np.array([next_state], dtype=np.float32)
    prev_q_input = np.expand_dims(next_state, axis=0)
    prev_q_values = q_net.predict_on_batch(prev_q_input)
    return policy.select_action(prev_q_values, is_train)

def evaluate_qs(eval_batch_curr_states, net, num_updates=None, plot_qs=None, show=True):
    eval_qs = net.predict_on_batch(eval_batch_curr_states)
    max_qs = eval_qs.max(axis=1)
    if plot_qs is not None and num_updates is not None:
        plot_qs.append([num_updates, float(max_qs.mean())])
    if show and plot_qs is not None and len(plot_qs)>0:
        xs = [tup[0] for tup in plot_qs]
        ys = [tup[1] for tup in plot_qs]
        plt.plot(xs, ys)
        plt.show()

def evaluate_rewards(env, preproc, net, n_episodes, name, folder, iter_num=None, rewards_history=None,
                     eval_epsilon=0.05, plot=False):
    # Ensure the env passed here is different from the env used for training
    eval_sample = get_first_state(env=env, preproc=preproc, render=False)
    rewards_list = []
    episode_reward = 0.0
    greedy_policy = GreedyEpsilonPolicy(epsilon=eval_epsilon)
    vid = list()
    while n_episodes>0:
        if eval_sample.is_terminal:
            rewards_list.append(episode_reward)
            n_episodes -= 1
            eval_sample, eval_reward, observation = get_first_state(env=env, preproc=preproc, render=False, ret_reward=True)
            episode_reward = 0.0
            ## write the video
            vid.append(observation)
            vid = np.stack(vid)
            skvideo.io.vwrite(name('vid%2d'%(n_episodes),'mp4', '%s/vid%09d'%(folder,iter_num)), vid)
            vid = list()
        else:
            action = get_action(preproc=preproc, prev_sample=eval_sample, policy=greedy_policy, q_net=net,
                                use_gpu=True, is_train=False)
            eval_sample, eval_reward, observation = get_next_sample(env=env, preproc=preproc, action=action, prev_sample=eval_sample,
                                          render=False, ret_reward=True)
        episode_reward += eval_reward
        vid.append(observation)

    rewards_list = np.array(rewards_list)

    if iter_num is not None and rewards_history is not None:
        rewards_history.append([iter_num, float(rewards_list.mean()), float(rewards_list.std())])
    if plot and rewards_history is not None:
        xs = [tup[0] for tup in rewards_history]
        ys = [tup[1] for tup in rewards_history]
        plt.plot(xs, ys)
        plt.show()
    return rewards_list
