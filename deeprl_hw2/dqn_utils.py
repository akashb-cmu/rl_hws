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
#from policy import *
import numpy as np

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
        return next_sample, reward

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

def evaluate_qs(eval_batch_curr_states, net, use_gpu, num_updates=None, plot_qs=None, show=True):
    if use_gpu:
        eval_batch_curr_states = eval_batch_curr_states.cuda()
    eval_qs = net(eval_batch_curr_states)
    if not use_gpu:
        max_qs = eval_qs.max(1)[0].data.numpy()
    else:
        max_qs = eval_qs.max(1)[0].cpu().data.numpy()
    if plot_qs is not None and num_updates is not None:
        plot_qs.append((num_updates, max_qs.mean()))
    if show and plot_qs is not None and len(plot_qs)>0:
        xs = [tup[0] for tup in plot_qs]
        ys = [tup[1] for tup in plot_qs]
        plt.plot(xs, ys)
        plt.show()

def evaluate_rewards(env, preproc, net, n_episodes, use_gpu, video_file_path=None, iter_num=None, rewards_history=None,
                     eval_epsilon=0.05, plot=False):
    # Ensure the env passed here is different from the env used for training
    if video_file_path is not None:
        env = gym.wrappers.Monitor(env,directory=video_file_path)#, video_callable=,write_upon_reset=)
    eval_sample = get_first_state(env=env, preproc=preproc, render=False)

    avg_reward = 0.0
    tot_episodes = n_episodes

    greedy_policy = GreedyEpsilonPolicy(epsilon=eval_epsilon)

    while n_episodes>0:
        if eval_sample.is_terminal:
            n_episodes -= 1
            eval_sample, eval_reward = get_first_state(env=env, preproc=preproc, render=False, ret_reward=True)
        else:
            action = get_action(preproc=preproc, prev_sample=eval_sample, policy=greedy_policy, q_net=net,
                                use_gpu=use_gpu, is_train=False)
            eval_sample, eval_reward = get_next_sample(env=env, preproc=preproc, action=action, prev_sample=eval_sample,
                                          render=False, ret_reward=True)
        # avg_reward += eval_sample.reward
        avg_reward += eval_reward
        if eval_reward!=eval_sample.reward:
            print("Rewards are different with actual reward=%f and clipped reward=%f"%(eval_reward, eval_sample.reward))
    avg_reward = avg_reward / tot_episodes
    if iter_num is not None and rewards_history is not None:
        rewards_history.append((iter_num, avg_reward))
    if plot and rewards_history is not None:
        xs = [tup[0] for tup in rewards_history]
        ys = [tup[1] for tup in rewards_history]
        plt.plot(xs, ys)
        plt.show()
    return avg_reward
