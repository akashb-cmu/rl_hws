"""Main DQN agent."""

from tqdm import tqdm
import numpy as np
import pdb
from time import time
import keras.optimizers as O
import keras.losses as losses
from core_akash import ReplayMemory
from preprocessors_akash import  HistoryPreprocessor, AtariPreprocessor, PreprocessorSequence
import gym
from dqn_utils import *
from policy_akash import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    
    def __init__(self,
                 q_network,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 eval_freq,
                 batch_size,
                 epsilon,
                 num_actions,
                 mode='train'):

        self.Q, self.Q_cap = q_network[0], q_network[1]
#        self.preprocessor = preprocessor
#        self.memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.eval_freq = eval_freq
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.mode = mode
        self.train_data = None



    def compile(self, optimizer='adam', loss_func='huber_loss'):
        """
        Define the loss function and optimizer
        """

        if optimizer == 'adam':
            self.optimizer = O.Adam(lr=0.0001)
        else:
            assert 1, 'Optimizer not found'

        if loss_func == 'huber_loss':
            self.loss_func = losses.mean_squared_error
        else:
            assert 1, 'Loss function not found'

        self.Q.compile(loss=self.loss_func, optimizer=self.optimizer)
        self.Q_cap.compile(loss=self.loss_func, optimizer=self.optimizer)

    def calc_q_values(self, state, Q, flag):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.
        train_data should contain actual state values for the batch


        Return
        ------
        Q-values for the state(s)
        """

        if flag == 0:
            self.train_data = np.concatenate([st.state for st in state])
            return Q.predict_on_batch(self.train_data)
        elif flag == 1:
            return Q.predict_on_batch(np.concatenate([st.next_state for st in state]))
        
    def select_action(self, env, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """

        self.epsilon = 1 - (len(self.memory.memory)*1.0)/self.memory.max_size + 0.05
        if np.random.uniform() <= self.epsilon:
            index = env.action_space.sample()
        else:
            index = np.argmax(self.Q.predict(state))

        return index


    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        pass



    def fit_akash(self, train_env, eval_env, n_hist_history=4,
                  n_atari_history=4, crop_size=(84,84), block_size=(2,2), mem_size=1000000,
                  batch_size=32, eval_batch_size=100, gamma=0.99, start_epsilon=1.0, end_epsilon=0.1,
                  epsilon_anneal_steps=1000000,
                  tot_frames=10000000, eval_states=100, n_eval_episodes=20, eval_plot_period=30000, use_target_fix=True,
                  target_fix_freq=10000, burn_in_time=50000, video_freq=30000, render=False):
        train_replay_cache = ReplayMemory(max_size=mem_size)
        eval_replay_cache = ReplayMemory(max_size=5000)
        train_hist_preproc = HistoryPreprocessor(n_history=n_hist_history, flatten=False, crop_size=None,
                                                 block_size=None,replay_mem_cache=train_replay_cache)
        train_atari_preproc = AtariPreprocessor(n_history=n_atari_history, flatten=False, crop_size=crop_size,
                                              block_size=block_size,replay_mem_cache=None)

        train_preproc = PreprocessorSequence([train_atari_preproc, train_hist_preproc])

        eval_hist_preproc = HistoryPreprocessor(n_history=n_hist_history, flatten=False, crop_size=None,
                                                 block_size=None, replay_mem_cache=eval_replay_cache)

        eval_atari_preproc = AtariPreprocessor(n_history=n_atari_history, flatten=False, crop_size=crop_size,
                                                block_size=block_size, replay_mem_cache=None)

        eval_preproc = PreprocessorSequence([eval_atari_preproc, eval_hist_preproc])

        eval_sample = None

        for i in range(5000):
            if eval_sample is None or eval_sample.is_terminal:
                eval_sample = get_first_state(env=eval_env,preproc=eval_preproc,render=render,ret_reward=False)
            else:
                action = eval_env.action_space.sample()
                eval_sample = get_next_sample(env=eval_env,preproc=eval_preproc,action=action,prev_sample=eval_sample,
                                              render=render,ret_reward=False)

        q_eval_samples = eval_replay_cache.sample(batch_size=eval_batch_size)
        q_eval_curr_states, q_eval_next_states, q_eval_actions, q_eval_rewards, q_is_terms = eval_preproc.process_batch(q_eval_samples)

        prev_sample = None

        exploration_policy = LinearDecayGreedyEpsilonPolicy(start_epsilon=start_epsilon, end_epsilon=end_epsilon,
                                                            num_steps=epsilon_anneal_steps)

        running_loss = 0.0
        running_reward = []
        episode_loss = 0.0
        episode_reward = 0.0
        episode_count = 0.0
        frames_per_episode = 0.0
        eval_qs = []
        eval_rewards = []

        evaluate_qs(eval_batch_curr_states=q_eval_curr_states, net=self.Q, num_updates=0, plot_qs=eval_qs, show=False)

        for i in tqdm(range(tot_frames)):
            if prev_sample is None or prev_sample.is_terminal:
                if prev_sample is not None:
                    tqdm.write('Avg. Loss: %f, Avg. Reward: %f, Epi Loss: %f, Epi Reward: %f,  epsilon: %f, '
                               'buffer: %f, count: %d' %(running_loss / (i+1.0),
                                                 np.sum(running_reward)/(episode_count),
                                                 episode_loss/(frames_per_episode*1.0),
                                                 episode_reward,
                                                 exploration_policy.curr_epsilon,
                                                 (len(train_replay_cache.memory) * 1.0) / train_replay_cache.capacity,
                                                 frames_per_episode))

                episode_count += 1.0
                prev_sample = get_first_state(env=train_env, preproc=train_preproc,render=render,ret_reward=False)
                frames_per_episode = 1
                episode_reward = 0
                episode_loss = 0

            else:
                action = get_action(preproc=train_preproc, prev_sample=prev_sample, policy=exploration_policy,
                                    q_net=self.Q,
                                    is_train=True,use_gpu=True)
                prev_sample = get_next_sample(env=train_env, preproc=train_preproc, action=action,
                                              prev_sample=prev_sample, render=render)
                frames_per_episode += 1
 

            if i > burn_in_time:
                sampled_experience_batch = train_replay_cache.sample(batch_size=batch_size)
                batch_curr_states, batch_next_states, actions, rewards, is_terminals = train_preproc.process_batch(
                    sampled_experience_batch)

                y = self.Q.predict_on_batch(batch_curr_states) # should return (batch_size, num_actions)

                next_pred = self.Q_cap.predict_on_batch(batch_next_states)
                target_vec = gamma*(next_pred.max(axis=1)) + rewards
                next_actions = next_pred.argmax(axis=1)
                # target_vec should have size (batch_size, )

                for m in range(self.batch_size):
                    if is_terminals[m]:
                        y[m, next_actions[m]] = rewards[m]
                    else:
                        y[m, next_actions[m]] = target_vec[m]
                loss = self.Q.fit(batch_curr_states, y, epochs=1, batch_size=self.batch_size, verbose=0)
                running_loss += loss.history['loss'][0]
                running_reward.append(prev_sample.reward)
                episode_reward += prev_sample.reward
                episode_loss += loss.history['loss'][0]

                if i%eval_plot_period == 0:
                    evaluate_qs(eval_batch_curr_states=q_eval_curr_states, net=self.Q, num_updates=i-burn_in_time+1,
                                plot_qs=eval_qs, show=False)
                    evaluate_rewards(env=eval_env, preproc=eval_preproc, net=self.Q, n_episodes=n_eval_episodes,
                                     iter_num=i-burn_in_time, rewards_history=eval_rewards, eval_epsilon=0.05,
                                     plot=False)


            if i>burn_in_time and use_target_fix and (i+1)%target_fix_freq == 0:
                self.Q_cap.set_weights(self.Q.get_weights())

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """

        # count=0
        # while count < self.num_burn_in:
        #     ## reset environment
        #     next_state = torch.Tensor(1, self.preprocessor.window, *self.preprocessor.new_size)
        #     next_state_ = self.preprocessor.process_state_for_network(env.reset())
        #     next_state[0, 0] = torch.Tensor(next_state_)
        #     for k in range(1, self.preprocessor.window):
        #         next_state_, _, _, _ = env.step(env.action_space.sample())
        #         next_state_ = self.preprocessor.process_state_for_network(next_state_)
        #         next_state[0, k] = torch.Tensor(next_state_)
                
        #     ## preprocess memory
        #     next_state = torch.autograd.Variable(next_state).cuda()
        #     while(1):
        #         action = env.action_space.sample()
        #         observation, reward, done, _ = env.step(action)

        #         observation = self.preprocessor.process_state_for_network(observation)
        #         state = next_state.cpu().cuda()
        #         next_state[0,:-1] = next_state[0, 1:].data
        #         next_state[0, -1] = torch.autograd.Variable(torch.Tensor(observation).cuda()).data
                
        #         ## append the new state to replay_memory
        #         self.memory.add(state.cpu(), action, reward, next_state.cpu(), done)
        #         count+=1
        #         if count%1000==0:
        #             print count
        #         if done:
        #             break


        
        global_count=0
        next_state = np.zeros(tuple([1]+[self.preprocessor.window] +list(self.preprocessor.new_size)))

        for i in tqdm(range(num_iterations)):
            ## reset environment
            next_state[0,0] = self.preprocessor.process_state_for_network(env.reset())

            for k in range(1, self.preprocessor.window):
                next_state_, _, _, _ = env.step(env.action_space.sample())
                next_state[0,k] = self.preprocessor.process_state_for_network(next_state_)            
            running_loss = 0
            running_reward = list()
            count = 0
            time1 = 0
            start2 = time()
            while(1):
                action = self.select_action(env, next_state)
                observation, reward, done, _ = env.step(action)

                observation = self.preprocessor.process_state_for_network(observation)
                state = next_state.copy()
                next_state[0,:-1] = next_state[0, 1:]
                next_state[0, -1] = observation

                ## append the new state to replay_memory
                self.memory.add(state, action, reward, next_state, done)

                ## sample states
                sample = self.memory.sample(self.batch_size)
                ## Calculate Q-values
                Q_cap_values = self.calc_q_values(sample, self.Q_cap, flag=1)
                # same as next state tensors, with dim (batch_size, num_channels, row, col)
#                Q_values = self.calc_q_values(sample, self.Q, flag=0)
                
                ## loss function
                #y = np.zeros((self.batch_size, self.num_actions))
                y = self.calc_q_values(sample, self.Q, 0)
                
                for m in range(self.batch_size):
                    act = np.argmax(Q_cap_values[m])
                    if sample[m].is_terminal:
                        y[m,act] = sample[m].reward
                    else:
                        y[m,act] = sample[m].reward + self.gamma * np.max(Q_cap_values[m])

                start = time()                        
                loss = self.Q.fit(self.train_data, y, epochs=1, batch_size=self.batch_size, verbose=0)
                end = time()
#                pdb.set_trace()
                running_loss += loss.history['loss'][0]
                running_reward.append(reward)
                count+=1.0

                global_count+=1.0

                ## target fixing
                if (global_count+1) % self.target_update_freq == 0 :
                    self.Q_cap.set_weights(self.Q.get_weights())

                time1 += end-start
                end2 = time()
                if done:
                    break
                
            tqdm.write('Loss: %f, reward: %f, epsilon: %f, buffer: %f, time: %f, time2: %f, count: %d' % (running_loss/(count+1.0), np.sum(running_reward), self.epsilon, (len(self.memory.memory)*1.0)/self.memory.max_size, time1/(count*1.0), (end2-start2-time1)/(count*1.0),count))

#            if (i+1) % self.eval_freq == 0:  
#                running_reward_test = self.evaluate(env, num_episodes=10, max_episode_length=max_episode_length)
#                tqdm.write('EVAL:: avg_reward: %f, std_dev: %f' %(np.mean(running_reward_test), np.std(running_reward_test)))
        pdb.set_trace()
                
                

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        running_reward = []
        for i in tqdm(range(num_episodes)):
            ## reset environment
            next_state = torch.Tensor(1, self.preprocessor.window, *self.preprocessor.new_size)
            next_state_ = self.preprocessor.process_state_for_network(env.reset())
            next_state[0, 0] = torch.Tensor(next_state_)
            for k in range(1, self.preprocessor.window):
                next_state_, _, _, _ = env.step(env.action_space.sample())
                next_state_ = self.preprocessor.process_state_for_network(next_state_)
                next_state[0, k] = torch.Tensor(next_state_)

            ## preprocess memory
            next_state = torch.autograd.Variable(next_state).cuda()

            mini_reward = 0
            #for j in range(max_episode_length):
            while(1):
                action = self.select_action(env, next_state)                    
                observation, reward, done, _ = env.step(action)

                observation = self.preprocessor.process_state_for_network(observation)
                state = next_state.cpu().cuda()
                next_state[0,:-1] = next_state[0, 1:].data

                mini_reward+=reward
                if done:
                    break
            running_reward.append(mini_reward)

        return running_reward
