"""Main DQN agent."""

import torch
from tqdm import tqdm
import numpy as np
import pdb

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
                 preprocessor,
                 memory,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 epsilon,
                 mode='train'):

        self.Q, self.Q_cap = q_network[0], q_network[1]
        self.preprocessor = preprocessor
        self.memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.mode = mode
        

    def compile(self, optimizer='adam', loss_func='huber_loss'):
        """
        Define the loss function and optimizer
        """

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.001)
        else:
            assert 1, 'Optimizer not found'

        if loss_func == 'huber_loss':
            self.loss_func = torch.nn.SmoothL1Loss()
        else:
            assert 1, 'Loss function not found'

    def calc_q_values(self, state, Q, flag):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        if flag == 0:
            return Q(torch.cat([st.state for st in state]))
        elif flag == 1:
            return Q(torch.cat([st.next_state for st in state]))
        
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
            value, index = torch.max(self.Q(state).cpu(), 1)
            index = index.cpu().data.numpy()[0,0]

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

        for i in tqdm(range(num_iterations)):
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

            running_loss = 0
            
#            for j in range(max_episode_length):
            count = 0
            while(1):
                action = self.select_action(env, next_state)                    
                observation, reward, done, _ = env.step(action)

                observation = self.preprocessor.process_state_for_network(observation)
                state = next_state.cpu().cuda()
                next_state[0,:-1] = next_state[0, 1:].data
                next_state[0, -1] = torch.autograd.Variable(torch.Tensor(observation).cuda()).data

                ## append the new state to replay_memory
                self.memory.add(state, action, reward, next_state, done)

                ## sample states
                sample = self.memory.sample(self.batch_size)

                ## Calculate Q-values
                Q_values = self.calc_q_values(sample, self.Q, flag=0)
                Q_cap_values = self.calc_q_values(sample, self.Q_cap, flag=1)

                ## loss function
                y = torch.autograd.Variable(torch.Tensor(self.batch_size))
                for i in range(self.batch_size):
                    if sample[i].is_terminal:
                        y[i] = sample[i].reward
                    else:
                        y[i] = sample[i].reward + self.gamma * torch.max(Q_cap_values[i].data.cpu())

                y = y.cuda()
                y_cap = torch.stack([Q_values[i, sam.action] for i,sam in enumerate(sample)])
                
                self.optimizer.zero_grad()
                loss = self.loss_func(y_cap,y)
                loss.backward()

                running_loss += loss.data[0]
                count+=1.0
                
                if done:
                    break

            ## target fixing
            self.Q_cap.load_state_dict(self.Q.state_dict())
                
            running_reward = self.evaluate(env, num_episodes=10, max_episode_length=max_episode_length)
            tqdm.write('Loss: %f, avg_reward: %f, std_dev: %f, epsilon: %f, buffer: %f' % (running_loss/(count+1.0), np.mean(running_reward), np.std(running_reward), self.epsilon, (len(self.memory.memory)*1.0)/self.memory.max_size))
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
            for j in range(max_episode_length):                    
                action = self.select_action(env, next_state)                    
                observation, reward, done, _ = env.step(action)

                observation = self.preprocessor.process_state_for_network(observation)
                state = next_state.cpu().cuda()
                next_state[0,:-1] = next_state[0, 1:].data

                mini_reward+=reward
            running_reward.append(mini_reward)

        return running_reward
