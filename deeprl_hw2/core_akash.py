import numpy as np
from skimage.measure import block_reduce
"""Core classes."""



class Sample:
    """Represents a reinforcement learning sample.

    Objects of this class are what are passed around to preprocessors. They are assembled from a StorageSample and
    an internal map of state id to state tensor maintained within the Replay Memory class

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
      Note that the state here is an encoding of all environment outputs
      so far. For the sake of tractability, you encode each sequence using
      only the last 4 environment outputs
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.

    Note: the state encodings stored should NOT be the outputs from the forward pass of a network since the parameters
    of the network change over time and reusing stale encodings of state will be detrimental. Therefore, the state here
    is the concatenation of the past 4 environment outputs.
    """

    def __init__(self, action, curr_ip_state, curr_ip_state_id, next_state, next_state_id, reward, is_terminal=False):
        self.curr_ip_state = np.array(curr_ip_state, dtype=np.uint8) # the grayscale images are represented by uint8
        self.curr_ip_state_id = curr_ip_state_id
        self.next_ip_state = np.array(next_state, dtype=np.uint8)
        self.next_state_id = next_state_id
        self.action = action
        self.reward = reward
        self.is_terminal = is_terminal

class StorageSample:
    """
    Objects of this class are what are actually stored in Replay Memory and sampled. Once the samples are obtained, the
    state ids are used to substitute in the actual tensors associated with those states.
    """
    def __init__(self, action, curr_ip_state_id, next_state_id, reward, is_terminal=False):
        self.curr_ip_state_id = curr_ip_state_id
        self.next_ip_state_id = next_state_id
        self.action = action
        self.reward = reward
        self.is_terminal = is_terminal


class Preprocessor(object):
    """Preprocessor base class.

    This is a suggested interface for the preprocessing steps. You may
    implement any of these functions. Feel free to add or change the
    interface to suit your needs.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """

    def __init__(self, replay_mem_cache):
        self.replay_mem_cache = replay_mem_cache

    ####################################################################################################################
    #                                 SHARED UTILITIES FOR ALL PREPROCESSORS                                           #
    ####################################################################################################################
    def rgb2grey(self, rgb_env_output):
        assert len(rgb_env_output.shape)==3, "rgb2grey converts a single image to grayscale"
        return np.dot(rgb_env_output[..., :3], [0.299, 0.587, 0.114])

    def downsample(self, grey_env_output, block_size = (2, 2)):
        if block_size is None:
            return grey_env_output
        return block_reduce(grey_env_output, block_size=block_size, func=np.mean)

    def crop(self, grey_env_output, crop_size=(84,84)):
        if crop_size is None:
            return grey_env_output
        return grey_env_output[:crop_size[0],:crop_size[1]]

    # Sample usage of the above 3 preprocessing methods is
    # preproc.crop(preproc.downsample(preproc.rgb2grey(observation)), crop_size=(78, 84))
    # where the crop size needs to be tuned for the specific game to approximately cover the playing area

    def preprocess(self, raw_image, crop_size, block_size, use_float=True):
        """
        Processes a single output from the environment (raw_state) which is assumed to have 3 channels (RGB). This is
        first converted to grayscale, then downsampled and finally cropped.
        :param raw_image:
        :param crop_size:
        :param block_size:
        :param use_float:
        :return:
        """
        return np.array(self.crop(self.downsample(self.rgb2grey(raw_image), block_size=block_size),
                                  crop_size=crop_size), dtype=np.float32 if use_float else np.uint8)

    ####################################################################################################################
    #                                     TEMPLATE FUNCTIONS TO BE OVERRIDDEN                                          #
    ####################################################################################################################

    def process_state_for_memory(self, env_output, curr_state_id, action, reward,is_terminal,
                                 crop_size=(84,84), block_size=(2,2), register_state=True):
        """
        My understanding:
        The preprocessor will be used to take the latest image  output from the environment and return something that
        can be populated into the Replay Memory. This function is responsible for doing this.
        """

        """Scale, convert to greyscale and store as uint8.

        Should be called just before appending this to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """

        processed_env_op = self.preprocess(env_output, crop_size=crop_size,block_size=block_size,use_float=False)
        # In this template function, it is assumed that the new processed env output is the new state by itself.
        # Specific preprocessors may instead update the previous state with this new processed env output to get the
        # new state.

        if register_state:
            new_state_id = self.replay_mem_cache.register_state(processed_env_op, curr_state_id)

            mem_sample = StorageSample(action=action, curr_ip_state_id=curr_state_id, next_state_id=new_state_id,
                                       reward=reward,
                                       is_terminal=is_terminal)
            self.replay_mem_cache.append(mem_sample)
            return mem_sample
        else:
            return processed_env_op

    def process_state_for_network(self, storage_sample):
        """
        My Understanding:

        This method will be used to process individual samples in a batch which will be used for training Q networks.
        Replay memory can store samples in a different format from what neural networks expect. This function converts
        them appropriately.

        Specifically, the replay memory stores StorageSamples instead of Samples. This method is responsible for the conversion

        """

        """Scale, convert to greyscale and store as float32.

        Preprocess the given state before giving it to the network.

        Should be called just before the action is selected.

        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        return (1.0/256.0)*np.array(np.concatenate(self.replay_mem_cache.state_id2state[storage_sample.curr_ip_state_id]), dtype=np.float32), \
               (1.0/256.0)*np.array(np.concatenate(self.replay_mem_cache.state_id2state[storage_sample.next_ip_state_id]), dtype=np.float32), \
               storage_sample.action, storage_sample.reward, storage_sample.is_terminal


    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        curr_states = []
        next_states = []
        actions = []
        rewards = []
        is_terminals = []
        for sample in samples:
            curr_state, next_state, action, reward, is_terminal = self.process_state_for_network(storage_sample=sample)
            curr_states.append(curr_state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            is_terminals.append(is_terminal)
        # Two separate lists are returned, one for each of the Q networks involved in evaluating the loss
        return np.array(curr_states, dtype=np.float32), np.array(next_states, dtype=np.float32), \
               np.array(actions, dtype=np.int64), np.array(rewards, dtype=np.float32), is_terminals

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        # Useful for things like reward clipping. The Atari environments
        # from DQN paper do this. Instead of taking real score, they
        # take the sign of the delta of the score.
        return 1 if reward > 0 else -1 if reward < 0 else 0


    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it possible to do history snapshots.
        """
        assert False, "This method must be implemented by subclass"


class ReplayMemory:
    """
    My understanding of replay memory:

    Ideally, this should store transitions of the form (curr_state, action, reward, next_state, is_terminal), i.e., it
    should store samples which can then be sampled for the actual learning.

    It may however be more efficient to store states as standalone entities and instead store tuples with ids into the
    state set. The sample method would then sample among the tuples, replace the state ids with their actual tensors
    and return values

    """

    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw saamples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.state_id2state = {} # mapping from state id to state
        self.state_id2count = {} # mapping from state id to to the count of number of occurrences of that state
        self.capacity = max_size
        self.memory = []
        self.curr_pos = 0
        self.max_state_id = 0

        # Try using dictionary keyed on episode id and another keyed on timestep within the episode
        # Sampling in this case would involve first sampling an episode and then sampling a timestep within that

    def register_state(self, state_tensor, prev_state_id):
        # Mapping the new state tensors to its id and acknowledging that the prev_state_id is now involved in another
        # tuple
        self.state_id2state[self.max_state_id] =  state_tensor
        self.state_id2count[self.max_state_id] = 1
        assert self.state_id2count.has_key(prev_state_id) or prev_state_id == -1, "Invalid prev_state_id to cache"
        if prev_state_id!=-1 or self.state_id2count.has_key(prev_state_id):
            self.state_id2count[prev_state_id] += 1
        else: # prev_state_id is -1 and it hasn't been registered
            self.state_id2count[prev_state_id] = 1
            self.state_id2state[prev_state_id] = np.zeros(shape=state_tensor.shape,dtype=state_tensor.dtype)
        self.max_state_id += 1
        return self.max_state_id-1

    # def append(self, state, action, reward):

    def decrement(self):
        rep_prev_state = self.memory[self.curr_pos].curr_ip_state_id
        rep_next_state = self.memory[self.curr_pos].next_ip_state_id
        self.state_id2count[rep_prev_state] -= 1
        if self.state_id2count[rep_prev_state] == 0:
            del self.state_id2count[rep_prev_state]
            del self.state_id2state[rep_prev_state]
        self.state_id2count[rep_next_state] -= 1
        if self.state_id2count[rep_next_state] == 0 == 0:
            del self.state_id2count[rep_next_state]
            del self.state_id2state[rep_next_state]

    def append(self, storage_sample):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        curr_ip_state_id = storage_sample.curr_ip_state_id
        next_state_id = storage_sample.next_ip_state_id
        assert self.state_id2state.has_key(curr_ip_state_id) and self.state_id2count.has_key(curr_ip_state_id) \
               and self.state_id2state.has_key(next_state_id) and self.state_id2count.has_key(next_state_id), \
            "Unknown states involved in the transition"
        if self.memory[self.curr_pos] is not None:
            self.decrement()

        self.memory[self.curr_pos] = storage_sample
        self.curr_pos = (self.curr_pos + 1) % self.capacity



    # def append(self, curr_ip_state_id, action, next_state_id, reward, is_terminal=False):
    #     if len(self.memory) < self.capacity:
    #         self.memory.append(None)
    #     assert self.state_id2state.has_key(curr_ip_state_id) and self.state_id2count.has_key(curr_ip_state_id) \
    #             and self.state_id2state.has_key(next_state_id) and self.state_id2count.has_key(next_state_id), \
    #             "Unknown states involved in the transition"
    #     if self.memory[self.curr_pos] is not None:
    #         # Need to hangle deletion of the storage sample being deleted
    #         self.decrement()
    #
    #     self.memory[self.curr_pos] = StorageSample(action=action, curr_ip_state_id=curr_ip_state_id,
    #                                                next_state_id=next_state_id, reward=reward,
    #                                                is_terminal=is_terminal)
    #     # Curr state should amd next state should be obtained from the process_state_for_memory method. Reward and
    #     # is_terminal should be obtained directly form the enviroment.
    #     self.curr_pos = (self.curr_pos+1)%self.capacity
    #
    # # def end_episode(self, final_state, is_terminal):
    # #     If Sample objects are directly being stored in memory, there is no need for this
    # #     raise NotImplementedError('This method should be overridden')

    def sample(self, batch_size, indexes=None):
        sample_ids = np.random.randint(low=0, high=len(self.memory),size=(batch_size,))
        return [self.memory[sample_id] for sample_id in sample_ids]

    def clear(self):
        self.memory = []
