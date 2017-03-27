"""Suggested Preprocessors."""

import numpy as np
from PIL import Image
from collections import deque


# import utils
# from deeprl_hw2.core import Preprocessor
from core import Preprocessor, StorageSample, Sample

class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    This preprocessor is basically a stacker/flattener. It consumes inputs which are of the form

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, n_history=1, flatten=False, crop_size=(84, 84), block_size=(2, 2), replay_mem_cache=None):
        self.n_history = n_history
        self.flatten = flatten
        self.crop_size = crop_size
        self.block_size = block_size
        self.replay_mem_cache = replay_mem_cache
        self.memory = deque() # memory of the past k frames
        # This memory is different from the replay memory in that it is used for preprocessing purposes and is limited
        # to size k

    def stack_history(self):
        """
        There are two ways of stacking history.
        1. Concatenate vectors. This is used for a linear function approximator
        2. Channelize vectors. This is useful for convolutional Q networks.

        In either case, an extra dimension is introduced since torch expects batches by defaults

        :param flatten: If True, history vectors are simply concatenated
        :param use_float: Whether the returned numpy vector should be a float/uint array
        :return: Concatenated/stacked vectors
        """
        if self.flatten:
            return np.array([np.concatenate([state for state in self.memory])],
                            dtype=np.uint8)
        else:
            return np.array([[state for state in self.memory]],dtype=np.uint8)

    def add_to_memory(self, mem_item):
        if len(self.memory) > 0:
            assert (len(self.memory) == self.n_history), "History improperly populated"
            self.memory.popleft()
        else:
            ip_size = mem_item.shape
            dummy_item = np.array(np.zeros(ip_size),dtype=np.uint8)
            if self.flatten:
                dummy_item = dummy_item.flatten()
            for i in range(self.n_history - 1):
                self.memory.append(dummy_item)
        self.memory.append(mem_item.flatten() if self.flatten else mem_item)

    def process_state_for_memory(self, env_output, curr_state_id=None, action=None, reward=None,is_terminal=None,
                                 preprocess=True, register_state=True):
        if preprocess:
            env_output = self.preprocess(env_output,crop_size=self.crop_size,block_size=self.block_size,use_float=False)
        self.add_to_memory(mem_item=env_output)
        state_for_mem = self.stack_history()
        if not register_state:
            return state_for_mem
        else:
            assert None not in [curr_state_id, action, reward, is_terminal, self.replay_mem_cache], "Requisite args not" \
                   " provided for state registration!"
            new_state_id = self.replay_mem_cache.register_state(state_for_mem, curr_state_id)
            mem_sample = StorageSample(action=action,curr_ip_state_id=curr_state_id,next_state_id=new_state_id,reward=reward,
                                       is_terminal=is_terminal)
            self.replay_mem_cache.append(mem_sample)
            return mem_sample

    def process_state_for_network(self, storage_sample):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        assert self.replay_mem_cache is not None, "Replay cache has not been configured"
        # Returns the states corresponding to the current_state and the next state
        return super(HistoryPreprocessor, self).process_state_for_network(storage_sample=storage_sample)

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        assert self.replay_mem_cache is not None, "Replay cache has not been configured"
        return super(HistoryPreprocessor, self).process_batch(samples=samples)

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.memory.clear()

    def get_config(self):
        return {'n_history': self.n_history,
                'flatten': self.flatten,
                'replay_mem_cache': self.replay_mem_cache}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, n_history=1, flatten=False, crop_size=(84, 84), block_size=(2, 2),
                 replay_mem_cache=None):
        self.crop_size = crop_size
        self.block_size = block_size
        self.n_history = n_history
        self.flatten = flatten
        self.replay_mem_cache = replay_mem_cache
        self.memory = deque()

    def add_to_memory(self, mem_item):
        if len(self.memory) > 0:
            assert (len(self.memory) == self.n_history), "History improperly populated"
            self.memory.popleft()
        else:
            ip_size = mem_item.shape
            dummy_item = np.array(np.zeros(ip_size), dtype=np.uint8)
            if self.flatten:
                dummy_item = dummy_item.flatten()
            for i in range(self.n_history-1):
                self.memory.append(dummy_item)
        self.memory.append(mem_item.flatten() if self.flatten else mem_item)

    def process_history(self):
        assert(len(self.memory) > 0), "Can't process empty history!"
        max_state = self.memory[0]
        for i in range(1, self.n_history):
            max_state = np.maximum(max_state, self.memory[i])
        return max_state

    def process_state_for_memory(self, env_output, curr_state_id=None, action=None, reward=None,is_terminal=None,
                                 preprocess=True, register_state=True):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        if preprocess:
            env_output = self.preprocess(env_output,crop_size=self.crop_size,block_size=self.block_size,use_float=False)
        self.add_to_memory(mem_item=env_output)
        state_for_mem = self.process_history()
        if not register_state:
            return state_for_mem
        else:
            assert None not in [curr_state_id, action, reward, is_terminal, self.replay_mem_cache], "Requisite args " \
                   "not provided for state registration!"
            new_state_id = self.replay_mem_cache.register_state(state_for_mem, curr_state_id)
            mem_sample = StorageSample(action=action, curr_ip_state_id=curr_state_id, next_state_id=new_state_id,
                                       reward=reward, is_terminal=is_terminal)
            self.replay_mem_cache.append(mem_sample)
            return mem_sample

    def process_state_for_network(self, storage_sample):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        assert self.replay_mem_cache is not None, "Replay cache has not been configured"
        return super(AtariPreprocessor, self).process_state_for_network(storage_sample=storage_sample)

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        assert self.replay_mem_cache is not None, "Replay cache has not been configured"
        return super(AtariPreprocessor, self).process_batch(samples=samples)

    def reset(self):
        self.memory.clear()


class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def process_state_for_memory(self, env_output, curr_state_id=None, action=None, reward=None, is_terminal=None,
                                 preprocess=True, register_state=True):
        env_output= self.preprocessors[0].process_state_for_memory(env_output, curr_state_id=curr_state_id,
                                           action=action, reward=reward,is_terminal=is_terminal,
                                           preprocess=True, register_state=False if len(self.preprocessors)>1 else
                                                                                                         register_state)
        if len(self.preprocessors) > 1:
            for i in range(1, len(self.preprocessors)-1):
                env_output = self.preprocessors[i].process_state_for_memory(env_output, curr_state_id=curr_state_id,
                                        action=action, reward=reward, is_terminal=is_terminal,
                                        preprocess=False, register_state=False)

            env_output = self.preprocessors[-1].process_state_for_memory(env_output, curr_state_id=curr_state_id,
                                        action=action, reward=reward, is_terminal=is_terminal,
                                        preprocess=False, register_state=register_state)
        return env_output


    def process_state_for_network(self, storage_sample):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        return self.preprocessors[-1].process_state_for_network(storage_sample=storage_sample)


    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        return self.preprocessors[-1].process_batch(samples=samples)

    def reset(self):
        for preproc in self.preprocessors:
            preproc.reset()

    def get_config(self):
        return {
                'preprocessors': self.preprocessors
               }

