"""Suggested Preprocessors."""

import numpy as np
from PIL import Image
import torch

#from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=1):
        pass

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""
        pass

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        pass

    def get_config(self):
        return {'history_length': self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    window: history length
    """

    def __init__(self, new_size, window):
        self.new_size = new_size
        self.window = window

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        state = Image.fromarray(state)
        state = state.convert('YCbCr')
        state = state.resize(self.new_size, Image.ANTIALIAS)
        state = np.array(state)
        return state[:,:,0]

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        state = Image.fromarray(state)
        state = state.convert('YCbCr')
        state = state.resize(self.new_size, Image.ANTIALIAS)
        state = np.array(state)
        return state[:,:,0]/256.0
        

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        state = np.zeros_like(state)

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        pass


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
        pass
