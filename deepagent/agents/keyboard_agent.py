import platform
if platform.system() == 'Windows':
    from msvcrt import getch
else:
    import getch

import numpy as np
from absl import flags

from deepagent.agents.memory import BCMemory
from deepagent.agents.policy_gradient import PolicyGradient

FLAGS = flags.FLAGS
FLAGS([''])


class KeyboardGradient(PolicyGradient):

    def __init__(self, deep_agent_action_space):
        self._action_space = deep_agent_action_space
        self._num_actions = self._action_space.vector_space.shape[0]
        self._input_chars = list(r'xwdaczeq12345678')[:self._num_actions]
        #self._input_chars = list(r'`12345678qwertyuiopasdfghjkl;zxcvbnm,./')[:self._num_actions]
        self._action_map = {k: v for v, k in enumerate(self._input_chars)}
        self.networks = {}

    def update_networks(self, step_count):
        pass

    def get_batches(self):
        pass

    def get_local_policy_action(self, state, step_count):
        while True:
            try:
                input_char = getch().decode('utf-8')
                action_idx = self._action_map[input_char]
                print(f'action chosen: {action_idx}, char_entered: {input_char}')
                break
            except KeyError as e:
                print(
                    f'Bad input char, received {input_char}. Please input a valid char, choices:\n{self._input_chars}')

        action = np.zeros(self._action_space.vector_space.shape)
        action[action_idx] = 1.0
        actions = np.tile(action, (len(state[0]), 1))
        return actions

    def pre_train(self, bc_memory: BCMemory):
        raise NotImplementedError
