import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Type, List, Dict, Tuple

import gym
import numpy as np
import tensorflow as tf
from past.utils import old_div
from six import iteritems

from deepagent.agents.game_episodes import AbstractGameEpisodes
from deepagent.agents.policy_gradient import PolicyGradient
from deepagent.agents.util import normalize
from deepagent.envs.env_constants import EnvType
from deepagent.envs.environment_wrappers import AbstractTrainingEnvironment
from deepagent.envs.spaces import DeepAgentSpace
from deepagent.loggers.logging import get_tensorboard_writer, update_tensorboard_step


class ActionOutputType(Enum):
    NON_CONVOLUTIONAL = 'non-convolutional'
    CONVOLUTIONAL = 'convolutional'
    MULTI_DISCRETE = 'multi-discrete'


class AgentFactoryException(ValueError):
    pass


class AbstractAgentTrainer(ABC):
    def __init__(self, environment_wrapper: AbstractTrainingEnvironment, agent_factory: 'AbstractAgentFactory'):
        self.environment_wrapper = environment_wrapper
        self.agent_factory = agent_factory
        self.agents = self._create_agents()
        # from deepagent.experiments.params import params
        # self.writer = tf.summary.create_file_writer(params.ModuleParams.tensorboard_dir)
        self.writer = get_tensorboard_writer()
        self.min_report = {}
        self.max_report = {}
        self.avg_report = {}

    def _create_agents(self):
        return AbstractAgentTrainer._create_agents_static(self.agent_factory, self.environment_wrapper)

    @staticmethod
    def _create_agents_static(agent_factory, environment_wrapper):
        return AbstractAgentTrainer._create_agents_no_env(agent_factory, environment_wrapper.observation_space, environment_wrapper.action_space)

    @staticmethod
    def _create_agents_no_env(agent_factory, state_space_dict, action_space_dict):
        print(f'Creating policies for agents: {state_space_dict.keys()}')
        factory_name = agent_factory.__class__.__name__

        class PossiblyMissingAgentsDict(dict):
            def __missing__(self, key):
                raise NotImplementedError('{} does not create agents for {} units, '
                                          'but unit types are present in the environment.'.format(
                    factory_name, key
                ))

        agents = PossiblyMissingAgentsDict()

        for brain_type, action_space in iteritems(action_space_dict):
            state_space = state_space_dict[brain_type]

            from deepagent.experiments.params import params
            if params.EnvironmentParams.env_type == EnvType.atari or params.EnvironmentParams.env_type == EnvType.atari_rnd:
                brain_type = 'atari'
                state_space = DeepAgentSpace(image_spaces=[gym.spaces.Box(-1.0, 1.0, shape=state_space[0].shape)])
                action_space = DeepAgentSpace(vector_space=gym.spaces.Box(0.0, 1.0, shape=(action_space[0].n,)))

            agent = agent_factory.create_agent(brain_type, state_space, action_space)

            if agent is not None:
                print('Creating agent for type: {}'.format(brain_type))
                print('State Shape: {}\nAction Shape:{}'.format(state_space, action_space))
                agents[brain_type] = agent
                agent.print_summary()
            else:
                print('Warning: Not creating agent for type: {}'.format(brain_type))

        return agents

    def load_weights(self):
        for agent in list(self.agents.values()):
            agent.load_weights()

    def save_weights(self):
        for agent in list(self.agents.values()):
            agent.save_weights()

    def log_reward(self, reward, step):
        self.log_simple_value('Total Reward per Episode', reward, step)

    def log_score(self, score, step):
        self.log_simple_value('score', score, step)

    def log_games_played(self, games_played, step):
        self.log_simple_value('Number of Episodes Completed', games_played, step)

    def log_episodes_played(self, episodes_played, step):
        self.log_simple_value('Number of Episodes Completed', episodes_played, step)

    def log_simple_value(self, name, value, step):
        if name not in self.min_report.keys():
            self.min_report[name] = value
            self.max_report[name] = value
            self.avg_report[name] = [value]
        else:
            self.min_report[name] = value if self.min_report[name] > value else self.min_report[name]
            self.max_report[name] = value if self.max_report[name] < value else self.max_report[name]
            self.avg_report[name].append(value)

        with self.writer.as_default():
            tf.summary.scalar(name, value, step=step)

    def write_report_summary(self):
        with self.writer.as_default():
            for k in self.min_report.keys():
                tf.summary.text("Final Report", f'{k}  \nMin: {self.min_report[k]}  \nMax: {self.max_report[k]}  \nAvg: {sum(self.avg_report[k]) / len(self.avg_report[k])}', step=0)

            t = self.avg_report["Total Reward per Episode"]
            t.reverse()
            t = np.array(t)
            t = (t - np.min(t)) / np.ptp(t) #normalize 0-1
            reward_diffs = np.diff(t)

            batch_size = int(len(reward_diffs) / 100)
            threshold = .2
            outlier_threshold = 1

            print(f'batch_size: {batch_size} | diffs: {len(reward_diffs)}')

            if batch_size < 10 or len(reward_diffs) < 30:
                tf.summary.text("Final Report", f'Approx Episode Plateau:  \nN/A - Sample too Small', step=0)
            else:
                for i in range(0, len(reward_diffs), batch_size):
                    episode_index = len(reward_diffs) - i
                    diff_chunk = reward_diffs[i:i + batch_size]
                    if np.count_nonzero(diff_chunk > threshold) > outlier_threshold:
                        tf.summary.text("Final Report", f'Approx Episode Plateau:  \n{episode_index}', step=0)
                        break


        if hasattr(self.environment_wrapper.env, "write_report_summary"):
            self.environment_wrapper.env.write_report_summary()
        if hasattr(self.environment_wrapper.env, 'dictionary_env') and hasattr(self.environment_wrapper.env.dictionary_env, 'write_report_summary'):
                            self.environment_wrapper.env.dictionary_env.write_report_summary()

    @staticmethod
    def get_training_log_path():
        from deepagent.experiments.params import params
        return os.path.join(params.ModuleParams.weights_dir, 'training_log.txt')

    @staticmethod
    def get_agent_game_episodes_size(agent_episodes: Dict[Tuple[int, int], AbstractGameEpisodes], agent_type: str):
        """
        :param agent_episodes: Dictionary mapping (env_id, unit_id) to the game_episodes.
        :param agent_type: The type of agent to return the size for.
        :return: The sum of the number of episodes for all agents of agent_type.
        """
        size = 0
        for (id, a_type), v in agent_episodes.items():
            if agent_type == a_type:
                size += v.size()
        return size

    @staticmethod
    def get_all_game_episodes_size(agent_episodes: Dict[Tuple[int, int], AbstractGameEpisodes]):
        size = 0
        for v in agent_episodes.values():
            size += v.size()
        return size

    @staticmethod
    def remove_agent_game_episodes(agent_episodes: Dict[Tuple[int, int], AbstractGameEpisodes], agent_type: str):
        """
        Remove all game_episodes objects from the agent_epsiodes dict for the specified agent_type.
        :param agent_episodes: Dictionary mapping (env_id, unit_id) to the game_episodes.
        :param agent_type: The type of agent to return the size for.
        """
        for (id, type) in list(agent_episodes):
            if agent_type == type:
                del agent_episodes[(id, type)]

    @staticmethod
    def one_hot(action) -> np.array:
        if len(action.shape) > 1:
            return np.array(list(AbstractAgentTrainer._one_hot(a) for a in action))
        else:
            return AbstractAgentTrainer._one_hot(action)

    @staticmethod
    def _one_hot(action):
        flatten_size = action.shape[0]
        flatten = np.reshape(action, (flatten_size,))
        action_ridx = np.random.choice(flatten_size, p=flatten)
        binary = np.zeros((flatten_size,))
        binary[action_ridx] = 1.0
        binary_reshape = np.reshape(binary, action.shape)
        return binary_reshape

    @staticmethod
    def one_hot_idx(action) -> int:
        # This is expecting the action vector to by of the shape (x-dim, y-dim, num_actions) or (num_actions,), so the environment must match that
        if len(action.shape) > 1:
            flatten_size = action.shape[0] * action.shape[1] * action.shape[2]
        else:
            flatten_size = action.shape[0]
        flatten = np.reshape(action, (flatten_size,))
        action_ridx = np.random.choice(flatten_size, p=flatten)
        return action_ridx

    @staticmethod
    def one_hot_batch(actions: np.array) -> np.array:
        if isinstance(actions, list):
            first = True
            multi_discrete_list = []
            for action in actions:
                if first:
                    for r in range(action.shape[0]):
                        multi_discrete_list.append([])
                    first = False
                for r in range(action.shape[0]):
                    multi_discrete_list[r].append(AbstractAgentTrainer.one_hot_idx(action[r]))
            return multi_discrete_list
        else:
            for r in range(actions.shape[0]):
                actions[r] = AbstractAgentTrainer.one_hot(actions[r])
            return actions

    @abstractmethod
    def test(self, num_episodes: int, gamma: float, testing_suffix: str = ''):
        """
        Test the network.
        :param num_episodes: Total number of episodes to test for.
        :param gamma: The reward discount
        :param testing_suffix: For running multiple tests and saving results to different test files.
        """
        pass

    def population_test(self):
        pass

    @abstractmethod
    def train(self, steps: int, gamma: float, lam: float):
        """
        Train the network.
        :param steps: The number of serial steps to perform.
        :param gamma: The reward discount.
        :param lam: Lambda for GAE (bias variance trade-off).
        """
        pass

    def record(self):
        self.load_weights()

        # TODO: I'm not sure we should be making tensorboard logs for agents in a recording session
        for agent in self.agents.values():
            agent.set_tensorboard_writer(self.writer)

        step_count = 0
        update_tensorboard_step(step_count)
        batch_decision_request = self.environment_wrapper.reset()

        try:
            while True:
                step_count += 1
                update_tensorboard_step(step_count)

                batch_actions, _ = self.compute_action_value_batches(batch_decision_request, step_count)

                batch_decision_request, completed_steps, completed_episode_discounted_rewards = self.environment_wrapper.step(
                    action_batches=batch_actions)

        except Exception as e:
            self.environment_wrapper.close()
            raise e

    def compute_action_value_batches(self, batch_decision_request, step_count):
        batch_actions = {}  # type: Dict[str, Tuple[np.array, List[int]]]
        V1_batch = {}  # type: Dict[str, np.array]

        for brain_name, brain_decision_requests in batch_decision_request.items():
            batch_s0 = brain_decision_requests.batch_s0
            batch_mask = brain_decision_requests.batch_mask

            input_tensors = []
            for s in batch_s0:
                input_tensors.append(tf.cast(s, dtype=tf.float32))
            actions, value = self.agents[brain_name].get_local_policy_action(input_tensors)
            if isinstance(actions, list):
                brain_batch_actions = [a.numpy() for a in actions]
                with self.writer.as_default():
                    for i, action in enumerate(brain_batch_actions):
                        tf.summary.scalar('max_prob_action_'+str(i)+'_'+brain_name, np.mean(np.max(action, axis=-1)), step=step_count)
            else:
                brain_batch_actions = actions.numpy()
                brain_batch_actions = self.mask_and_renorm_batch(brain_batch_actions, batch_mask)
                with self.writer.as_default():
                    tf.summary.scalar('max_prob_'+brain_name, np.mean(np.max(brain_batch_actions, axis=-1)), step=step_count)
            try:
                brain_batch_actions = self.one_hot_batch(brain_batch_actions)
            except ValueError as e:
                print(f'{brain_name} exception during one_hot_batch: {e}')
                raise e
            batch_actions[brain_name] = brain_batch_actions
            V1_batch[brain_name] = value.numpy()

        return batch_actions, V1_batch

    @staticmethod
    def _valid_indices_bool_mask(mask):
        return np.nonzero(np.invert(mask))

    @staticmethod
    def _valid_indices_int_mask(mask):
        return np.nonzero(mask)

    @staticmethod
    def mask_and_renorm_batch(actions: np.array, masks: np.array) -> np.array:
        if isinstance(masks, np.ndarray) and masks.dtype == np.bool:
            actions[masks] = 0.0
            valid_indices = AbstractAgentTrainer._valid_indices_bool_mask
        else:
            actions[masks == 0] = 0.0
            valid_indices = AbstractAgentTrainer._valid_indices_int_mask
        for r in range(actions.shape[0]):
            actions[r] = normalize(actions[r], valid_indices(masks[r]))
        return actions

    @staticmethod
    def mask_and_renorm(action: np.array, action_mask: np.array) -> np.array:
        if len(action.shape) > 1:
            '''Not handling masking convolutional actions for now since unity can currently only send action masks
            for discrete action environments'''
            return action
        action[action_mask == 0.0] = 0.0  # apply mask
        action = normalize(action, action_mask)  # re-normalize
        return action

    def log_train_count(self, start, step_count, train_count, training_log):
        training_log.write('TrainCount:{}, Iter/Sec:{}, Step/Sec:{}'.format(
            train_count,
            old_div(train_count, (time.time() - start)),
            old_div(step_count, (time.time() - start))
        ))

    def postprocess_actions(self, batch_mask, brain_batch_actions, brain_name):
        action_space = self.environment_wrapper.action_space[brain_name]
        # TODO: Codify the way we handle spaces, this monkey patch dependency is bad
        if action_space.vector_space and hasattr(action_space.vector_space, 'is_onehot') and not action_space.vector_space.is_onehot:
            return brain_batch_actions

        brain_batch_actions = self.mask_and_renorm_batch(brain_batch_actions, batch_mask)
        brain_batch_actions = self.one_hot_batch(brain_batch_actions)
        return brain_batch_actions

    @staticmethod
    def instantiates_envs() -> bool:
        return False


class AbstractAgentFactory:
    def __init__(self, custom_objects: Dict = None):
        """
        :param custom_objects: Custom keras objects.
        """
        self.custom_objects = {} if custom_objects is None else custom_objects

    def multi_agent_training_class(self) -> Type[AbstractAgentTrainer]:
        pass

    def create_agent(self, agent_type: str, state_space: DeepAgentSpace, action_space: DeepAgentSpace) -> PolicyGradient:
        """
        Create a policy gradient to control agents for the specified type.
        :param type: The name of the agent type.
        :param state_space: The state space for the agent.
        :param action_space: The action space for the agent.
        :return:
        """
        pass
