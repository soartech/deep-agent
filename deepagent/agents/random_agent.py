import os
from collections import Counter
from typing import Dict, Tuple, List
from gym import spaces

import numpy as np

from deepagent.agents.common import AbstractAgentTrainer
from deepagent.agents.memory import BCMemory
from deepagent.agents.policy_gradient import PolicyGradient
from deepagent.loggers.logging import update_tensorboard_step

from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])


class RandomGradient(PolicyGradient):

    def __init__(self, deep_agent_action_space):
        self._action_space = deep_agent_action_space
        self.networks = {}

    def update_networks(self, step_count):
        pass

    def get_batches(self):
        pass

    def get_local_policy_action(self, state, step_count):
        actions = []
        import sys
        np.printoptions(precision=2, threshold=sys.maxsize)
        # for _ in range(0, state[0].shape[0]):
        for _ in range(len(state[0])):
            action = self._action_space.vector_space.sample()
            actions.append(action)
        return np.array(actions)

    def pre_train(self, bc_memory: BCMemory):
        raise NotImplementedError


class DebugAgentTester(AbstractAgentTrainer):
    def population_test(self):
        raise NotImplementedError

    def test(self, num_episodes: int, gamma: float, testing_suffix: str = ''):
        from deepagent.experiments.params import params
        testing_log = open(os.path.join(params.ModuleParams.weights_dir, 'testing_log' + testing_suffix + '.txt'), 'w',
                           buffering=1)
        reward_log = open(os.path.join(params.ModuleParams.weights_dir, 'reward_log' + testing_suffix + '.txt'), 'w',
                          buffering=1)

        testing_log.write(
            'Starting test for params={}\n'.format(params.ModuleParams.name))

        batch_decision_request = self.environment_wrapper.reset()

        discounted_rewards_sum = 0
        rewards_sum = 0
        step_count = 0
        total_steps = 0
        episode_count = 0
        list_discounted_rewards = []
        raw_counter = 0
        r_sum = 0.0
        r_sum_arra = []

        last_keys = {}

        testing_log.write('Testing for {} episodes\n'.format(num_episodes))
        import time
        start = time.time()
        while episode_count < num_episodes:

            batch_actions = {}  # type: Dict[str, Tuple[np.array, List[int]]]
            step_count += 1
            update_tensorboard_step(step_count)

            for brain_name, brain_decision_requests in batch_decision_request.items():
                batch_s0 = brain_decision_requests.batch_s0
                batch_mask = brain_decision_requests.batch_mask

                brain_batch_actions = self.agents[brain_name].get_local_policy_action(batch_s0, step_count)
                if not isinstance(self.agents[brain_name]._action_space.vector_space, spaces.MultiDiscrete):
                    brain_batch_actions = self.postprocess_actions(batch_mask, brain_batch_actions, brain_name)

                batch_actions[brain_name] = brain_batch_actions

            #time.sleep(0.01) #todo: slow things down for testing so it's easier to watch viz
            # print(f'steps/sec={step_count / (time.time() - start + 1e-6)}')
            batch_decision_request, completed_steps, completed_episode_discounted_rewards = self.environment_wrapper.step(
                action_batches=batch_actions)

            new_episode = False
            brains_to_completed_step_info = {}
            for key, val in completed_steps.completed_steps.items():
                if key not in last_keys:
                    new_episode = True
                brain_name = key[1]
                if brain_name not in brains_to_completed_step_info:
                    brains_to_completed_step_info[brain_name] = []
                brains_to_completed_step_info[brain_name].append(f'{val.r} {val.t} {key[0][1]}')

            last_keys = completed_steps.completed_steps.keys()

            # print(f'Rewards={np.array([c.r for c in completed_steps.completed_steps.values()])}')
            cs = Counter([c.r for c in completed_steps.completed_steps.values()])
            rewards_sum += sum(cs)
            if len(cs) > 0:
                for comp_step_rewards in cs:
                    r_sum += comp_step_rewards
                    if len(r_sum_arra) <= raw_counter:
                        r_sum_arra.append([])
                    r_sum_arra[raw_counter].append(r_sum)
                    raw_counter += 1

            total_steps += sum(len(actions) for actions in batch_actions.values())

            if params.EnvironmentParams.env_render:
                self.environment_wrapper.render()

            if len(completed_episode_discounted_rewards) > 0:
                unique_completed_episode_discounted_rewards = Counter(completed_episode_discounted_rewards)
                discounted_rewards_sum += sum(unique_completed_episode_discounted_rewards)
                episode_count += len(unique_completed_episode_discounted_rewards)
                list_discounted_rewards.extend(unique_completed_episode_discounted_rewards)

                self.log_reward(np.mean(completed_episode_discounted_rewards), step_count)
                self.log_games_played(episode_count, step_count)

                avg_rewards = rewards_sum / episode_count

                avg_discnt_rewards = discounted_rewards_sum / episode_count

                variance = sum([dr - avg_discnt_rewards for dr in list_discounted_rewards]) / (
                            episode_count - 1) if episode_count > 1 else 0

                testing_log.write(
                    'Episodes: {}, Serial Steps: {}, Total Steps: {}, Discnt Rwds: {}, Avg Discnt Rwds: {}, Total Rwds: {}, Avg Rwds: {}, Variance: {}\n'.format(
                        episode_count,
                        step_count,
                        total_steps,
                        completed_episode_discounted_rewards,
                        avg_discnt_rewards,
                        rewards_sum,
                        avg_rewards,
                        variance
                    )
                )
                r_sum = 0.0
                raw_counter = 0
        testing_log.close()
        string_of_stuff = ""
        print(r_sum_arra)
        for stuff in r_sum_arra:
            string_of_stuff += str(sum(stuff) / len(stuff)) + ","
        reward_log.write(string_of_stuff[:-1])
        reward_log.close()
        self.environment_wrapper.close()

    #Thunderdome only
    def compute_action_value_batches(self, batch_decision_request, step_count):
        import tensorflow as tf
        batch_actions = {}  # type: Dict[str, Tuple[np.array, List[int]]]
        V1_batch = {}  # type: Dict[str, np.array]

        for brain_name, brain_decision_requests in batch_decision_request.items():
            batch_s0 = brain_decision_requests.batch_s0
            batch_mask = brain_decision_requests.batch_mask

            input_tensors = []
            for s in batch_s0:
                input_tensors.append(tf.cast(s, dtype=tf.float32))
            actions = self.agents[brain_name].get_local_policy_action(input_tensors, step_count)
            if isinstance(actions, list):
                brain_batch_actions = [a for a in actions]
            else:
                brain_batch_actions = actions
            batch_actions[brain_name] = brain_batch_actions

        return batch_actions, V1_batch

    def train(self, steps: int, gamma: float, lam: float):
        raise NotImplementedError("ScriptedHeterogeneousAgent does not train!")

    def record(self):
        batch_decision_request = self.environment_wrapper.reset()
        step_count = 0
        try:
            while True:
                batch_actions = {}  # type: Dict[str, Tuple[np.array, List[int]]]
                step_count += 1

                for brain_name, brain_decision_requests in batch_decision_request.items():
                    batch_s0 = brain_decision_requests.batch_s0
                    batch_mask = brain_decision_requests.batch_mask

                    brain_batch_actions = self.agents[brain_name].get_local_policy_action(batch_s0, step_count)
                    brain_batch_actions = self.mask_and_renorm_batch(brain_batch_actions, batch_mask)
                    brain_batch_actions = self.one_hot_batch(brain_batch_actions)

                    batch_actions[brain_name] = brain_batch_actions

                batch_decision_request, completed_steps, completed_episode_discounted_rewards = self.environment_wrapper.step(
                    action_batches=batch_actions)
        except Exception as e:
            self.environment_wrapper.close()
            raise e


class RacerDijsktraTester(AbstractAgentTrainer):
    def population_test(self):
        raise NotImplementedError

    def test(self, num_episodes: int, gamma: float, testing_suffix: str = ''):
        from deepagent.experiments.params import params

        batch_decision_request = self.environment_wrapper.reset()

        step_count = 0
        episode_count = 0

        while episode_count < num_episodes:
            step_count += 1
            update_tensorboard_step(step_count)

            self.environment_wrapper.render()

            batch_actions = self.compute_action_value_batches(batch_decision_request, step_count)

            batch_decision_request, completed_steps, completed_episode_discounted_rewards = self.environment_wrapper.step(action_batches=batch_actions)

            if len(completed_episode_discounted_rewards) > 0:
                unique_completed_episode_discounted_rewards = Counter(completed_episode_discounted_rewards)
                episode_count += len(unique_completed_episode_discounted_rewards)

        self.environment_wrapper.close()

    #RACER only
    def compute_action_value_batches(self, batch_decision_request, step_count):
        action_batch = {}
        for brain_name, brain_decision_requests in batch_decision_request.items():
            idx = self.environment_wrapper.env.dictionary_env.envs[0].env.get_best_action()
            actions = np.zeros((1,8))
            actions[0, idx] = 1.0
            action_batch[brain_name] = actions
        return action_batch

    def train(self, steps: int, gamma: float, lam: float):
        raise NotImplementedError("ScriptedHeterogeneousAgent does not train!")