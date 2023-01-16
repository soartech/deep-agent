import os
from collections import deque
from typing import Dict, Tuple, List

import numpy as np
from six import iteritems
import tensorflow as tf

from deepagent.agents.util import get_atari_epinfos
from deepagent.agents.game_episodes import GameEpisodes
from deepagent.agents.common import AbstractAgentTrainer
from deepagent.envs.env_constants import EnvType
from deepagent.loggers.logging import update_tensorboard_step


class A2CAgentTrainer(AbstractAgentTrainer):
    def __init__(self, environment_wrapper, agent_factory):
        super(A2CAgentTrainer, self).__init__(environment_wrapper, agent_factory)

    def train(self, steps: int, gamma: float, lam: float):
        from deepagent.experiments.params import params

        if params.ModuleParams.prev_weights is not None:
            print('Loading previous weights before training')
            self.load_weights()
        else:
            print('Saving initial weights before training')
            self.save_weights()

        for agent in self.agents.values():
            agent.set_tensorboard_writer(self.writer)

        batch_decision_request = self.environment_wrapper.reset()

        from deepagent.experiments.params import utils as param_utils
        param_utils.write_params()

        # Open a log file for training
        training_log = open(self.get_training_log_path(), 'w', buffering=1)
        training_log.write('Starting Non-Concurrent training for params={}\n'.format(params.ModuleParams.name))

        episode_count = 0
        total_steps = 0

        ###
        import time
        start = time.time()
        train_count = 0
        ###

        agent_game_episodes = dict()

        training_log.write('Starting training for {} steps\n'.format(steps))

        epinfobuf = deque(maxlen=100)
        step_count = -1
        batch_actions, V1_batch = self.compute_action_value_batches(batch_decision_request, step_count)

        for step_count in range(steps):
            update_tensorboard_step(step_count)
            batch_decision_request_v = batch_decision_request.copy()
            V_batch = V1_batch.copy()
            if step_count % 10000 == 0:
                print('batch_actions={}'.format(batch_actions))

            batch_decision_request, completed_steps, completed_episode_discounted_rewards = self.environment_wrapper.step(action_batches=batch_actions)
            batch_actions, V1_batch = self.compute_action_value_batches(batch_decision_request, step_count)

            if params.EnvironmentParams.env_render:
                self.environment_wrapper.render()

            if self.environment_wrapper.env_type == EnvType.atari:
                epinfobuf.extend(get_atari_epinfos(self.environment_wrapper.get_env()))

            # add each agent's episode to memory
            for key, completed_step in completed_steps.completed_steps.items():
                unit_id, brain_type = key
                batch_id = batch_decision_request_v[brain_type].batch_id(unit_id)
                V = V_batch[brain_type][batch_id][0]

                if completed_step.t:
                    V1 = 0.0
                    tensorboard_name = str(brain_type) + '-reward'
                    self.log_simple_value(tensorboard_name, completed_step.total_game_reward, step_count)
                else:
                    bdr_batch_id = batch_decision_request[brain_type].batch_id(unit_id)
                    V1 = V1_batch[brain_type][bdr_batch_id][0]

                if key in agent_game_episodes:
                    game_episodes = agent_game_episodes[key]
                else:
                    game_episodes = GameEpisodes()
                    agent_game_episodes[key] = game_episodes

                game_episodes.append((completed_step.s0, V, completed_step.a, completed_step.r, V1, completed_step.t))

            # check if any agents should be updated
            for type, agent in self.agents.items():
                if self.get_agent_game_episodes_size(agent_game_episodes, type) >= params.TrainingParams.batch_size * params.TrainingParams.batches_per_update:
                    self.environment_wrapper.pause()
                    # calculate advantage for all episodes
                    # start1 = time.time()
                    for (id, episode_type), episodes in agent_game_episodes.items():
                        if type == episode_type:
                            episodes.complete(gamma)

                            self.agents[type].memory.add_lists(episodes.s, episodes.a, episodes.R, episodes.advantage)
                    # end1 = time.time()
                    # print('complete() time: ', (end1-start1))

                    self.remove_agent_game_episodes(agent_game_episodes, type)

                    # update networks
                    print('Updating network with ', agent.memory.num_batches(params.TrainingParams.batch_size), ' mini batches ', step_count, ' step count')
                    agent.update(step_count)

                    train_count += 1

                    self.environment_wrapper.resume()

            if len(completed_episode_discounted_rewards) > 0:
                float_rewards = [float(cedr) for cedr in completed_episode_discounted_rewards]
                self.log_reward(np.mean(float_rewards), step_count)
                self.log_games_played(episode_count, step_count)

                training_log.write(
                    'Episodes: {}, Serial Steps: {}, Total Steps: {}, Discnt Rwds: {}\n'.format(
                        episode_count,
                        step_count,
                        total_steps,
                        completed_episode_discounted_rewards
                    )
                )

            if step_count % 100000 == 0:
                print('saving population weights step ', step_count)
                for type, agent in self.agents.items():
                    agent.save_weights(prefix=str(step_count) + '_')

            if step_count != 0 and step_count % 10000 == 0:
                self.log_train_count(start, step_count, train_count, training_log)
                training_log.write('Saving Weights\n')
                self.save_weights()

        self.log_train_count(start, step_count, train_count, training_log)
        training_log.write('Saving Weights\n')
        self.save_weights()

        training_log.write('Training complete')

        training_log.close()
        self.environment_wrapper.close()

    def test(self, num_episodes: int, gamma: float, testing_suffix: str = ''):
        from deepagent.experiments.params import params
        testing_log = open(os.path.join(params.ModuleParams.weights_dir, 'testing_log.txt'), 'w', buffering=1)

        testing_log.write('Loading weights\n')
        self.load_weights()

        self.environment_wrapper.reset()

        testing_log.write('Starting test for params={}, weights={}\n'.format(params.ModuleParams.name, params.ModuleParams.weights_dir))

        rewards = []
        state = None

        step_count = 0

        for game in range(1, num_episodes + 1):
            while True:
                if state is None:
                    state = self.environment_wrapper.reset()

                actions = dict()

                # use target policy networks to choose actions for every unit

                for k, v in iteritems(state):
                    id = k[0]
                    type = k[1]

                    # expand dimension for batch size of 1
                    state_type = [np.expand_dims(s, axis=0) for s in v]
                    action = self.agents[type].get_local_policy_action(state_type)

                    # remove batch dimension
                    action = np.squeeze(action, axis=0)
                    action = self.mask_and_renorm(action, id)
                    actions[(id, type)] = self.one_hot(action)

                update_tensorboard_step(step_count)
                env = self.environment_wrapper.step(actions)
                step_count +=1

                if params.EnvironmentParams.env_render:
                    env.render()

                rewards.append(env.r)

                state = env.s1

                # calculate the reward and reset state/rewards if the episode is over
                # if all_terminal(env.terminal):
                #     testing_log.write('Episode count={}\n'.format(game))
                #     rewards = []
                #     state = None
                #     testing_log.write(
                #         'Games:{} | Wins:{} | WinRate:{}\n'.format(game, env.episode_wins, str(old_div(float(env.episode_wins), float(game)))))
                #     break

        testing_log.close()

    def record(self):
        raise NotImplementedError