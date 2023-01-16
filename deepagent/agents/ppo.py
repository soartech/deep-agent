import os
import time
from typing import Dict, Tuple, List

import numpy as np

from deepagent.agents.common import AbstractAgentTrainer
from deepagent.agents.game_episodes import GameEpisodesGeneralizedAdvantage
from deepagent.loggers.logging import update_tensorboard_step

def log_state_tuple(tup,opath):
    val = np.array(tup,dtype=np.object)
    with open(opath,'ab') as ofp:
        np.save(ofp,val,allow_pickle=True)

class PPOAgentTrainer(AbstractAgentTrainer):

    def __init__(self, environment_wrapper, agent_factory):
        super(PPOAgentTrainer, self).__init__(environment_wrapper, agent_factory)

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

        update_tensorboard_step(step=0)
        batch_decision_request = self.environment_wrapper.reset()

        from deepagent.experiments.params import utils as param_utils
        param_utils.write_params()

        # Open a log file for training
        training_log = open(self.get_training_log_path(), 'w', buffering=1)
        training_log.write('Starting Non-Concurrent training for params={}\n'.format(params.ModuleParams.name))

        episode_count = 0
        total_steps = 0

        ###
        start = time.time()
        train_count = 0
        ###

        agent_game_episodes = dict()

        training_log.write('Starting training for {} steps\n'.format(steps))

        step_count = -1
        batch_actions, V1_batch = self.compute_action_value_batches(batch_decision_request, step_count)

        for step_count in range(steps + 1):
            V_batch = V1_batch.copy()
            batch_decision_request_v = batch_decision_request.copy()
            if step_count % 10000 == 0:
                print('batch_actions={}'.format(batch_actions))

            batch_decision_request, completed_steps, completed_episode_discounted_rewards = self.environment_wrapper.step(
                action_batches=batch_actions)
            batch_actions, V1_batch = self.compute_action_value_batches(batch_decision_request, step_count)

            total_steps += sum(len(actions) for actions in batch_actions.values())
            episode_count += len(completed_episode_discounted_rewards)

            if params.EnvironmentParams.env_render:
                self.environment_wrapper.render()

            total_game_rewards = {}

            # add each agent's episode to memory
            for key, completed_step in completed_steps.completed_steps.items():
                unit_id, brain_type = key
                batch_id = batch_decision_request_v[brain_type].batch_id(unit_id)

                V = V_batch[brain_type][batch_id][0]

                if completed_step.t:
                    V1 = 0.0
                    if brain_type not in total_game_rewards:
                        total_game_rewards[brain_type] = []
                    total_game_rewards[brain_type].append(completed_step.total_game_reward)
                else:
                    bdr_batch_id = batch_decision_request[brain_type].batch_id(unit_id)
                    V1 = V1_batch[brain_type][bdr_batch_id][0]

                if key in agent_game_episodes:
                    game_episodes = agent_game_episodes[key]
                else:
                    game_episodes = GameEpisodesGeneralizedAdvantage()
                    agent_game_episodes[key] = game_episodes

                game_episodes.append((completed_step.s0, V, completed_step.a, completed_step.r, V1, completed_step.t))

            # check if any agents should be updated
            for type, agent in self.agents.items():
                if self.get_agent_game_episodes_size(agent_game_episodes,
                                                     type) >= params.TrainingParams.batch_size * params.TrainingParams.batches_per_update:
                    self.environment_wrapper.pause()
                    # calculate advantage for all episodes
                    for (id, episode_type), episodes in agent_game_episodes.items():
                        if type == episode_type:
                            episodes.complete(gamma, lam, params.TrainingParams.episodic_reward)

                            self.agents[type].memory.add_lists(episodes.s, episodes.a, episodes.R, episodes.GAE)

                    self.remove_agent_game_episodes(agent_game_episodes, type)

                    # update networks
                    print('Updating network with ', agent.memory.num_batches(params.TrainingParams.batch_size), ' mini batches ', step_count, ' step count')
                    agent.update(step_count)

                    train_count += 1

                    self.environment_wrapper.resume()
                    import os, psutil, datetime
                    print(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, ' MB ')

            if len(completed_episode_discounted_rewards) > 0:
                float_rewards = [float(cedr) for cedr in completed_episode_discounted_rewards]
                self.log_reward(np.mean(float_rewards), step_count)
                self.log_games_played(episode_count, step_count)

                for brain_name, reward_list in total_game_rewards.items():
                    tensorboard_name = str(brain_name) + '-reward'
                    self.log_simple_value(tensorboard_name, np.mean(reward_list), step_count)

                training_log.write(
                    'Episodes: {}, Serial Steps: {}, Total Steps: {}, Discnt Rwds: {}\n'.format(
                        episode_count,
                        step_count,
                        total_steps,
                        completed_episode_discounted_rewards
                    )
                )

            if step_count % 250000 == 0:
                print('saving population weights step ', step_count)
                for type, agent in self.agents.items():
                    agent.save_weights(prefix=str(step_count) + '_')

            if step_count != 0 and step_count % 10000 == 0:
                self.log_train_count(start, step_count, train_count, training_log)
                training_log.write('Saving Weights\n')
                self.save_weights()

            update_tensorboard_step(step_count)

        self.log_train_count(start, step_count, train_count, training_log)
        training_log.write('Saving Weights\n')
        self.save_weights()

        training_log.write('Training complete')

        training_log.close()
        self.environment_wrapper.close()

    def population_test(self):
        for agent in self.agents.values():
            agent.set_tensorboard_writer(self.writer)

        from deepagent.agents.population_test import PopulationTest
        tp = PopulationTest(self.agents, self.environment_wrapper, self.compute_action_value_batches)
        tp.test()

    def test(self, num_episodes: int, gamma: float, testing_suffix: str = ''):
        from deepagent.experiments.params import params
        testing_log = open(os.path.join(params.ModuleParams.weights_dir, 'testing_log' + testing_suffix + '.txt'), 'w',
                           buffering=1)

        for agent in self.agents.values():
            agent.set_tensorboard_writer(self.writer)

        testing_log.write('Loading weights\n')
        self.load_weights()

        testing_log.write(
            'Starting test for params={}, weights={}\n'.format(params.ModuleParams.name,
                                                               params.ModuleParams.weights_dir))

        batch_decision_request = self.environment_wrapper.reset()

        step_count = 0
        total_steps = 0
        episode_count = 0
        batch_actions = {}  # type: Dict[str, Tuple[np.array, List[int]]]

        state_logging_path = os.path.join(params.ModuleParams.weights_dir, 'state_log' + testing_suffix + '.npy')
        previous_episode_count = -1

        while episode_count < params.TestingParams.num_episodes:
            update_tensorboard_step(step_count)
            step_count += 1

            batch_actions, _ = self.compute_action_value_batches(batch_decision_request, step_count)

            batch_decision_request, completed_steps, completed_episode_discounted_rewards = self.environment_wrapper.step(
                action_batches=batch_actions)

            if episode_count != previous_episode_count:
                print('new episode',episode_count,step_count,completed_episode_discounted_rewards)

            if len(completed_episode_discounted_rewards) > 0:
                print('episode rewards',episode_count,step_count,completed_episode_discounted_rewards)

            if  params.TestingParams.do_state_logging and episode_count < params.TestingParams.state_logging_episode_count:
                otup = 'state_tup', episode_count, step_count, batch_actions, batch_decision_request, completed_steps, completed_episode_discounted_rewards
                log_state_tuple(otup,state_logging_path)
                if episode_count != previous_episode_count or len(completed_episode_discounted_rewards) > 0:
                    print('state_logging',episode_count,step_count,completed_episode_discounted_rewards)

            total_steps += sum(len(actions) for actions in batch_actions.values())
            previous_episode_count = episode_count
            episode_count += len(completed_episode_discounted_rewards)


            if params.EnvironmentParams.env_render:
                self.environment_wrapper.render()

            total_game_rewards = {}

            # compute per brain rewards
            for key, completed_step in completed_steps.completed_steps.items():
                unit_id, brain_type = key

                if completed_step.t:
                    if brain_type not in total_game_rewards:
                        total_game_rewards[brain_type] = []
                    total_game_rewards[brain_type].append(completed_step.total_game_reward)

            if len(completed_episode_discounted_rewards) > 0:
                float_rewards = [float(cedr) for cedr in completed_episode_discounted_rewards]
                self.log_reward(np.mean(float_rewards), step_count)
                self.log_games_played(episode_count, step_count)

                for brain_name, reward_list in total_game_rewards.items():
                    tensorboard_name = str(brain_name) + '-reward'
                    self.log_simple_value(tensorboard_name, np.mean(reward_list), step_count)

                testing_log.write(
                    'Episodes: {}, Serial Steps: {}, Total Steps: {}, Discnt Rwds: {}\n'.format(
                        episode_count,
                        step_count,
                        total_steps,
                        completed_episode_discounted_rewards
                    )
                )

        testing_log.close()
        self.environment_wrapper.close()
