import argparse
import os
import re

import numpy as np

from deepagent.experiments.experiment_runner import run_experiment
from deepagent.experiments.params.params import ExperimentType
from deepagent.experiments.params.utils import load_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', '-p', help="The trained params module to test against the baselines params: ex: "
                                               "'params1', 'params2', etc.", required=True)
    parser.add_argument('--weights', '-w', type=str,
                        help='The training weights folder to test with. Training directories are numbered in ascending '
                             'order and found in subdirectories of the parameters folders.', default=None)
    parser.add_argument('--num_runs', '-nr', type=int, help='The number of times to run the scripted_params and the '
                                                            'params each.', default=10)
    parser.add_argument('-u', action='store_true', help='This argument is just here because pycharm is dumb at '
                                                        'executing modules on remote interpreters. The workaround means '
                                                        'that it adds a "-u" field in the wrong place.')
    parser.add_argument('--num_envs', '-env', type=int, help='The number of environments to test with', default=1)
    parser.add_argument('--num_epis', '-epi', type=int, help='The number of episodes to run for', default=100)
    parser.add_argument('--gamma', '-g', type=float, help='Gamma value for testing', default=.99)

    args = parser.parse_args()

    load_params(name=args.params)
    print('Initializing Network Evaluator...')
    # Default params
    from deepagent.experiments.params import params
    params.EnvironmentParams.num_envs = args.num_envs
    params.UnityParams.unity_random_reset_max = 0
    params.TestingParams.num_episodes = args.num_epis
    params.TestingParams.gamma = args.gamma
    params.TrainingParams.gamma = args.gamma

    for i in range(args.num_runs):
        print('=' * 40)
        print('running the {} test'.format(i + 1))
        run_experiment(experiment_type=ExperimentType.test, prev_weights=None, weights=args.weights, test_suffix=str(
            i + 1), tensorboard_logging=False)

    print('=' * 40)
    print('Results:')
    print('-' * 40)
    # from deepagent.experiments.params import params
    regex = re.compile(r'.*Avg Discnt Rwds: (-?\d+\.\d+).*')
    num_loss_regex = re.compile(r'/\[-//gn')
    avg_discnt_rewards = []
    num_losses = []
    for i in range(args.num_runs):
        # with open(os.path.join('deepagent/experiments/params/scripted_deep_scenario/', 'testing_log' + str(i + 1) + '.txt'), 'r') as f:
        with open(os.path.join(params.ModuleParams.weights_dir, 'testing_log' + str(i + 1) + '.txt'), 'r') as f:
            lines = f.readlines()
            match = regex.match(lines[-1])
            avg_discnt_rewards.append(float(match.group(1)))
        with open(os.path.join(params.ModuleParams.weights_dir, 'testing_log' + str(i + 1) + '.txt'), 'r') as f:
            # count the number of games with negative score
            num_losses.append(f.read().count('[-'))

    avg_avg_rewards = sum(avg_discnt_rewards) / args.num_runs
    max_avg_reward = max(avg_discnt_rewards)
    min_avg_reward = min(avg_discnt_rewards)
    std_dev_rewards = np.std(avg_discnt_rewards)
    win_rate = 1.0 - (float(sum(num_losses)) / float(args.num_runs * args.num_epis))
    std_win_rate = np.std(num_losses) / 100.0
    print(
        'params={} weights={} min_avg_discnt_rwd={} max_avg_discnt_rwd={} avg_avg_discnt_rwd={} std_dev_discnt_rwds={} num_test_runs={} '
        'episodes_per_run={} win_rate={}, std_win_rate={}'.format(
            args.params, args.weights, min_avg_reward, max_avg_reward, avg_avg_rewards, std_dev_rewards, args.num_runs,
            params.TestingParams.num_episodes, win_rate, std_win_rate
        ))
    print('Note: win rate calculation assumes losses are <0 score games and wins are >=0 score games')


if __name__ == '__main__':
    main()
