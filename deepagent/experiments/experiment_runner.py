import matplotlib
matplotlib.use('Agg')

import argparse
import glob
import os
import shutil
import sys
import json
from tqdm import tqdm
import tensorflow as tf

import numpy as np
from gym import spaces

from deepagent.agents.common import AbstractAgentTrainer
from deepagent.agents.random_agent import DebugAgentTester, RacerDijsktraTester
from deepagent.common.auto_dir import get_numbered_directories, get_next_numbered_directory
from deepagent.envs.environment_wrappers import TrainingEnvironment
from deepagent.envs.factory import EnvironmentFactory
from deepagent.envs.spaces import DeepAgentSpace
from deepagent.envs.env_constants import EnvType
from deepagent.experiments.params import params
from deepagent.experiments.params.params import ExperimentType, ImitationLearning
from deepagent.experiments.params.utils import load_params
from deepagent.recording.recording import record_atari, MultiPlayerDemoReader
from deepagent.agents.memory import BCMemory

import logging
logging.basicConfig(level=logging.INFO)
from deepagent.experiments.params import utils as param_utils


def setup_environment(env_file_str, env_type, params_name, random_seed=0, arena_nbr=0):
    seed = 123 + random_seed
    np.random.seed(seed)

    env1 = EnvironmentFactory.get_env(env_file_str, env_type, seed, params_name, random_seed, arena_nbr)
    env_wrapper = TrainingEnvironment(env1)

    return env_wrapper


def setup_multi_agent(env_wrapper):
    agent_factory = params.AgentParams.agent_factory
    multi_agent_training_class = agent_factory.multi_agent_training_class()
    return multi_agent_training_class(environment_wrapper=env_wrapper, agent_factory=agent_factory)

import os
from typing import List, Tuple
import numpy as np

INITIAL_POS = 33
SUPPORTED_DEMONSTRATION_VERSIONS = frozenset([0, 1])


def get_demo_files(path: str) -> List[str]:
    """
    Retrieves the demonstration file(s) from a path.
    :param path: Path of demonstration file or directory.
    :return: List of demonstration files

    Raises errors if |path| is invalid.
    """
    if os.path.isfile(path):
        if not path.endswith(".demo"):
            raise ValueError("The path provided is not a '.demo' file.")
        return [path]
    elif os.path.isdir(path):
        paths = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith(".demo")
        ]
        if not paths:
            raise ValueError("There are no '.demo' files in the provided directory.")
        return paths
    else:
        raise FileNotFoundError(
            f"The demonstration file or directory {path} does not exist."
        )


def get_state_shape(file_path: str) -> Tuple[int]:
    state = None
    with open(file_path) as f:
        for line_idx, line in enumerate(tqdm(f)):
            if 'Prev Obs: [ ' not in line:
                continue
            items = line.strip().split(' | ')
            state = items[2][12:-1]
            state = np.fromstring(state, dtype=np.float32, sep=' ')
            break
    if state is None:
        raise Exception('Could not find state shape from recording file')
    return state.shape


def create_pretraining_generator(recording_files: List[str], action_shape: Tuple[int]):
    def load_data_generator():
        num_episodes = 0
        for recording_file in recording_files:
            with open(recording_file) as f:
                episode_idx = 0
                for line_idx, line in enumerate(f):
                    if 'Prev Obs: [ ' not in line:
                        continue
                    items = line.strip().split(' | ')
                    state = items[2][12:-1]
                    state = np.fromstring(state, dtype=np.float32, sep=' ')
                    action = items[1].split()[-1]
                    action = np.eye(action_shape[0])[int(action)]
                    episode_idx = int(line.strip().split('UnitId: (0, ')[1].split(')')[0])
                    yield state, action
                num_episodes += episode_idx
        print(f'Loaded {num_episodes} episodes from recording files.')
    return load_data_generator


def get_pretraining_length(recording_files):
    steps = 0
    for recording_file in recording_files:
        with open(recording_file) as f:
            for line_idx, line in enumerate(tqdm(f)):
                if 'Prev Obs: [ ' in line:
                    steps += 1
    return steps


def pre_train():
    for brain_name, pre_train_args in params.ImitationLearning.pre_training_args.items():
        # init policies
        if pre_train_args.recording_files is not None:
            # Loads pretraining data from recording files
            if pre_train_args.action_shape_override is None:
                raise Exception(
                    'Action shape override was found to be None. Action shape override must be supplied in '
                    'ImitationLearning.PreTrainArgs if recording files are used.'
                )
            action_shape = pre_train_args.action_shape_override
            state_shape = get_state_shape(pre_train_args.recording_files[0])

            # TF dataset is used so training can be done even if the full data doesn't fit in memory
            print(f'Loading pretraining data from recording files: {", ".join(pre_train_args.recording_files)}')
            dataset = tf.data.Dataset.from_generator(create_pretraining_generator(pre_train_args.recording_files, action_shape),
                                                     output_signature=(
                                                      tf.TensorSpec(shape=state_shape, dtype=tf.float32),
                                                      tf.TensorSpec(shape=action_shape, dtype=tf.float32,))
                                                     )
            dataset_len = get_pretraining_length(pre_train_args.recording_files)
            print(f'{dataset_len} examples found in pretraining data')
            dataset = dataset.apply(tf.data.experimental.assert_cardinality(dataset_len))
            vec_space = spaces.Box(-1.0, 1.0, state_shape)
            state_space = DeepAgentSpace(vector_space=vec_space, agent_name=brain_name)
            vector_space = spaces.Box(0.0, 1.0, shape=action_shape, dtype=np.float32)
            action_space = DeepAgentSpace(vector_space=vector_space, agent_name=brain_name)
        else:
            reader = MultiPlayerDemoReader(pre_train_args.player_runs)
            meta = reader.load_meta()
            state_shape = meta.state_shape
            action_shape = meta.action_shape
            brain_name = meta.student_brain

            print(state_shape)
            print(action_shape)

            if state_shape is None:
                state_shape = pre_train_args.state_shape_override

            if action_shape is None:
                action_shape = pre_train_args.action_shape_override

            if params.EnvironmentParams.env_type.is_atari():
                frame_stack = 4
            elif params.EnvironmentParams.env_type.is_unity_ml():
                frame_stack = params.UnityParams.unity_frame_stack
            else:
                raise ValueError(f"Don't know how to pre_train for env_type: {params.EnvironmentParams.env_type}")

            num_actions = action_shape[0]

            state_shape = (state_shape[0], state_shape[1], state_shape[2] * 4)

            # TODO: I don't think the low and high actually matter here, but space management needs to be cleaned up and made consistent in general
            state_space = DeepAgentSpace(image_spaces=spaces.Box(low=-1.0, high=1.0, shape=state_shape))

            if params.EnvironmentParams.env_type.is_atari():
                action_space = DeepAgentSpace(vector_space=spaces.Discrete(n=action_shape[0]))
            elif params.EnvironmentParams.env_type.is_unity_ml():
                action_space = DeepAgentSpace(vector_space=spaces.Box(low=-1.0, high=1.0, shape=action_shape))
            else:
                raise ValueError(f"Don't know how to pre_train for env_type: {params.EnvironmentParams.env_type}")

            state_shape = [x.shape for x in state_space.image_spaces]

        agent_factory = params.AgentParams.agent_factory

        policy = agent_factory.create_agent(agent_type=brain_name, state_space=state_space, action_space=action_space)

        policy.print_summary()
        # import tensorflow as tf
        #policy.set_tensorboard_writer(tf.summary.create_file_writer(params.ModuleParams.tensorboard_dir))
        from deepagent.loggers.logging import get_tensorboard_writer
        policy.set_tensorboard_writer(get_tensorboard_writer())
        print('Loading pre-training data ...')
        if pre_train_args.recording_files is not None:
            memory = BCMemory()
            memory.dataset = dataset
        else:
            memory = reader.bc_memory(frame_stack=frame_stack, num_actions=num_actions,
                                      use_terminals_as_start_state=params.EnvironmentParams.env_type.is_atari(),
                                      tile_first=params.EnvironmentParams.env_type.is_unity_ml())

            action_counts, action_distribution = memory.action_distribution()
            print(f'\nAction Counts: {action_counts}, Action Distributions: {action_distribution}')

        # np.set_printoptions(threshold=500, precision=4, linewidth=150)
        # mean, std = memory.state_mean_and_std()
        # print(f'State Index Mean:\n{mean}')
        # print(f'\nState Index Standard Deviations:\n{std}')
        # np.set_printoptions(threshold=1000, precision=8, linewidth=75) #defaults
        # print(mean.shape)
        # print(std.shape)

        policy.pre_train(memory)



def train(multi_agent: AbstractAgentTrainer):
    multi_agent.train(steps=params.TrainingParams.steps, gamma=params.TrainingParams.gamma,
                      lam=params.TrainingParams.lam)
    #multi_agent.write_report_summary()


def test(multi_agent: AbstractAgentTrainer, test_suffix: str = ''):
    multi_agent.test(num_episodes=params.TestingParams.num_episodes, gamma=params.TestingParams.gamma,
                            testing_suffix=test_suffix)
    #multi_agent.write_report_summary()


def population_test(multi_agent: AbstractAgentTrainer):
    return multi_agent.population_test()


def copy_weights(from_dir, to_dir):
    print('Copying previous weights from {} to {}'.format(from_dir, to_dir))
    files = glob.iglob(os.path.join(from_dir, '*.h5'))
    for file in files:
        shutil.copy2(file, to_dir)


def copy_reward_masks(from_dir, to_dir):
    print('Copying previous reward masks from {} to {}'.format(from_dir, to_dir))
    files = glob.iglob(os.path.join(from_dir, '*.pkl'))
    for file in files:
        shutil.copy2(file, to_dir)


def set_weights_dir(experiment_type, prev_weights, weights, weights_num):
    if experiment_type == ExperimentType.train or experiment_type == ExperimentType.pre_train:
        params.ModuleParams.weights_dir = get_next_numbered_directory(params.ModuleParams.params_dir)
        if prev_weights is not None:
            prev_weights = os.path.join(params.ModuleParams.params_dir, prev_weights)
            copy_weights(from_dir=prev_weights, to_dir=params.ModuleParams.weights_dir)
            copy_reward_masks(from_dir=prev_weights, to_dir=params.ModuleParams.weights_dir)
    elif experiment_type == ExperimentType.test or experiment_type == ExperimentType.record or experiment_type == ExperimentType.population_test:
        if params.AgentParams.agent_factory.multi_agent_training_class() == DebugAgentTester or params.AgentParams.agent_factory.multi_agent_training_class() == RacerDijsktraTester:
            weights = params.ModuleParams.params_dir
        elif weights == "latest":
            weights = str(sorted(get_numbered_directories(params.ModuleParams.params_dir))[-1])
        elif weights is None or weights == '' or weights not in os.listdir(params.ModuleParams.params_dir):
            raise ValueError(
                'Weights number must be specified for testing or recording if not using RandomAgentTester for the AgentTrainer.'
                '\n\tTraining directories are created in ascending order and found in subdirectories of the parameters folders.'
                '\n\tAvailable training folders for {} are {}'.format(params.ModuleParams.name, sorted(
                    get_numbered_directories(params.ModuleParams.params_dir))))
        
        params.ModuleParams.weights_dir = os.path.join(params.ModuleParams.params_dir, weights)
        params.ModuleParams.weights_num = weights_num


def set_tensorboard_dir():
    tensorboard_top_dir = os.path.join(params.ModuleParams.params_dir, 'tensorboard')
    if not os.path.exists(tensorboard_top_dir):
        os.makedirs(tensorboard_top_dir)
    params.ModuleParams.tensorboard_dir = get_next_numbered_directory(tensorboard_top_dir)


parser = argparse.ArgumentParser()
parser.add_argument('--params', '-p', help="The params module to use: ex: 'params1', 'params2', etc.")
parser.add_argument('--experiment_type', '-et', type=ExperimentType, choices=list(ExperimentType),
                    help="The type of experiment to run:\n\tperform supervised imitation learning\n\tperform "
                         "reinforcement learning training\n\ttest the trained network", required=True)
parser.add_argument('--weights', '-w', type=str,
                    help='The training weights folder to test with. Training directories are numbered in ascending '
                         'order and found in subdirectories of the parameters folders.')
parser.add_argument('--weights_num', '-wn', type=str,
                    help='# The step number at which the weights were saved (or None to use the final weights)'
                         'For example: 400000')
parser.add_argument('--prev_weights', '-pw', type=str,
                    help='A weights folder to use at the start of training. Training will still occur in the next '
                         'highest numbered subdirectory of the parameters folders, but the weights can be loaded '
                         'from any of the previous numbered directories.')
parser.add_argument('--test_suffix', '-ts', type=str, default='',
                    help='# This string is added to the test log filename (default="")')
parser.add_argument('-u', action='store_true', help='This argument is just here because pycharm is dumb at '
                                                    'executing modules on remote interpreters. The workaround means '
                                                    'that it adds a "-u" field in the wrong place.')
parser.add_argument('--env_args', '-ea', nargs='*')
parser.add_argument('--param_overrides', '--po', '-po', type=str,
                    help='Parameters to override in the params module in JSON format. '
                         'Ex: --param_overrides="{\'UnityParams.unity_port\': 5006, \'TrainingParams.batch_size\': 8}"')


def jsonstr_to_kwarg(jsonStr):
    param_overrides = jsonStr.replace("\'", "\"")
    param_overrides = json.loads(param_overrides)
    param_kwargs = {}

    # parse params into a dict (class name) of kwargs (param_name:param_val)
    for param_key, param_value in param_overrides.items():
        param_class, param_var = param_key.split('.')
        if param_class not in param_kwargs.keys():
            param_kwargs[param_class] = {}  # first time seeing override for this class, create default/empty kwargs
        param_kwargs[param_class][param_var] = param_value

    return param_kwargs


def main(argv=sys.argv[1:]):
    print('CLEARED FOR RELEASE')
    print('''
    
                                ▓▀^^"""  """""▀▒
                                ╬              ╬
                               ╒▓              ▒∩
                              ,▓▄              ╓╬⌐
                                ╬              ╬⌐
                                ╬              ╬
                                ▀▒ÆÆÆÆÆ  ╦╦╦╦Æ▒▀




   #▒▀▀▀▒       ,,                       ╚▀▀▀▓▓▀▀▀              ,
  ▐╬        ,▒▀^ `╙▒╦    @▀▒⌐    ▒▌^^^╙▒▄    ╬Γ    ║╬^^^^^` ╓▒▀^ ^▀▒ j╬     ▒Γ
   "▀▒▒Æ╗   ╬⌐      ▒∩  @▀  ▒µ   ▒▌    ╫▀    ╬Γ    ║╬,,,,, ]╬        j╬,,,,,▒Γ
         ╬∩ ╬∩      ▓∩ @╬╗╗╗╣╬⌐  ▒▌╙╙▀▓      ╬Γ    ║╬      └╬        j╬     ▒Γ
   ▒Æ╗╖╖▒▀   ▀▒╖╓╓#▓` @▀      ▒µ ▒▌   ╙▓╕    ╬Γ    ║▓╖╖╖╖╖⌐ ^▀▄╖╓╓#▀ j╬     ▒Γ


 _____     ______     ______     ______   ______     ______     ______     __   __     ______  
/\  __-.  /\  ___\   /\  ___\   /\  == \ /\  __ \   /\  ___\   /\  ___\   /\ "-.\ \   /\__  _\ 
\ \ \/\ \ \ \  __\   \ \  __\   \ \  _-/ \ \  __ \  \ \ \__ \  \ \  __\   \ \ \-.  \  \/_/\ \/ 
 \ \____-  \ \_____\  \ \_____\  \ \_\    \ \_\ \_\  \ \_____\  \ \_____\  \ \_\\\\"\_\    \ \_\ 
  \/____/   \/_____/   \/_____/   \/_/     \/_/\/_/   \/_____/   \/_____/   \/_/ \/_/     \/_/ 

''')
    args = parser.parse_args(argv)
    print('argv=',argv)
    params_name = args.params
    experiment_type = args.experiment_type
    params.RuntimeParams.experiment_type = experiment_type
    prev_weights = args.prev_weights
    weights = args.weights
    weights_num = args.weights_num
    test_suffix = args.test_suffix
    load_params(name=params_name)
    if args.env_args:
        params.AgentParams.agent_kwargs = dict([pair.split('=') for pair in args.env_args])
    else:
        params.AgentParams.agent_kwargs = {}

    # Override params if they are in the command line args
    if args.param_overrides:
        param_kwargs = jsonstr_to_kwarg(args.param_overrides)
        params.param_override(param_kwargs)

    invalid_keys = ['']
    for k in invalid_keys:
        if k in params.AgentParams.agent_kwargs:
            del params.AgentParams.agent_kwargs[k]
    if len(params.AgentParams.agent_kwargs) == 0:
        params.AgentParams.agent_kwargs = None

    run_experiment(experiment_type, prev_weights, weights, weights_num, params_name, test_suffix, args=args)


def run_experiment(experiment_type, prev_weights, weights, weights_num, params_name, test_suffix='', tensorboard_logging=True, args=None):
    print(f'setting weights->{weights}')
    set_weights_dir(experiment_type=experiment_type, prev_weights=prev_weights, weights=weights, weights_num=weights_num)

    param_utils.write_commandline_params(args.__str__(), "commandline_full")
    if args.param_overrides:
        param_kwargs: dict = jsonstr_to_kwarg(args.param_overrides)
        param_utils.write_commandline_params(param_kwargs.__str__(), "commandline_overrides")
    param_utils.write_params_state()

    if tensorboard_logging:
        set_tensorboard_dir()
    else:
        params.ModuleParams.tensorboard_dir = None

    print('Experiment type:{}'.format(experiment_type))
    print("Loaded params:{}".format(params.ModuleParams.params_dir))
    print('Weights dir set to {}'.format(params.ModuleParams.weights_dir))
    if prev_weights is not None:
        params.ModuleParams.prev_weights = prev_weights
        print('Using prev weights: {}'.format(prev_weights))

    env_type = params.EnvironmentParams.env_type
    runtime = params.RuntimeParams

    # Pre-Train the network from previously recorded data
    if runtime.is_pre_training():
        pre_train()
        return

    # Record an atari environment
    if runtime.is_recording() and env_type.is_atari():
        # This is always the recording setup to use for atari, setting the teacher_brain and student_brain
        # just affects the data in the meta data files for the atari recordings.
        record_atari(player_id=params.ImitationLearning.atari_player_id)
        return

    # Setup environments for training, testing, or unity recording.
    # Population trainers setup and tear-down environments by themselves
    agent_factory = params.AgentParams.agent_factory
    multi_agent_training_class = agent_factory.multi_agent_training_class()
    if multi_agent_training_class.instantiates_envs():
        multi_agent = multi_agent_training_class()
    else:
        env_wrapper = setup_environment(params.EnvironmentParams.environment, params.EnvironmentParams.env_type, params_name)
        multi_agent = setup_multi_agent(env_wrapper)

    # Train atari or unity
    if runtime.is_training():
        train(multi_agent)
    # Test atari or unity
    elif runtime.is_testing():
        test(multi_agent, test_suffix=test_suffix)


if __name__ == '__main__':
    main()
