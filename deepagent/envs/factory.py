import multiprocessing
import os
import sys

from deepagent.envs.multi_deep_agent_env import MultiDeepAgentPythonEnv
from deepagent.envs.racer.racer import RacerEnv

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds, retro_wrappers
from baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from baselines.common.wrappers import ClipActionsWrapper
from baselines.run import get_env_type

from deepagent.experiments.params import params
from deepagent.envs.env_constants import EnvType
from deepagent.openai_addons.rnd.rnd_atari_wrappers import make_atari, wrap_deepmind

from functools import partial


class EnvironmentFactory:
    @classmethod
    def get_env(cls, env_file_str, env_type, seed, params_name, random_seed, arena_nbr=0):

        if env_type == EnvType.atari:  # Atari and baseline envs
            import baselines.common.cmd_util

            arg_parser = baselines.common.cmd_util.common_arg_parser()
            args, _ = arg_parser.parse_known_args()
            args.env = env_file_str
            args.seed = seed
            args.num_env = params.EnvironmentParams.num_envs
            return create_atari_env(args)
        elif env_type == EnvType.racer:
            return MultiDeepAgentPythonEnv(RacerEnv, num_envs=params.EnvironmentParams.num_envs, starting_random_seed=random_seed)
        else:
            raise ValueError('Unknown environment type: {}'.format(env_type))


def create_atari_env(args):
    from baselines.common.vec_env.vec_frame_stack import VecFrameStack
    from deepagent.openai_addons.baselines.dictionary_vec_frame_stack_wrapper import \
        DictionaryVecFrameStackWrapper
    env_type, env_id = get_env_type(args)

    frame_stack_size = 4

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu

    env = _make_vec_env(env_id, env_type, nenv, args.seed, gamestate=args.gamestate,
                        reward_scale=args.reward_scale)
    env = VecFrameStack(env, frame_stack_size)
    env = DictionaryVecFrameStackWrapper(env, env_type)
    return env


def _make_vec_env(env_id, env_type, num_env, seed,
                  wrapper_kwargs=None,
                  env_kwargs=None,
                  start_index=0,
                  reward_scale=1.0,
                  flatten_dict_observations=True,
                  gamestate=None,
                  initializer=None,
                  force_dummy=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()

    def make_thunk(rank, initializer=None):
        return lambda: _make_env(
            env_id=env_id,
            env_type=env_type,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            env_kwargs=env_kwargs,
            logger_dir=logger_dir,
            initializer=initializer
        )

    set_global_seeds(seed)
    if not force_dummy and num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index, initializer=initializer) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(i + start_index, initializer=None) for i in range(num_env)])


def _make_env(env_id, env_type, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None,
              flatten_dict_observations=True, wrapper_kwargs=None, env_kwargs=None, logger_dir=None, initializer=None):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    if ':' in env_id:
        import re
        import importlib
        module_name = re.sub(':.*', '', env_id)
        env_id = re.sub('.*:', '', env_id)
        importlib.import_module(module_name)
    if env_type == 'atari':
        env = make_atari(env_id)
    elif env_type == 'retro':
        import retro
        gamestate = gamestate or retro.State.DEFAULT
        env = retro_wrappers.make_retro(game=env_id, max_episode_steps=10000,
                                        use_restricted_actions=retro.Actions.DISCRETE, state=gamestate)
    else:
        env = gym.make(env_id, **env_kwargs)

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        keys = env.observation_space.spaces.keys()
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    if env_type == 'atari':
        env = wrap_deepmind(env, **wrapper_kwargs)
    elif env_type == 'retro':
        if 'frame_stack' not in wrapper_kwargs:
            wrapper_kwargs['frame_stack'] = 1
        env = retro_wrappers.wrap_deepmind_retro(env, **wrapper_kwargs)

    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)

    if reward_scale != 1:
        env = retro_wrappers.RewardScaler(env, reward_scale)

    return env
