# NOTE: Do not use this module directly. This is just a placeholder.
# Whenever running an experiment, use deepagent.params.utils.load_params to load in the actual params for the experiment.
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Callable, List, NamedTuple, Dict, Tuple, Union, Type

from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig
from pysc2.env.sc2_env import Race
from tensorflow.keras.layers import PReLU, Conv2D
from tensorflow.python.keras.layers.convolutional import Conv

from deepagent.envs.env_constants import EnvType
from deepagent.recording.recording import PlayerType

import trueskill
import sys
import inspect

thismodule = sys.modules[__name__]


# from abc import abstractmethod, ABC

def param_override(cls_kwargs):
    for clsstr in cls_kwargs:
        if hasattr(thismodule, clsstr):
            param_type: IParam = getattr(thismodule, clsstr)
            param_type.set(**cls_kwargs[clsstr])
        else:
            raise ValueError('Class "{}" not found in params'.format(clsstr))
    pass


def get_param_state():
    return UsedParams.__str__()


class IParam:
    @classmethod
    def set(cls, allow_add_attr=False, **kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k) or allow_add_attr:
                setattr(cls, k, v)
            else:
                raise ValueError('Parameter "{}" not found in class.{}'.format(k, cls.__name__))
        UsedParams.add(cls.__name__)

    @classmethod
    def __str__(cls):
        attr_str = ""
        for attr in dir(cls):
            if attr.startswith("_"):
                continue
            attr_str += f"\t{attr}: {getattr(cls, attr)}\n"
        return "{\n" + attr_str + "}"


class UsedParams:
    all_param_list: List = []

    @classmethod
    def add(cls, param_name: str):
        cls.all_param_list.append(param_name)

    @classmethod
    def __str__(cls):
        state_str = ""
        for name in cls.all_param_list:
            obj = getattr(thismodule, name)
            if issubclass(obj, IParam):
                state_str += f"{name}\n{obj.__str__()}\n\n"
        return state_str


class ExperimentType(Enum):
    pre_train = 'pretrain'
    train = 'train'
    test = 'test'
    population_test = 'ptest'
    record = 'record'

    def __str__(self):
        return self.value


class MaskType(Enum):
    sim = 'sim'
    env_wrapper = 'env_wrapper'

    def __str__(self):
        return self.value


class RemoteArenaParams(IParam):
    arena_servers: List = [('192.168.0.98', 16, 1), ('192.168.0.103', 16, 1), ('192.168.0.106', 16, 1)]
    manager_server: str = '192.168.132'
    port: int = 50000


class RemoteDeploymentParams(IParam):
    # expected list of strings, format|address|port
    # tcp://192.168.86.57:6060
    server_list: List = None


class FSMParams(IParam):
    fsm_action_path: str = None
    fsm_event_path: str = None
    fsm_transprob_path: str = None
    fsm_def_path: str = None


class EnvironmentParams(IParam):
    environment: str = None
    env_type: EnvType = EnvType.atari
    num_envs: int = 1
    env_render: bool = False  # Turn this on to call Env.render in the train or test loop
    terminal_as_start: bool = False
    mask_type: MaskType = MaskType.sim
    feature_extractor: Callable = None
    norm_funct: Callable = None
    static_sim_agentIds: bool = False
    reward_types: None


class AbstractUnityStatsHandler(ABC):
    @abstractmethod
    def handle_stats(self, stats: Dict[int, Dict]):
        """
        Decide what to do with stats returned from unity environments.
        :param stats: A dictionary of stats: key=env_id, val=env_stats_dict
        """
        pass


class TrainingParams(IParam):
    steps: int = 25000000
    gamma: float = 0.99
    update_interval: int = 128
    num_updates: int = 4
    batch_size: int = 256
    batches_per_update: int = 4
    episodic_reward: bool = True
    lam: float = 0.95


class TrainingFunctions(IParam):
    @staticmethod
    def curiosity_schedule(step_count: int) -> float:
        return 0.01

    @staticmethod
    def clip_schedule(step_count: int) -> float:
        return 0.1 * max(0.0001, 1.0 - (step_count / float(TrainingParams.steps)))

    @staticmethod
    def lr_schedule(step_count: int) -> float:
        return 2.5e-5 * max(0.0, 1.0 - (step_count / float(TrainingParams.steps)))

    curiosity_schedule: Callable[[int], float] = curiosity_schedule
    clip_schedule: Callable[[int], float] = clip_schedule
    lr_schedule: Callable[[int], float] = lr_schedule


class TestingParams(IParam):
    num_episodes: int = 100
    gamma: float = .99
    do_state_logging: bool = False
    state_logging_episode_count: int = 4


class AgentParams(IParam):
    agent_factory = None
    agent_kwargs = None


class ModuleParams:
    """
    These values will be automatically set by the experiment runner.
    """
    params_dir: str = None  # The parent directory of the params module passed in to the params loader/
    weights_dir: str = None  # The weights directory to use for training or testing.
    weights_num: str = None  # The step number at which the weights were saved (or None to use the final weights)
    prev_weights = None  # This value will optionally be added by the experiment runner.
    name: str = None  # The name of the param module used.
    tensorboard_dir: str = None  # The tensorboard directory for the experiment.


class RuntimeParams:
    """
    These values will be automatically set via user arguments and the experiment runner.
    """
    experiment_type: ExperimentType = None

    @classmethod
    def is_recording(cls):
        return cls.experiment_type == ExperimentType.record and (
                ImitationLearning.num_recording_pairs() > 0 or ImitationLearning.atari_player_id is not None)

    @classmethod
    def is_testing(cls):
        return cls.experiment_type == ExperimentType.test

    @classmethod
    def is_population_testing(cls):
        return cls.experiment_type == ExperimentType.population_test

    @classmethod
    def is_training(cls):
        return cls.experiment_type == ExperimentType.train

    @classmethod
    def is_pre_training(cls):
        return cls.experiment_type == ExperimentType.pre_train and ImitationLearning.num_pre_training_brains() > 0


class NetworkInit(IParam):
    @staticmethod
    def cnns() -> Union[List[List[Conv]], Dict[str, List[Conv]]]:
        '''
        @return: Returns a list of cnn architectures or a dictionary of cnn architectures. This will be need to be
            overridden in your params file if the agent(s) has more than one image input, or if you want a different
            architecture, or if different agents have different numbers/types of images. If your different agents
            have different numbers of images and/or cnn architectures, then you'll need to return a dictionary,
            mapping the agent name to the cnn architecture.
        '''
        return [
            [
                Conv2D(32, kernel_size=8, strides=4, padding='same', kernel_initializer='glorot_normal'),
                Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_initializer='glorot_normal'),
                Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_normal')
            ]
        ]

    base: List = []
    shared: List = []
    policy: List = []
    q_value: List = []
    activation_type = PReLU
    kernel_initializer = 'glorot_normal'
    kernel_regularizer = None

    @classmethod
    def get_cnn_list(cls, agent_name) -> List[List[Conv2D]]:
        cnns = cls.cnns()
        if type(cnns) == dict:
            print(f'cnn keys={cnns.keys()}')
            return cnns[agent_name]
        return cnns


class ImitationLearning(IParam):
    class LearningPair(NamedTuple):
        # The type of the teacher to record, e.x. "MarinePlayerBrain"
        teacher_brain: str

        # The type of the student that this teacher is supposed to provide demonstrations, e.x. "MarineLearningBrain.
        # The only thing that is actually necessary is that the observation and action space align.
        student_brain: str

        teacher_player_type: PlayerType = PlayerType.human

        def as_list(self):
            return [self.teacher_brain, self.student_brain]

    class PreTrainArgs(NamedTuple):
        brain: str
        player_runs: Dict[str, List[int]]  # Dict mapping player_id to run_ids
        recording_files: List[str] = None  # List of file names containing recorded state and action pairs
        state_shape_override: Tuple[
            int, int, int] = None  # If the state shape was not recorded, supply it here, *Don't include frame stacking*
        action_shape_override: Tuple[int] = None  # If the action shape was not recorded, supply it here

    atari_player_id: str = None  # This is the only imitation learning param you have to set for atari recordings
    recording_pairs: Dict[str, LearningPair] = defaultdict(LearningPair)
    pre_training_args: Dict[str, PreTrainArgs] = defaultdict(PreTrainArgs)

    @classmethod
    def set(cls, allow_add_attr=False, **kwargs):
        super().set(allow_add_attr, **kwargs)
        cls.recording_pairs = {lp.teacher_brain: lp for lp in
                               cls.recording_pairs} if cls.recording_pairs is not None else []
        cls.pre_training_args = {ptb.brain: ptb for ptb in
                                 cls.pre_training_args} if cls.pre_training_args is not None else []

    @classmethod
    def recording_targets(cls):
        return set(cls.recording_pairs.keys())

    @classmethod
    def num_recording_pairs(cls):
        return len(cls.recording_pairs)

    @classmethod
    def num_pre_training_brains(cls):
        return len(cls.pre_training_args)


class Racer(IParam):
    frame_stack: int = 4
    map_gen_config: MapGenConfig = None
    include_a_priori_cost: bool = False
    nadded_walls: int = 5
    ndeleted_walls: int = 5
    max_delta: int = 12
    nspeeds: int = 5
    randomize_maps: bool = True
    patch_octaves: int = 17
    map_octaves: int = 4
    wall_octaves: int = 20
    include_vector_state: bool = True
    alternate_confidence: bool = False
    real_a_priori_map: bool = False

    @classmethod
    def should_reload_map(cls, episodes):
        mgc = cls.map_gen_config
        return mgc and mgc.episodes_per_map > 0 and episodes % mgc.episodes_per_map == 0


class MapGenConfig:
    def __init__(self, x: int, y: int, episodes_per_map: int, real_map_list: List[str] = None,
                 real_map_probability: float = 0.0):
        self.x = x
        self.y = y
        self.episodes_per_map = episodes_per_map
        self.real_map_list = real_map_list if real_map_list is not None else []
        self.real_map_probability = real_map_probability
        self._map_index = 0

    def should_use_real_map(self) -> bool:
        return random.random() < self.real_map_probability

    def get_real_map_name(self) -> str:
        map_name = self.real_map_list[self._map_index]
        self._map_index += 1
        self._map_index %= len(self.real_map_list)
        return map_name


class CustomParams(IParam):
    @classmethod
    def set(cls, allow_add_attr=False, **kwargs):
        super().set(allow_add_attr=allow_add_attr, **kwargs)
