import math

from tensorflow.keras.layers import PReLU, Conv2D, GlobalAveragePooling2D

from deepagent.agents.factories import RandomAgentFactory, KeyboardAgentFactory, PPOAgentFactory
from deepagent.envs.env_constants import EnvType
from deepagent.experiments.params.params import EnvironmentParams, TrainingParams, TestingParams, \
    AgentParams, TrainingFunctions, NetworkInit, MapGenConfig, Racer

EnvironmentParams.set(
    environment='test_logans_run_004_melee',
    env_type=EnvType.racer,
    num_envs=2,
    env_render=True
)


def curiosity_schedule(step_count):
    return 0.01 * max(0.0, 1.0 - (step_count/float(TrainingParams.steps)))


def lr_schedule(step_count):
    return math.sqrt(TrainingParams.batch_size/256.0) * 1.0e-4 * max(0.001, 1.0 - (step_count / float(TrainingParams.steps)))


def clip_schedule(step_count):
    return 0.1 * max(0.0001, 1.0 - (step_count / TrainingParams.steps))


TrainingFunctions.set(
    curiosity_schedule=curiosity_schedule,
    lr_schedule=lr_schedule,
    clip_schedule=clip_schedule
)

TrainingParams.set(
    steps=2000000,
    gamma=0.99,
    update_interval=128,
    num_updates=4,
    batch_size=5000,
    batches_per_update=8,
    episodic_reward=True,
)

TestingParams.set(
    num_episodes=100,
    gamma=.99
)

AgentParams.set(
    agent_factory=PPOAgentFactory(
        custom_objects={}
    )
)

Racer.set(
    frame_stack=1,
    map_gen_config=MapGenConfig(y=64, x=64, episodes_per_map=1000),
    include_a_priori_cost=False,
    nadded_walls=5,
    ndeleted_walls=10,
    max_delta=0,
    nspeeds=1,
    randomize_maps=True,
    patch_octaves=6,
    map_octaves=4,
    wall_octaves=9,
    include_vector_state=False,
    alternate_confidence=True,
    real_a_priori_map=False
)


def create_cnn():
    return [[Conv2D(32, kernel_size=9, strides=4, padding='same', kernel_initializer='he_normal'),
             Conv2D(32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal'),
             Conv2D(32, kernel_size=5, strides=2, padding='same', kernel_initializer='he_normal'),
             Conv2D(32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')]]


NetworkInit.set(
        cnns=create_cnn,
        base=[],
        shared=[512, 512],
        policy=[128, 128],
        q_value=[128, 128],
        #activation_type can be PReLU, ReLU, or a string activation such as 'tanh'
        activation_type=PReLU,
        kernel_initializer='he_normal',
        kernel_regularizer=None
)

# INTENDED FOR DEVELOPER USE
# set any/all params that are assigned a path to this file, so they wont be committed to the repo and cause merge issues
# comment/remove this import for production
from . import dev_experiment_param_override
