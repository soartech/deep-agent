from tensorflow.keras.layers import ReLU, Conv2D

from deepagent.agents.factories import PPOAgentFactory
from deepagent.experiments.params.params import EnvironmentParams, TrainingFunctions, TrainingParams, TestingParams, \
    AgentParams, ImitationLearning, NetworkInit
from deepagent.envs.env_constants import EnvType

EnvironmentParams.set(
    environment='PongNoFrameskip-v4',
    env_type=EnvType.atari,
    num_envs=8,
    env_render=False
)


def curiosity_schedule(step_count):
    return 0.01


def lr_schedule(step_count):
    return 2.5e-4 * max(0.0, 1.0 - (step_count / TrainingParams.steps))


def clip_schedule(step_count):
    return 0.1 * max(0.0001, 1.0 - (step_count / TrainingParams.steps))


TrainingFunctions.set(
    curiosity_schedule=curiosity_schedule,
    lr_schedule=lr_schedule,
    clip_schedule=clip_schedule
)

TrainingParams.set(
    steps=1250000,
    gamma=.99,
    update_interval=128,
    batches_per_update=4,
    batch_size=256,
    num_updates=4,
    episodic_reward=True
)

TestingParams.set(
    num_episodes=600,
    gamma=.99
)

AgentParams.set(
    agent_factory=PPOAgentFactory(custom_objects={})
)

def create_cnn():
    return [[Conv2D(32, kernel_size=8, strides=4, padding='same', kernel_initializer='he_normal'),
             Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal'),
             Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')]]

NetworkInit.set(
        cnns=create_cnn,
        base=[],
        shared=[512],
        policy=[],
        q_value=[],
        #activation_type can be PReLU, ReLU, or a string activation such as 'tanh'
        activation_type=ReLU,
        kernel_initializer='he_normal',
        kernel_regularizer=None
)

ImitationLearning.atari_player_id = 'cade.sperlich'
ImitationLearning.set(
    pre_training_args=[ImitationLearning.PreTrainArgs(brain='atari', player_runs={'cade.sperlich': [1]})])

# INTENDED FOR DEVELOPER USE
# set any/all params that are assigned a path to this file, so they wont be committed to the repo and cause merge issues
# comment/remove this import for production
from . import dev_experiment_param_override
