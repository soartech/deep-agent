from deepagent.agents.factories import NatureCNNA2CAgentFactory
from deepagent.experiments.params.params import EnvironmentParams, TrainingFunctions, TrainingParams, TestingParams, \
    AgentParams
from deepagent.envs.env_constants import EnvType

EnvironmentParams.set(
    environment='PongNoFrameskip-v4',
    env_type=EnvType.atari,
    num_envs=16,
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
    update_interval=5,
    num_updates=1,
    batch_size=80,
    batches_per_update=1,
    episodic_reward=True
)

TestingParams.set(
    num_episodes=600,
    gamma=.99
)

AgentParams.set(
    agent_factory=NatureCNNA2CAgentFactory(custom_objects={})
)

# INTENDED FOR DEVELOPER USE
# set any/all params that are assigned a path to this file, so they wont be committed to the repo and cause merge issues
# comment/remove this import for production
from . import dev_experiment_param_override
