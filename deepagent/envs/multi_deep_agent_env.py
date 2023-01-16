from typing import Type

from deepagent.envs.data import UnitActions, DeepAgentSpaces, EnvReturn
from deepagent.envs.deepagent_env import AbstractDeepAgentEnv, AbstractStepData, AbstractDeepAgentDictionaryEnv
from deepagent.envs.env_constants import EnvType
from deepagent.envs.reward_masks import RewardTypes
from deepagent.experiments.params import params
from deepagent.experiments.params.utils import load_params

import multiprocessing as mp


class MultiDeepAgentPythonEnv(AbstractDeepAgentDictionaryEnv):
    def __init__(self, single_env_class, reward_types_enum=None, num_envs=1, starting_random_seed=0):
        self.env_type = single_env_class.__name__
        self.reward_types_enum = reward_types_enum

        self.starting_random_seed = starting_random_seed
        self.num_envs = num_envs
        self.envs = []
        if self.num_envs == 1:
            self.envs.append(PythonSingleProcess(params.ModuleParams.name, single_env_class, self.starting_random_seed))
        else:
            for i in range(self.num_envs):
                self.envs.append(
                    PythonEnvProcess(params.ModuleParams.name, single_env_class, self.starting_random_seed + i))

    def set_reward_masks(self, masks):
        for env in self.envs:
            env.set_reward_masks(masks)

    @property
    def reward_types(self) -> Type[RewardTypes]:
        return self.reward_types_enum

    @property
    def action_space(self) -> DeepAgentSpaces:
        return self.envs[0].action_space

    @property
    def observation_space(self) -> DeepAgentSpaces:
        return self.envs[0].observation_space

    @property
    def use_terminals_as_start_states(self) -> bool:
        return False

    @property
    def obs_value_names(self):
        return self.envs[0].obs_value_names

    def reset(self) -> EnvReturn:
        states = dict()
        rewards = dict()
        terminals = dict()
        masks = dict()
        dones = dict()

        for i, env in enumerate(self.envs):
            env.start_reset()

        for i, env in enumerate(self.envs):
            single_env_states, single_env_rewards, single_env_terminals, single_env_masks, done = env.finish_reset()
            dones[i] = done

            self._update_multi_env_states(single_env_states, i, states)
            self._update_multi_env_dict(single_env_rewards, i, rewards)
            self._update_multi_env_dict(single_env_terminals, i, terminals)
            self._update_multi_env_dict(single_env_masks, i, masks)

        return states, rewards, terminals, masks, dones

    def step(self, actions: UnitActions) -> EnvReturn:
        states = dict()
        rewards = dict()
        terminals = dict()
        masks = dict()
        dones = dict()

        list_actions = []
        for i in range(self.num_envs):
            list_actions.append(dict())

        for ((env_id, unit_id), unit_type), actions in actions.items():
            list_actions[env_id][(unit_id, unit_type)] = actions

        for i, env in enumerate(self.envs):
            env.start_step(list_actions[i])

        for i, env in enumerate(self.envs):
            single_env_states, single_env_rewards, single_env_terminals, single_env_masks, done = env.finish_step()
            dones[i] = done

            self._update_multi_env_states(single_env_states, i, states)
            self._update_multi_env_dict(single_env_rewards, i, rewards)
            self._update_multi_env_dict(single_env_terminals, i, terminals)
            self._update_multi_env_dict(single_env_masks, i, masks)

        self.log_tensorboard_vars_per_step()
        return states, rewards, terminals, masks, dones

    def log_tensorboard_vars_per_step(self):
        pass

    def _update_multi_env_states(self, single_env_states, env_id, multi_env_states):
        for k, v in dict(single_env_states).items():
            new_key = ((env_id, k[0]), k[1])
            multi_env_states[new_key] = v

    def _update_multi_env_dict(self, single_env_dict, env_id, multi_env_dict):
        for k, v in dict(single_env_dict).items():
            new_key = (env_id, k)
            multi_env_dict[new_key] = v

    def close(self):
        for env in self.envs:
            env.close()

    def render(self):
        for env in self.envs:
            env.render()


STEP = 'step'
RESET = 'reset'
CLOSE = 'close'
OBSERVATION_SPACE = 'observation_space'
ACTION_SPACE = 'action_space'
RENDER = 'render'
REWARD_MASKS = 'reward_masks'


class PythonEnvProcess():
    def __init__(self, params_name, env_class, random_seed):
        ctx = mp.get_context('spawn')
        self.pipe, child_pipe = ctx.Pipe()
        self.process = ctx.Process(target=PythonEnvProcess.main,
                                   args=[params_name, child_pipe, env_class, random_seed, params.RuntimeParams.experiment_type])
        self.process.start()

    def start_reset(self):
        self.pipe.send([RESET])

    def finish_reset(self):
        return self.pipe.recv()

    def start_step(self, actions):
        self.pipe.send([STEP, actions])

    def finish_step(self):
        return self.pipe.recv()

    def close(self):
        self.pipe.send([CLOSE])
        self.process.join()

    def set_reward_masks(self, masks: 'RewardMasks'):
        self.pipe.send([REWARD_MASKS, masks])

    def render(self):
        self.pipe.send([RENDER])

    @property
    def observation_space(self):
        self.pipe.send([OBSERVATION_SPACE])
        return self.pipe.recv()

    @property
    def action_space(self):
        self.pipe.send([ACTION_SPACE])
        return self.pipe.recv()

    @staticmethod
    def main(params_name, pipe, env_class, random_seed, experiment_type):
        load_params(params_name)
        params.RuntimeParams.experiment_type = experiment_type
        env = env_class(random_seed=random_seed)

        while True:
            l = pipe.recv()
            cmd = l[0]

            if cmd == STEP:
                actions = l[1]
                states = env.step(actions)
                pipe.send(states)
            elif cmd == RESET:
                states = env.reset()
                pipe.send(states)
            elif cmd == CLOSE:
                env.close()
                break
            elif cmd == OBSERVATION_SPACE:
                pipe.send(env.observation_space)
            elif cmd == ACTION_SPACE:
                pipe.send(env.action_space)
            elif cmd == RENDER:
                env.render()
            elif cmd == REWARD_MASKS:
                env.set_reward_masks(l[1])


class PythonSingleProcess():
    def __init__(self, params_name, env_class, random_seed):
        self.env = env_class(random_seed=random_seed)
        self.step_data = None

    def start_reset(self):
        pass

    def finish_reset(self):
        return self.env.reset()

    def start_step(self, actions):
        self.step_data = self.env.step(actions)

    def finish_step(self):
        return self.step_data

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()

    def set_reward_masks(self, masks: 'RewardMasks'):
        self.env.set_reward_masks(masks)

    @property
    def obs_value_names(self):
        return self.env.obs_value_names

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space