from abc import ABC, abstractmethod

from deepagent.envs.data import DeepAgentSpaces, UnitActions, EnvReturn
from typing import Iterator, Type

from deepagent.envs.reward_masks import RewardTypes


class AgentStep:
    def __init__(self, id, unit_type, state, reward, terminal, mask):
        self.id = id
        self.unit_type = unit_type
        self.state = state
        self.reward = reward
        self.terminal = terminal
        self.mask = mask


class EnvTerminal:
    def __init__(self, id, data):
        self.id = id
        if not data:
            self.terminal = False
            self.ranking = None
        else:
            self.terminal = True
            self.ranking = data


class AbstractStepData(ABC):

    @abstractmethod
    def steps(self) -> Iterator[AgentStep]:
        pass

    @abstractmethod
    def env_terminals(self) -> Iterator[EnvTerminal]:
        pass


class DictionaryStepData(AbstractStepData):
    def __init__(self, states, rewards, terminals, masks, env_dones):
        self.states = states
        self.rewards = rewards
        self.terminals = terminals
        self.masks = masks
        self.env_dones = env_dones

    def steps(self):
        for key, state in self.states.items():
            id, unit_type = key
            yield AgentStep(id, unit_type, state, self.rewards[id], self.terminals[id], self.masks[id])

    def env_terminals(self):
        for key, done in self.env_dones.items():
            yield EnvTerminal(key, done)


class RewardsMixin:
    @property
    def reward_types(self) -> Type[RewardTypes]:
        print(f'WARN: {self.__class__.__name__} does not implement RewardTypes, using default emtpy RewardTypes.')
        return RewardTypes

    def set_reward_masks(self, masks):
        print(f'WARN: {self.__class__.__name__} does not implement set_reward_masks, statement has no effect.')


class AbstractDeepAgentEnv(RewardsMixin, ABC):
    '''
    New environments (StarCraft, Unity, Python, etc.) must extend this class.
    '''

    @abstractmethod
    def step(self, actions: UnitActions) -> AbstractStepData:
        pass

    @abstractmethod
    def reset(self) -> AbstractStepData:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> DeepAgentSpaces:
        pass

    @property
    @abstractmethod
    def action_space(self) -> DeepAgentSpaces:
        pass

    @property
    @abstractmethod
    def use_terminals_as_start_states(self) -> bool:
        pass

    @abstractmethod
    def close(self):
        pass


class AbstractDeepAgentDictionaryEnv(RewardsMixin, ABC):
    '''
    Deprecated

    New environments should use AbstractDeepAgentEnv
    '''

    @abstractmethod
    def step(self, actions: UnitActions) -> EnvReturn:
        pass

    @abstractmethod
    def reset(self) -> EnvReturn:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> DeepAgentSpaces:
        pass

    @property
    @abstractmethod
    def action_space(self) -> DeepAgentSpaces:
        pass

    @property
    @abstractmethod
    def use_terminals_as_start_states(self) -> bool:
        pass

    @abstractmethod
    def close(self):
        pass


class LegacyDeepAgentEnv(AbstractDeepAgentEnv):
    def __init__(self, dictionary_env: AbstractDeepAgentDictionaryEnv):
        self.dictionary_env = dictionary_env

    def step(self, actions: UnitActions):
        states, rewards, terminals, masks, env_dones = self.dictionary_env.step(actions)
        return DictionaryStepData(states, rewards, terminals, masks, env_dones)

    def reset(self):
        states, rewards, terminals, masks, env_dones = self.dictionary_env.reset()
        return DictionaryStepData(states, rewards, terminals, masks, env_dones)

    @property
    def observation_space(self) -> DeepAgentSpaces:
        return self.dictionary_env.observation_space

    @property
    def action_space(self) -> DeepAgentSpaces:
        return self.dictionary_env.action_space

    @property
    def use_terminals_as_start_states(self) -> bool:
        return self.dictionary_env.use_terminals_as_start_states

    def close(self):
        return self.dictionary_env.close()

    def render(self):
        self.dictionary_env.render()

    def set_reward_masks(self, masks):
        self.dictionary_env.set_reward_masks(masks)

    @property
    def reward_types(self) -> Type[RewardTypes]:
        return self.dictionary_env.reward_types
