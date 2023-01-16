from typing import Tuple, Dict, List

from abc import abstractmethod, ABC

import numpy as np
import copy

from deepagent.envs.deepagent_env import AbstractDeepAgentDictionaryEnv, LegacyDeepAgentEnv


class DecisionRequest:
    def __init__(self, s0, mask, terminal_start_state):
        self.s0 = s0
        self.mask = mask
        self.terminal_start_state = terminal_start_state

    def to_incomplete_step(self, a):
        return IncompleteStep(self.s0, a)


class BatchDecisionRequest:
    def __init__(self, decision_request: List[DecisionRequest], agent_ids: List[int]):
        self._s0s = []
        self._masks = []
        self.agent_ids = []
        self.agent_id_to_batch_id = {}
        first = True

        for dr, aid in zip(decision_request, agent_ids):
            if first:
                for _ in dr.s0:
                    self._s0s.append([])
                first = False
            for i, s0 in enumerate(dr.s0):
                self._s0s[i].append(s0)
            self._masks.append(dr.mask)
            self.agent_ids.append(aid)

        for i, id in enumerate(self.agent_ids):
            self.agent_id_to_batch_id[id] = i

        self.batch_s0 = []
        for s0 in self._s0s:
            self.batch_s0.append(np.array(s0, dtype=np.float32))
        self.batch_mask = np.array(self._masks)

    def batch_id(self, agent_id):
        return self.agent_id_to_batch_id[agent_id]

class IncompleteStep:
    def __init__(self, s0, a):
        self.s0 = s0
        self.a = a

    def to_complete_step(self, r, t, s1):
        return CompleteStep(self.s0, self.a, r, t, s1)


class CompleteStep:
    def __init__(self, s0, a, r, t, s1):
        self.s0 = s0
        self.a = a
        self.r = r
        self.t = t
        self.s1 = s1


class CompletedSteps:
    def __init__(self, completed_steps_dict: Dict[Tuple[Tuple[int, int], str], CompleteStep], env_terminals):
        self._completed_steps_dict = completed_steps_dict
        # (brain_name, batch_idx) to CompleteStep
        self._batch_idx_to_step = {}  # type: Dict[Tuple[str, int], CompleteStep]
        self._batch_to_agent_id = {}  # type: Dict[Tuple[str, int], Tuple[Tuple[int, int], str]]
        self._agent_id_to_batch_id = {} # type: Dict[Tuple[Tuple[int, int], str], int]
        self._s0_batches = {} # type: Dict[str, List[np.array]]
        self._s1_batches = {} # type: Dict[str, List[np.array]]
        self.env_terminals = env_terminals

        s0_batches = {}
        s1_batches = {}
        for key in self._completed_steps_dict.keys():
            agent_id, brain_name = key
            if brain_name not in s0_batches:
                s0_batches[brain_name] = None
                s1_batches[brain_name] = None

        for key, completed_step in self._completed_steps_dict.items():
            agent_id, brain_name = key

            if s0_batches[brain_name] != None:
                batch_idx = len(s0_batches[brain_name][0])
            else:
                batch_idx = 0
            self._batch_idx_to_step[(brain_name, batch_idx)] = completed_step
            self._batch_to_agent_id[(brain_name, batch_idx)] = key
            self._agent_id_to_batch_id[key] = batch_idx
            if s0_batches[brain_name] == None:
                s0_batches[brain_name] = [[] for _ in completed_step.s0]
            if s1_batches[brain_name] == None:
                s1_batches[brain_name] = [[] for _ in completed_step.s1]
            for i, s0 in enumerate(completed_step.s0):
                s0_batches[brain_name][i].append(s0)
            for i, s1 in enumerate(completed_step.s1):
                s1_batches[brain_name][i].append(s1)

        for brain_name in s0_batches.keys():
            self._s0_batches[brain_name] = [np.array(s0, dtype=np.float32) for s0 in s0_batches[brain_name]]
        for brain_name in s1_batches.keys():
            self._s1_batches[brain_name] = [np.array(s1, dtype=np.float32) for s1 in s1_batches[brain_name]]

    @property
    def mini_batches(self) -> Dict[str, Tuple[np.array, np.array]]:
        """
        :returns: A dict mapping brain_name to a tuple (s0_batch, s1_batch).
        WHERE
            s0_batch is a numpy array of all of the s0 states for the brain
            s1_batch is a numpy array of all of the s1 states for the brain
        """
        return {brain_name: (self._s0_batches[brain_name], self._s1_batches[brain_name]) for brain_name in
                self._s0_batches.keys()}

    def completed_step(self, brain_name, batch_index) -> CompleteStep:
        """
        :param brain_name: the name of the brain
        :param batch_index: the batch index as returned by self.mini_batches
        :return: Returns the CompleteStep for the given brain_name and batch_index
        """
        return self._batch_idx_to_step[(brain_name, batch_index)]

    @property
    def completed_steps(self) -> Dict[Tuple[Tuple[int, int], str], CompleteStep]:
        """
        :return: A dict mapping ((env_num, unit_id), brain_name) to CompleteStep
        WHERE
            env_num is the environment number
            unit_id is the id of the unit
            brain_name is the name of the brain
        """
        return self._completed_steps_dict

    def agent_id(self, brain_name, batch_index) -> Tuple[Tuple[int, int], str]:
        """
        :returns: The key for the agent corresponding to the brain_name and batch_index
        """
        return self._batch_to_agent_id[(brain_name, batch_index)]

    def batch_id(self, key) -> int:
        return self._agent_id_to_batch_id[key]

    def __str__(self):
        ret = ''
        for k, v in self._completed_steps_dict.items():
            ret += str(k) + str(v.t)
        return ret


class AbstractTrainingEnvironment(ABC):
    '''
    Interface used by training code to communicate with environments.
    Encapsulates idiosyncrasies around managing and collection state and communicating with environments.

    The methods must be called in the following order:

    environment.reset()
    while your_condition == True:
        state_dict = environment.get_decision_requests()
        actions = your_get_actions_function(state_dict)
        batch_decision_request, completed_steps, episode_rewards = environment.step(actions)
        your_memory_function(completed_steps)
        your_condition_update()

    '''

    def __init__(self, env: AbstractDeepAgentDictionaryEnv, gamma=1.0):
        self.env = env # type: AbstractDeepAgentDictionaryEnv
        self.gamma = gamma # type: float

        if not hasattr(self.env, 'pause'):
            print('WARNING: env does not implement pause, using no op implementation')
            self.env.pause = lambda *args, **kwargs: None
        if not hasattr(self.env, 'resume'):
            print('WARNING: env does not implement resume, using no op implementation')
            self.env.resume = lambda *args, **kwargs: None

    def get_env(self):
        return self.env

    @abstractmethod
    def reset(self) -> Dict[str, BatchDecisionRequest]:
        '''
        :returns: a dictionary mapping brain_type to a BatchDecisionRequest
        '''
        pass

    @property
    def env_type(self):
        return self.env.env_type

    @abstractmethod
    def step(self, action_batches: Dict[str, np.array]) -> Tuple[
        Dict[str, BatchDecisionRequest], CompletedSteps, List[float]]:
        '''
        :param: action_batches: A dictionary mapping brain_type to action_array.
            WHERE
            action_array is an np.array of actions, the order of each action_array needs to be the
                same as that last BatchDecisionRequest returned from this class.
        :returns: A tuple (batch_decision_requests, completed_steps, completed_episode_discounted_rewards).
            WHERE
            batch_decision_requests is a dictionary mapping brain_name to BatchDecisionRequest
            completed_steps is a CompletedSteps class object
            completed_episode_discounted_rewards is a list of floats representing the discounted rewards for all
                episodes that just concluded
        '''
        pass

    @property
    @abstractmethod
    def action_space(self):
        pass

    @property
    @abstractmethod
    def observation_space(self):
        pass

    def render(self):
        return self.env.render()

    @property
    def use_terminals_as_start_states(self):
        return self.env.use_terminals_as_start_states

    @property
    def team_game(self):
        return self.env.team_game

    def close(self):
        self.env.close()

    def pause(self):
        self.env.pause()

    def resume(self):
        self.env.resume()



class TrainingEnvironment(AbstractTrainingEnvironment):
    def __init__(self, env: AbstractDeepAgentDictionaryEnv):
        super(TrainingEnvironment, self).__init__(LegacyDeepAgentEnv(env))
        self.decision_requests = {}  # type: Dict[Tuple[int, str], DecisionRequest]
        self.incomplete_steps = {}  # type: Dict[Tuple[int, str], IncompleteStep]
        self.completed_steps = {}  # type: Dict[Tuple[int, str], CompleteStep]
        self._last_batch_decision_requests = {} # type: Dict[str, BatchDecisionRequest]
        self._environment_rewards = {} # type: Dict[int, Dict[int, List[float]]]
        self._smoothed_discount_reward = np.nan  # type: float

    def _get_batch_decision_request(self):
        brain_names = set(key[1] for key in self.decision_requests.keys())
        brain_names_to_drs = {bn: [] for bn in brain_names}
        brain_names_to_agents = {bn: [] for bn in brain_names}
        for key, dr in self.decision_requests.items():
            unit_id, brain_name= key
            brain_names_to_drs[brain_name].append(dr)
            brain_names_to_agents[brain_name].append(unit_id)

        batch_decision_requests = {}
        for brain_name, dr in brain_names_to_drs.items():
            batch_decision_requests[brain_name] = BatchDecisionRequest(dr, brain_names_to_agents[brain_name])

        self._last_batch_decision_requests = batch_decision_requests
        return batch_decision_requests

    def _get_complete_steps_and_clear(self):
        completed_steps = self.completed_steps
        self.completed_steps = {}
        return completed_steps

    def reset(self):
        from deepagent.experiments.params.params import EnvironmentParams
        self.decision_requests = {}
        self.incomplete_steps = {}
        self.completed_steps = {}
        self._last_batch_decision_requests = {}
        for env_id in range(len(self._environment_rewards)):
            self._environment_rewards[env_id] = {}
        step_data = self.env.reset()
        for step in step_data.steps():
            if EnvironmentParams.norm_funct is not None:
                step.state = EnvironmentParams.norm_funct(step.state)
            if EnvironmentParams.feature_extractor is not None:
                step.state = EnvironmentParams.feature_extractor(step.state, step.terminal, step.id)
            decision_request = DecisionRequest(step.state, step.mask, step.terminal)
            if self.env.use_terminals_as_start_states or not step.terminal:
                self.decision_requests[(step.id, step.unit_type)] = decision_request
        return self._get_batch_decision_request()

    def step(self, action_batches: Dict[str, np.array]) -> Tuple[Dict[str, BatchDecisionRequest], CompletedSteps, List[float]]:
        for brain_name, ab in action_batches.items():
            if np.isnan(ab).any():
                self.close()
                raise ValueError("Received NaN values in the action batch: brain_name={} action_batch={}".format(brain_name, ab))
        env_actions = {}
        for brain_name, action_batch in action_batches.items():
            agent_ids = self._last_batch_decision_requests[brain_name].agent_ids
            for r in range(len(agent_ids)):
                key = (agent_ids[r], brain_name)
                action = action_batch[r]
                env_actions[key] = action
                decision_request = self.decision_requests.pop(key)

                # add incomplete steps
                if decision_request.terminal_start_state:
                    if self.use_terminals_as_start_states:
                        self.incomplete_steps[key] = decision_request.to_incomplete_step(action)
                        #TODO: Find better place to reset custom stats when terminal state doesnt reset
                        if hasattr(self.env, 'reset_custom_stats'):
                            self.env.reset_custom_stats()
                        if hasattr(self.env, 'dictionary_env') and hasattr(self.env.dictionary_env, 'reset_custom_stats'):
                            self.env.dictionary_env.reset_custom_stats()

                else:
                    self.incomplete_steps[key] = decision_request.to_incomplete_step(action)

        step_data = self.env.step(env_actions)

        self.decision_requests = {}

        from deepagent.experiments.params.params import EnvironmentParams
        for step in step_data.steps():
			#TEST_MERGE_FLAG
            if EnvironmentParams.norm_funct is not None:
                step.state = EnvironmentParams.norm_funct(step.state)
            if EnvironmentParams.feature_extractor is not None:
                step.state = EnvironmentParams.feature_extractor(step.state, step.terminal, step.id)
            # add decision requests
            # Don't create decision requests on terminal states if env doesn't support that
            if self.env.use_terminals_as_start_states or not step.terminal:
                self.decision_requests[(step.id, step.unit_type)] = DecisionRequest(s0=step.state, mask=step.mask, terminal_start_state=step.terminal)

            # add complete steps
            if (step.id, step.unit_type) in self.incomplete_steps:
                incomplete_step = self.incomplete_steps.pop((step.id, step.unit_type))
                complete_step = incomplete_step.to_complete_step(r=step.reward, t=step.terminal, s1=step.state)
                self.completed_steps[(step.id, step.unit_type)] = complete_step
                env_id, unit_id = step.id
                self._add_reward(env_id, unit_id, complete_step.r)

        completed_episode_discounted_rewards = []

        for key, complete_step in self.completed_steps.items():
            env_id, unit_id = key[0]
            if complete_step.t:
                game_reward = self._calculate_individual_discount_rewards(env_id, unit_id)
                completed_episode_discounted_rewards.append(game_reward)
                complete_step.total_game_reward = game_reward
                if np.isnan(self._smoothed_discount_reward):
                    self._smoothed_discount_reward = game_reward  # Case where there are no previous logs for smoothing

        batch_decision_request = self._get_batch_decision_request()
        completed_steps = self._get_complete_steps_and_clear()

        return batch_decision_request, CompletedSteps(completed_steps, [data for data in step_data.env_terminals()]), completed_episode_discounted_rewards

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def _has_rewards(self, env_id):
        return env_id in self._environment_rewards and self._environment_rewards[env_id]

    def _add_reward(self, env_id, unit_id, r):
        if env_id not in self._environment_rewards:
            self._environment_rewards[env_id] = {}
        if unit_id not in self._environment_rewards[env_id]:
            self._environment_rewards[env_id][unit_id] = []
        self._environment_rewards[env_id][unit_id].append(r)

    def _calculate_individual_discount_rewards(self, env_id, unit_id):
        unit_rewards = copy.copy(self._environment_rewards[env_id][unit_id])
        del(self._environment_rewards[env_id][unit_id])

        discounted_reward = 0.0
        for reward in reversed(unit_rewards):
            discounted_reward = reward + self.gamma * discounted_reward
        return discounted_reward
