from baselines.common.vec_env.vec_frame_stack import VecFrameStack
import numpy as np


class DictionaryVecFrameStackWrapper(VecFrameStack):
    def __init__(self, vec_frame_stack, env_type):
        self.env_type = env_type
        VecFrameStack.__init__(self, vec_frame_stack.venv, vec_frame_stack.nstack)
        self.action_space = {env_type: [self.action_space]}
        self.observation_space = {env_type: [self.observation_space]}
        self.noop_action = 0
        self.episode_steps = {i:0 for i in range(self.venv.num_envs)}

    def step_wait(self):
        stacked_obs, rews, dones, infos = VecFrameStack.step_wait(self)

        self.infos = infos

        # zero center and normalize
        stacked_obs = [obs / 127.5 for obs in stacked_obs]
        stacked_obs = [obs - np.mean(obs) for obs in stacked_obs]

        stacked_obs = {((i,1), self.env_type): (stacked_obs,) for i, stacked_obs in enumerate(stacked_obs)}
        rews = {(i, 1): r for i, r in enumerate(rews)}
        dones = {(i, 1): n for i, n in enumerate(dones)}
        agent_ids = [key[0] for key in stacked_obs.keys()]
        masks = self._empty_masks(agent_ids=agent_ids)
        env_dones = self._env_dones_from_infos(infos)
        return stacked_obs, rews, dones, masks, env_dones

    def step(self, actions):
        if actions is None:
            return VecFrameStack.step(self, [self.noop_action for _ in range(self.venv.num_envs)])
        else:
            list_actions = [None]*self.venv.num_envs
            for i in range(self.venv.num_envs):
                key = ((i, 1), 'atari')
                if key in actions:
                    list_actions[i] = np.argmax(actions[key])
                else:
                    list_actions[i] = self.noop_action

            return VecFrameStack.step(self, list_actions)

    def reset(self):
        stacked_obs = super(DictionaryVecFrameStackWrapper, self).reset()
        stacked_obs = {((i,1), self.env_type): (stacked_obs,) for i, stacked_obs in enumerate(stacked_obs)}
        agent_ids = [(i,1) for i in range(len(stacked_obs))]
        rews, dones, masks, env_dones = self._empty_rewards(agent_ids), self._empty_terminals(
            agent_ids), self._empty_masks(agent_ids), self._empty_env_dones(agent_ids)
        return stacked_obs, rews, dones, masks, env_dones

    def _empty_rewards(self, agent_ids):
        return {agent_id: 0.0 for agent_id in agent_ids}

    def _empty_terminals(self, agent_ids):
        return {agent_id: False for agent_id in agent_ids}

    def _empty_masks(self, agent_ids):
        n = self.action_space[self.env_type][0].n
        return {agent_id: [1.0] * n for agent_id in agent_ids}

    def _empty_env_dones(self, agent_ids):
        return {agent_id: False for agent_id in agent_ids}

    @property
    def use_terminals_as_start_states(self):
        return True

    def _env_dones_from_infos(self, infos):
        env_dones = {}
        for i, info in enumerate(infos):
            env_dones[(i, 1)] = 'episode' in info
        return env_dones