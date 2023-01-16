from deepagent.agents.a2c import A2CAgentTrainer
from deepagent.agents.common import AbstractAgentFactory, AgentFactoryException
from deepagent.agents.keyboard_agent import KeyboardGradient
from deepagent.agents.policy_gradient import A2C, PPO, PPORND, PPOLSTM
from deepagent.agents.ppo import PPOAgentTrainer
from deepagent.agents.random_agent import RandomGradient, DebugAgentTester, RacerDijsktraTester
from deepagent.envs.spaces import DeepAgentSpace
from deepagent.experiments.params import params
from deepagent.agents.util import get_my_ip
import deepagent.networks.actor_critic_dense as actor_critic_dense

class NatureCNNA2CAgentFactory(AbstractAgentFactory):
    def multi_agent_training_class(self):
        return A2CAgentTrainer

    def create_agent(self, agent_type: str, state_space: DeepAgentSpace, action_space: DeepAgentSpace):
        if state_space.vector_space and state_space.image_spaces:
            actor_critic, _, __, ___ = actor_critic_dense.create_actor_critic_image_vector(state_space, action_space)
            return A2C(agent_type, actor_critic, self.custom_objects, state_space)
        if state_space.image_spaces:
            actor_critic, _, __, ___ = actor_critic_dense.create_actor_critic_image(state_space, action_space)
            return A2C(agent_type, actor_critic, self.custom_objects, state_space)
        if state_space.vector_space and action_space.vector_space:
            actor_critic, _, __, ___ = actor_critic_dense.create_actor_critic_vector(state_space, action_space)
            return A2C(agent_type, actor_critic, self.custom_objects, state_space)
        else:
            raise AgentFactoryException('This Method current has not other implementation')


class PPOAgentFactory(AbstractAgentFactory):
    def multi_agent_training_class(self):
        return PPOAgentTrainer

    def create_agent(self, agent_type: str, state_space: DeepAgentSpace, action_space: DeepAgentSpace):
        if state_space.vector_space and state_space.image_spaces:
            actor_critic, _, __, ___ = actor_critic_dense.create_actor_critic_image_vector(state_space, action_space)
            return PPO(agent_type, actor_critic, self.custom_objects, action_space, state_space)
        if state_space.image_spaces:
            actor_critic, _, __, ___ = actor_critic_dense.create_actor_critic_image(state_space, action_space)
            return PPO(agent_type, actor_critic, self.custom_objects, action_space, state_space)
        if state_space.vector_space and action_space.vector_space:
            actor_critic, _, __, ___ = actor_critic_dense.create_actor_critic_vector(state_space, action_space)
            return PPO(agent_type, actor_critic, self.custom_objects, action_space, state_space)
        else:
            raise AgentFactoryException('This Method current has not other implementation')


class RandomAgentFactory(AbstractAgentFactory):
    def create_agent(self, agent_type: str, state_space: DeepAgentSpace, action_space: DeepAgentSpace):
        return RandomGradient(action_space)

    def multi_agent_training_class(self):
        return DebugAgentTester


class RacerTestFactory(AbstractAgentFactory):
    def create_agent(self, agent_type: str, state_space: DeepAgentSpace, action_space: DeepAgentSpace):
        return RandomGradient(action_space)

    def multi_agent_training_class(self):
        return RacerDijsktraTester


class KeyboardAgentFactory(AbstractAgentFactory):
    def create_agent(self, agent_type: str, state_space: DeepAgentSpace, action_space: DeepAgentSpace):
        return KeyboardGradient(action_space)

    def multi_agent_training_class(self):
        return DebugAgentTester
