from __future__ import annotations

import os
import random
import shutil
import pickle
from abc import abstractmethod, ABC
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from gym import spaces
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop

from deepagent.agents.memory import PPOMemory, PPOLSTMMemory, BCMemory, A2CMemory
from deepagent.agents.util import clone_model, clone_model_with_new_weights
from deepagent.experiments.params import params

# type hint imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from deepagent.envs.spaces import DeepAgentSpace


class PolicyGradient(ABC):
    def __init__(self, unit_type, custom_objects, action_space: DeepAgentSpace, state_space: DeepAgentSpace):
        self.unit_type = unit_type
        self.custom_objects = custom_objects if custom_objects is not None else {}
        self.networks = dict()
        self.tensorboard_prefix = ''

        self.action_space = action_space
        self.state_space = state_space

    def is_single_action(self):
        return not isinstance(self.action_space.vector_space, spaces.MultiDiscrete)

    @abstractmethod
    def update_networks(self, step_count):
        pass

    def save_file(self, name, obj, prefix=''):
        file_path = self.file_path(prefix+name)

        if os.path.exists(file_path):
            backup = file_path + '.bkp'
            shutil.copy2(file_path, backup)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    def load_file(self, name, prefix=''):
        file_path = self.file_path(prefix + name)

        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_weights(self, prefix=''):
        for name, network in list(self.networks.items()):
            weights_file = self.weights_file_path(prefix+name)

            if os.path.exists(weights_file):
                backup = weights_file + '.bkp'
                print('Backing up weights file for {} network from {} to {}'.format(name, weights_file, backup))
                shutil.copy2(weights_file, backup)

            print('Saving weights for {} network to file: {}'.format(name, weights_file))
            network.save_weights(weights_file)

    def load_actor_critic_weights(self, file):
        self.networks['actor_critic'].load_weights(file)

    def load_weights(self, prefix=''):
        for name, network in list(self.networks.items()):
            weights_file = self.weights_file_path(prefix+name)
            if os.path.exists(weights_file):
                if weights_file.endswith('.pkl'):
                    print('Loading pkl weights for {} network from file: {}'.format(name, weights_file))
                    file = open(weights_file, 'rb')
                    weights = pickle.load(file)
                    network.set_weights(weights)
                else:
                    print('Loading weights for {} network from file: {}'.format(name, weights_file))
                    network.load_weights(weights_file)
            else:
                print('Warning: Weights file {} does not exist. Weights will not be loaded for {} network'.format(
                    weights_file, name))

    def load_sml_weights(self, num=None):
        for name, network in list(self.networks.items()):
            if num:
                path = os.path.join('SimpleMarineLearningWeights/'+str(num)+'/', '_'.join(['SimpleMarineLearning', name, 'weights.h5']))
            else:
                path = os.path.join('SimpleMarineLearningWeights','_'.join(['SimpleMarineLearning', name, 'weights.h5']))
            if os.path.exists(path):
                print('Loading weights for {} network from file: {}'.format(name, path))
                network.load_weights(path)
            else:
                print('Warning: Weights file {} does not exist. Weights will not be loaded for {} network'.format(path, name))

    def print_summary(self):
        for name, network in list(self.networks.items()):
            print('Summary for {} network:'.format(name))
            network.summary()

    def file_path(self, name):
        from deepagent.experiments.params import params
        return os.path.join(params.ModuleParams.weights_dir, name+'.pkl')

    def weights_file_path(self, network_name):
        return PolicyGradient.weights_file_path_static(network_name, self.unit_type)

    @staticmethod
    def weights_file_path_static(network_name, unit_type, file_type='.h5'):
        from deepagent.experiments.params import params
        # TODO: Handle this better, this is here because mlagents appends team_ids to agent behavior names
        # e.g. 3DBallAgent becomes 3DBallAgent?team=0
        unit_name = unit_type.replace('?', '_qmark_')
        if params.ModuleParams.weights_num is None:
            return os.path.join(params.ModuleParams.weights_dir,
                                '_'.join([unit_name, network_name, 'weights'+file_type]))
        weights_num = str(params.ModuleParams.weights_num)
        return os.path.join(params.ModuleParams.weights_dir,
                            '_'.join([unit_name, weights_num, network_name, 'weights'+file_type]))
        

    def set_tensorboard_writer(self, writer):
        self.writer = writer

    @abstractmethod
    def pre_train(self, bc_memory: BCMemory):
        pass


class A2C(PolicyGradient):
    def __init__(self, unit_type, actor_critic, custom_objects, action_space: DeepAgentSpace, state_space: DeepAgentSpace):
        super(A2C, self).__init__(unit_type, custom_objects, action_space, state_space)

        self.networks.clear()

        self.memory = A2CMemory()

        self.actor_critic = actor_critic
        self.networks['actor_critic'] = self.actor_critic

        self.actor_critic.optimizer=RMSprop(clipnorm=0.5, rho=0.99, epsilon=1e-5)

    @tf.function
    def get_local_policy_action(self, state):
        state_actor = self.state_space.get_policy_network_state(state)
        state_critic = self.state_space.get_value_function_state(state)
        if self.state_space.full_visibility_state:
            # different state representations, we need to do inference twice
            ret_actor = self.actor_critic(state_actor)
            ret_critic = self.actor_critic(state_critic)
        else:
            # state representations are the same, only do inference once
            ret_actor = self.actor_critic(state_actor)
            ret_critic = ret_actor

        if self.is_single_action():
            actions = ret_actor[0]
            v = ret_critic[2]
        else:
            actions = []
            for i in range(len(self.action_space.vector_space.nvec)):
                actions.append(ret_actor[i])
                v = ret_critic[-1]

        return actions, v

    @tf.function
    def v_loss(self, R_batch, critic_inputs):
        # inputs
        # all dimensions are (batch_size, )
        critic_inputs = self.state_space.get_value_function_state(critic_inputs)

        value_idx = -1

        V = tf.squeeze(self.actor_critic(critic_inputs)[value_idx])
        R = tf.stop_gradient(tf.squeeze(R_batch))

        V_square = tf.square(R - V)
        # dimension reduced to scalar value by mean
        V_loss = 0.5 * tf.reduce_mean(V_square)

        print('v_loss')
        print(V)
        print(R)

        return V_loss

    @tf.function
    def pi_loss(self, action_batch, advantage_batch, entropy_coefficient, actor_inputs):
        # inputs
        actor_inputs = self.state_space.get_policy_network_state(actor_inputs)
        pi_loss = 0

        if self.is_single_action():
            # pi outputs are shape (batch_size, num_actions)
            pi = self.actor_critic(actor_inputs)[0]
            pi_logits = self.actor_critic(actor_inputs)[1]
            # one hot representing sampled action with shape (batch_size, num_actions)
            actions = tf.stop_gradient(action_batch)
            # advantage has shape (batch_size, )
            advantage = tf.stop_gradient(advantage_batch)

            # policy gradient
            pg = tf.keras.losses.categorical_crossentropy(actions, pi_logits, from_logits=True)
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=pi_logits)
            # pg, advantage, and entropy have shape (batch_size, ) which is reduced to a scalar by reduce_mean
            pi_loss = tf.reduce_mean(pg * advantage) - (entropy_coefficient * tf.reduce_mean(entropy))
        else:
            N = len(self.action_space.vector_space.nvec)
            print('Multi action policy, N=', N)
            for i, num_actions in enumerate(self.action_space.vector_space.nvec):
                # pi outputs are shape (batch_size, num_actions)
                pi = self.actor_critic(actor_inputs)[i]
                pi_logits = self.actor_critic(actor_inputs)[N + i]
                # one hot representing sampled action with shape (batch_size, num_actions)
                actions = tf.stop_gradient(tf.one_hot(tf.cast(action_batch[:, i], tf.int32), num_actions))

                print('pi', i, ':', pi)
                print('pi_logits', i, ':', pi_logits)
                print('actions', i, ':', actions)

                # advantage has shape (batch_size, )
                advantage = tf.stop_gradient(advantage_batch)

                # policy gradient
                pg = tf.keras.losses.categorical_crossentropy(actions, pi_logits, from_logits=True)
                entropy = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=pi_logits)
                # pg, advantage, and entropy have shape (batch_size, ) which is reduced to a scalar by reduce_mean
                pi_loss += tf.reduce_mean(pg * advantage) - (entropy_coefficient * tf.reduce_mean(entropy))

        return pi_loss

    @tf.function
    def loss(self, R_batch, action_batch, advantage_batch, entropy_coefficient, inputs):
        v_loss = self.v_loss(R_batch, inputs)
        p_loss = self.pi_loss(action_batch, advantage_batch, entropy_coefficient, inputs)
        loss = p_loss + v_loss
        return loss, p_loss, v_loss

    def update(self, step_count):
        states = []
        actions = []
        returns = []
        advantages = []
        self.memory.randomize()
        for s, a, r, adv in self.memory.batch_generator(params.TrainingParams.batch_size):
            states.append(s)
            actions.append(a)
            returns.append(r)
            advantages.append(adv)

        self.update_networks(step_count, states, actions, returns, advantages)

        self.memory.clear()

    def update_networks(self, step_count, states, actions, rewards, advantage):
        from deepagent.experiments.params.params import TrainingFunctions

        lr = TrainingFunctions.lr_schedule(step_count)
        entropy_coefficient = TrainingFunctions.curiosity_schedule(step_count)

        with self.writer.as_default():
            tf.summary.scalar('lr', lr, step=step_count)
            tf.summary.scalar('entropy_coefficient', entropy_coefficient, step=step_count)

        self.actor_critic.optimizer.lr.assign(lr)

        self.update_networks_batch_args(tf.constant(step_count, tf.int64),
                                        states,
                                        actions,
                                        rewards,
                                        advantage,
                                        tf.constant(entropy_coefficient, tf.float32))

    @tf.function
    def update_networks_batch_args(self, step_count, states, actions, rewards, advantage, entropy_coefficient):
        from deepagent.experiments.params import params
        for _ in range(params.TrainingParams.num_updates):
            for state0_batch, action_batch, total_reward_batch, advantage_batch in zip(states, actions, rewards, advantage):
                total_reward_batch = tf.expand_dims(total_reward_batch, axis=-1)

                self.update_networks_tf(step_count, entropy_coefficient, state0_batch, action_batch, total_reward_batch, advantage_batch)

    @tf.function
    def update_networks_tf(self, step_count, entropy_coefficient, state0_batch, action_batch, total_reward_batch, advantage_batch):
        with tf.GradientTape() as tape:
            loss, p_loss, v_loss = self.loss(total_reward_batch, action_batch, advantage_batch, entropy_coefficient, state0_batch)
            gradients = tape.gradient(loss, self.actor_critic.trainable_variables)
            self.actor_critic.optimizer.apply_gradients(zip(gradients, self.actor_critic.trainable_variables))

        with self.writer.as_default():
            tf.summary.scalar('loss', loss, step=step_count)
            tf.summary.scalar('v_loss', v_loss, step=step_count)
            tf.summary.scalar('p_loss', p_loss, step=step_count)

    def pre_train(self):
        raise NotImplementedError


class PPO(PolicyGradient):
    def __init__(self, unit_type, actor_critic, custom_objects, action_space: DeepAgentSpace, state_space: DeepAgentSpace):
        super(PPO, self).__init__(unit_type, custom_objects, action_space, state_space)

        self.networks.clear()

        self.memory = PPOMemory()

        self.actor_critic = actor_critic
        self.optimizer = Adam(clipnorm=0.5, epsilon=1e-5)
        self.networks['actor_critic'] = self.actor_critic

        self.target_actor_critic = clone_model(actor_critic, custom_objects)

    def initialize_optimizer(self):
        self.optimizer = Adam(clipnorm=0.5, epsilon=1e-5)
        grad_vars = self.actor_critic.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        # apply gradients which don't change Adam state
        self.optimizer.apply_gradients(zip(zero_grads, grad_vars))

    @tf.function
    def get_local_policy_action(self, state):
        state_actor = self.state_space.get_policy_network_state(state)
        state_critic = self.state_space.get_value_function_state(state)
        if self.state_space.full_visibility_state:
            # different state representations, we need to do inference twice
            ret_actor = self.actor_critic(state_actor)
            ret_critic = self.actor_critic(state_critic)
        else:
            # state representations are the same, only do inference once
            ret_actor = self.actor_critic(state_actor)
            ret_critic = ret_actor

        if self.is_single_action():
            actions = ret_actor[0]
            v = ret_critic[2]
        else:
            actions = []
            for i in range(len(self.action_space.vector_space.nvec)):
                actions.append(ret_actor[i])
                v = ret_critic[-1]

        return actions, v

    @tf.function
    def v_loss(self, R_batch, clip_range, critic_inputs):
        # inputs
        # all dimensions are (batch_size, )
        critic_inputs = self.state_space.get_value_function_state(critic_inputs)

        value_idx = -1

        V = tf.squeeze(self.actor_critic(critic_inputs)[value_idx])
        V_old = tf.stop_gradient(tf.squeeze(self.target_actor_critic(critic_inputs)[value_idx]))
        R = tf.stop_gradient(tf.squeeze(R_batch))

        V_square = tf.square(R - V)
        # Not in the paper, but PPO2 in OpenAI baselines repo also clips the value function
        V_clipped = V_old + tf.clip_by_value(V - V_old, -1.0 * clip_range, clip_range)
        V_clipped_square = tf.square(R - V_clipped)
        # dimension reduced to scalar value by mean
        V_loss = 0.5 * tf.reduce_mean(tf.maximum(V_square, V_clipped_square))

        print('q_loss')
        print(V)
        print(V_old)
        print(R)
        print(V_clipped)
        print(V_clipped_square)

        return V_loss

    @tf.function
    def pi_loss(self, action_batch, advantage_batch, clip_range, entropy_coefficient, actor_inputs):
        actor_inputs = self.state_space.get_policy_network_state(actor_inputs)
        pi_loss = 0
        # single set of discrete actions
        if self.is_single_action():
            # pi outputs are shape (batch_size, num_actions)
            pi = self.actor_critic(actor_inputs)[0]
            pi_logits = self.actor_critic(actor_inputs)[1]
            pi_old_logits = tf.stop_gradient(self.target_actor_critic(actor_inputs)[1])
            # one hot representing sampled action with shape (batch_size, num_actions)
            actions = tf.stop_gradient(action_batch)
            # advantage has shape (batch_size, )
            advantage = tf.stop_gradient(K.squeeze(advantage_batch, axis=-1))

            # pi / pi_old = exp(log(pi / pi_old)) = exp(log(pi) - log(pi_old))
            # cross entropy gives negative log so we reverse the order of p_new and p_old
            ratio = tf.exp(tf.keras.losses.categorical_crossentropy(actions, pi_old_logits, from_logits=True) - tf.keras.losses.categorical_crossentropy(actions, pi_logits, from_logits=True))
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range)
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=pi_logits)
            # ratios, advantage, and entropy have shape (batch_size, ) which is reduced to a scalar by reduce_mean
            pi_loss = (-1.0 * tf.reduce_mean(tf.minimum(ratio * advantage, clipped_ratio * advantage))) - (entropy_coefficient * tf.reduce_mean(entropy))
        # multiple discrete sets of actions
        else:
            N = len(self.action_space.vector_space.nvec)
            print('Multi action policy, N=', N)
            for i, num_actions in enumerate(self.action_space.vector_space.nvec):
                # pi outputs are shape (batch_size, num_actions)
                pi = self.actor_critic(actor_inputs)[i]
                pi_logits = self.actor_critic(actor_inputs)[N+i]
                pi_old_logits = tf.stop_gradient(self.target_actor_critic(actor_inputs)[N+i])
                # one hot representing sampled action with shape (batch_size, num_actions)
                actions = tf.stop_gradient(tf.one_hot(tf.cast(action_batch[:,i], tf.int32), num_actions))

                print('pi', i, ':', pi)
                print('pi_logits', i, ':', pi_logits)
                print('actions', i, ':', actions)

                # advantage has shape (batch_size, )
                advantage = tf.stop_gradient(K.squeeze(advantage_batch, axis=-1))

                # pi / pi_old = exp(log(pi / pi_old)) = exp(log(pi) - log(pi_old))
                # cross entropy gives negative log so we reverse the order of p_new and p_old
                ratio = tf.exp(tf.keras.losses.categorical_crossentropy(actions, pi_old_logits, from_logits=True) - tf.keras.losses.categorical_crossentropy(actions, pi_logits, from_logits=True))
                clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range)
                entropy = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=pi_logits)
                # ratios, advantage, and entropy have shape (batch_size, ) which is reduced to a scalar by reduce_mean
                pi_loss += (-1.0 * tf.reduce_mean(tf.minimum(ratio * advantage, clipped_ratio * advantage))) - (entropy_coefficient * tf.reduce_mean(entropy))

        return pi_loss

    @tf.function
    def pi_loss_behavioral_cloning(self, action_batch, actor_inputs):
        # inputs
        # pi outputs are shape (batch_size, num_actions)
        pi = self.actor_critic(actor_inputs)[0]
        pi_logits = self.actor_critic(actor_inputs)[1]
        # one hot representing sampled action with shape (batch_size, num_actions)
        actions = tf.stop_gradient(action_batch)

        crossentropy = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(actions, pi_logits, from_logits=True))

        print('p_loss')
        print(pi)
        print(pi_logits)
        print(actions)
        print(crossentropy)

        return crossentropy

    @tf.function
    def loss(self, R_batch, action_batch, advantage_batch, clip_range, entropy_coefficient, inputs):
        return self.v_loss(R_batch, clip_range, inputs) + self.pi_loss(action_batch, advantage_batch, clip_range, entropy_coefficient, inputs)

    def update(self, step_count):
        states = []
        actions = []
        returns = []
        gae = []
        self.memory.randomize()
        for s, a, r, g in self.memory.batch_generator(params.TrainingParams.batch_size):
            states.append(s)
            actions.append(a)
            returns.append(r)
            normalized_gae_batch = (g - np.mean(g)) / (np.std(g) + 1e-8)
            normalized_gae_batch = np.expand_dims(normalized_gae_batch, axis=-1)
            gae.append(normalized_gae_batch)

        loss = self.update_networks(step_count, states, actions, returns, gae)

        self.memory.clear()

        return loss

    def update_networks(self, step_count, states, actions, rewards, gae, previous_weights=None):
        from deepagent.experiments.params.params import TrainingFunctions

        lr = TrainingFunctions.lr_schedule(step_count)
        clip = TrainingFunctions.clip_schedule(step_count)

        entropy_coefficient = TrainingFunctions.curiosity_schedule(step_count)

        # weights at start of update
        if previous_weights is None:
            self.target_actor_critic.set_weights(self.actor_critic.get_weights())
        else:
            self.target_actor_critic.set_weights(previous_weights)

        self.optimizer.lr.assign(lr)

        with self.writer.as_default():
            tf.summary.scalar(self.tensorboard_prefix+'lr', lr, step=step_count)
            tf.summary.scalar(self.tensorboard_prefix+'clip', clip, step=step_count)
            tf.summary.scalar(self.tensorboard_prefix+'entropy_coefficient', entropy_coefficient, step=step_count)

        loss = self.update_networks_batch_args(tf.constant(step_count, tf.int64),
                                               states,
                                               actions,
                                               rewards,
                                               gae,
                                               tf.constant(clip, tf.float32),
                                               tf.constant(entropy_coefficient, tf.float32),
                                               self.tensorboard_prefix)

        return loss

    @tf.function
    def update_networks_batch_args(self, step_count, states, actions, rewards, gae, clip, entropy_coefficient, tensorboard_prefix):
        loss = 0.0
        from deepagent.experiments.params import params
        for _ in range(params.TrainingParams.num_updates):
            for state0_batch, action_batch, total_reward_batch, gae_batch in zip(states, actions, rewards, gae):
                total_reward_batch = tf.expand_dims(total_reward_batch, axis=-1)

                loss += self.update_networks_tf(step_count, clip, entropy_coefficient, state0_batch, action_batch, total_reward_batch, gae_batch, tensorboard_prefix)

        return loss

    @tf.function
    def update_networks_tf(self, step_count, clip, entropy_coefficient, state0_batch, action_batch, total_reward_batch, normalized_gae_batch, tensorboard_prefix):
        with tf.GradientTape() as tape:
            loss = self.loss(total_reward_batch, action_batch, normalized_gae_batch, clip, entropy_coefficient, state0_batch)
            gradients = tape.gradient(loss, self.actor_critic.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.actor_critic.trainable_variables))

        with self.writer.as_default():
            tf.summary.scalar(tensorboard_prefix+'loss', loss, step=step_count)

        return loss

    def update_networks_behavioral_cloning(self, train, num_train, batch_size, epoch_count):
        for state_batch, action_batch in tqdm(train):
            self.update_networks_tf_behavioral_cloning_batch(tf.convert_to_tensor(state_batch), tf.convert_to_tensor(action_batch), tf.constant(num_train, tf.int64),
                                                       tf.constant(batch_size, tf.int64),
                                                       tf.constant(epoch_count, tf.int64))

    @tf.function
    def update_networks_tf_behavioral_cloning_batch(self, state_batch, action_batch, num_examples, batch_size, epoch_count):
        batches_per_epoch = tf.cast(num_examples / batch_size, tf.int64)
        step_count = epoch_count * batches_per_epoch
        with tf.GradientTape() as tape:
            p_loss = self.pi_loss_behavioral_cloning(action_batch, state_batch)
            gradients = tape.gradient(p_loss, self.actor_critic.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.actor_critic.trainable_variables))

            with self.writer.as_default():
                tf.summary.scalar(self.tensorboard_prefix+'p_loss', p_loss, step=step_count)

            step_count += 1

    def randomize(self, dataset_split):
        seed = 1234

        random.Random(seed).shuffle(dataset_split)
        return dataset_split

    def pre_train(self, bc_memory: BCMemory):
        batch_size = 32

        # bc_memory.randomize()
        print('Converting loaded data into train, validate, and test split generators.')
        train, num_train, validate, num_validate, test, num_test = bc_memory.batch_generator_splits(train_percent=.8,
                                                                                            validate_percent=.1,
                                                                                            test_percent=.1,
                                                                                            batch_size=batch_size)
        
        if isinstance(train, tf.data.Dataset):
            train = train.shuffle(10000, reshuffle_each_iteration=True)
            train = train.batch(batch_size, drop_remainder=True)
            validate = validate.shuffle(10000, reshuffle_each_iteration=True)
            validate = validate.batch(batch_size, drop_remainder=True)
            test = test.batch(batch_size, drop_remainder=True)
        else:
            train = list(train)
            validate = list(validate)
            test = list(test)

            train = self.randomize(train)
            validate = self.randomize(validate)

        num_train_batches = int(num_train / batch_size)

        prev_accuracy = 0
        epochs_since_improvement = 0
        max_epochs_w_no_improvement_before_stopping = 5
        epochs = 300
        self.optimizer.lr.assign(1e-4)

        print('Starting PPO pre_training...'
              f'\nmax_epochs={epochs} stop_after_epochs_w_no_improvement={max_epochs_w_no_improvement_before_stopping}'
              f'\nexamples: train={num_train} validate={num_validate} test={num_test}')
        for i in range(epochs):
            if not isinstance(train, tf.data.Dataset):
                # TODO: Do 1/10th random swaps between batches, this is not a great way to shuffle
                for _ in range(int(num_train / 10)):
                    batch_1 = train[random.randint(0, num_train_batches - 1)]
                    batch_2 = train[random.randint(0, num_train_batches - 1)]
                    idx = random.randint(0, batch_size - 1)
                    temp_state = batch_1[0][idx]
                    temp_action = batch_1[1][idx]
                    batch_1[0][idx] = batch_2[0][idx]
                    batch_1[1][idx] = batch_2[1][idx]
                    batch_2[0][idx] = temp_state
                    batch_2[1][idx] = temp_action

            self.update_networks_behavioral_cloning(train, num_train, batch_size, i)

            num_correct = self.test_networks_tf_behavioral_cloning(validate, tf.cast(i, tf.int64))

            total = num_validate
            num_correct = num_correct.numpy().astype(np.float32)
            accuracy = num_correct / total
            print(f'epoch_{i}: validation accuracy = {accuracy}')
            if accuracy > prev_accuracy:
                prev_accuracy = accuracy
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement == max_epochs_w_no_improvement_before_stopping:
                    print(f'no improvements for {epochs_since_improvement}, aborting training...')
                    break

            self.save_weights(prefix=f'epoch{i}_')

        num_correct = self.test_networks_tf_behavioral_cloning(test, tf.cast(i, tf.int64))

        total = num_test
        num_correct = num_correct.numpy().astype(np.float32)
        accuracy = num_correct / total
        print('Pre-training Results:'
              f'\n\ttest accuracy = {accuracy}, num_test_examples = {num_test}, num_train_examples = {num_train}')

        self.save_weights()

    @tf.function
    def test_networks_tf_behavioral_cloning(self, tf_data, epoch_count):
        num_correct = tf.cast(0, tf.int64)
        for states, actions in tf_data:
            predicted_actions = self.actor_critic(states)[0]
            num_correct += tf.cast(tf.reduce_sum(predicted_actions * actions), tf.int64)

        # max_prob = tf.math.reduce_max(predicted_actions_list)
        # with self.writer.as_default():
        #     tf.summary.scalar(self.tensorboard_prefix+'maximum_pretrain_probability', max_prob, step=epoch_count)

        return num_correct

