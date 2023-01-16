import random
import warnings
from collections import deque, namedtuple

import numpy as np
import tensorflow as tf
from past.builtins import xrange
from past.utils import old_div

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')
DiffExperience = namedtuple('DiffExperience',
                            'state0_actor, state0_critic, action, reward, state1_actor, state1_critic, terminal1')


def sample_batch_indexes(low, high, size):
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        try:
            r = xrange(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn(
            'Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


class RingBuffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def zeroed_observation(observation):
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.


class Memory:
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config


class SequentialDiffMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialDiffMemory, self).__init__(**kwargs)

        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.actor_observations = RingBuffer(limit)
        self.critic_observations = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            # Draw random indexes such that we have at least a single entry before each index.
            batch_idxs = sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            assert 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = self.observations[idx - 1]
            state0_actor = self.actor_observations[idx - 1]
            state0_critic = self.critic_observations[idx - 1]
            for offset in range(0, self.window_length - 1):
                raise RuntimeError("window length not supported")
                # current_idx = idx - 2 - offset
                # current_terminal = self.terminals[current_idx - 1] if current_idx - 1 > 0 else False
                # if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                #     # The previously handled observation was terminal, don't add the current one.
                #     # Otherwise we would leak into a different episode.
                #     break
                # state0.insert(0, self.observations[current_idx])
                # state0_actor.insert(0, self.actor_observations[current_idx])
                # state0_critic.insert(0, self.critic_observations[current_idx])
            while len(state0) < self.window_length:
                raise RuntimeError("Window length not supported")
                # state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = self.observations[idx]
            state1_actor = self.actor_observations[idx]
            state1_critic = self.critic_observations[idx]

            assert len(state1) == len(state0)
            experiences.append(DiffExperience(state0_actor=state0_actor, state0_critic=state0_critic,
                                              action=action, reward=reward,
                                              state1_actor=state1_actor, state1_critic=state1_critic,
                                              terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation, actor_observation, critic_observation, action, reward, terminal, training=True):
        self.recent_observations.append(actor_observation)
        self.recent_terminals.append(terminal)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actor_observations.append(actor_observation)
            self.critic_observations.append(critic_observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        return len(self.observations)


class SequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)

        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            # Draw random indexes such that we have at least a single entry before each
            # index.
            batch_idxs = sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            assert 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                current_terminal = self.terminals[current_idx - 1] if current_idx - 1 > 0 else False
                if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation, action, reward, terminal, training=True):
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        return len(self.observations)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config


class EpisodeParameterMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(EpisodeParameterMemory, self).__init__(**kwargs)
        self.limit = limit

        self.params = RingBuffer(limit)
        self.intermediate_rewards = []
        self.total_rewards = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(0, self.nb_entries, size=batch_size)
        assert len(batch_idxs) == batch_size

        batch_params = []
        batch_total_rewards = []
        for idx in batch_idxs:
            batch_params.append(self.params[idx])
            batch_total_rewards.append(self.total_rewards[idx])
        return batch_params, batch_total_rewards

    def append(self, observation, action, reward, terminal, training=True):
        super(EpisodeParameterMemory, self).append(observation, action, reward, terminal, training=training)
        if training:
            self.intermediate_rewards.append(reward)

    def finalize_episode(self, params):
        total_reward = sum(self.intermediate_rewards)
        self.total_rewards.append(total_reward)
        self.params.append(params)
        self.intermediate_rewards = []

    @property
    def nb_entries(self):
        return len(self.total_rewards)

    def get_config(self):
        config = super(EpisodeParameterMemory, self).get_config()
        config['limit'] = self.limit
        return config


class A2CMemory:
    def __init__(self):
        self.states = None
        self.actions = []
        self.returns = []
        self.advantages = []

    def add_lists(self, states, actions, returns, advantages):
        if self.states == None:
            self.states = [[] for _ in states]
        for i in range(len(self.states)):
            self.states[i].extend(states[i])
        self.actions.extend(actions)
        self.returns.extend(returns)
        self.advantages.extend(advantages)

    def randomize(self):
        seed = 1234

        for i in range(len(self.states)):
            random.Random(seed).shuffle(self.states[i])
        random.Random(seed).shuffle(self.actions)
        random.Random(seed).shuffle(self.returns)
        random.Random(seed).shuffle(self.advantages)

    def clear(self):
        self.states = None
        self.actions = []
        self.returns = []
        self.advantages = []

    def num_batches(self, batch_size):
        return old_div(len(self.states[0]), batch_size)

    def batch_generator(self, batch_size):

        num_batches = self.num_batches(batch_size)

        for i in range(num_batches):
            states = [[] for _ in self.states]
            actions = []
            returns = []
            advantages = []

            for b in range(batch_size):
                idx = i * batch_size + b
                for i in range(len(states)):
                    states[i].append(self.states[i][idx])
                actions.append(self.actions[idx])
                returns.append(self.returns[idx])
                advantages.append(self.advantages[idx])

            states_np = [np.array(states[i], dtype=np.float32) for i in range(len(states))]

            yield states_np,\
                  np.array(actions, dtype=np.float32),\
                  np.array(returns, dtype=np.float32),\
                  np.array(advantages, dtype=np.float32)


class BCMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.dataset = None  # TF dataset. Used when loading from recording files, in place of self.states and self.actions.

    def add_lists(self, states, actions):
        self.states.extend(states)
        self.actions.extend(actions)

    def randomize(self):
        seed = 1234

        random.Random(seed).shuffle(self.states)
        random.Random(seed).shuffle(self.actions)

    def clear(self):
        self.states = []
        self.actions = []

    def num_batches(self, batch_size):
        return old_div(len(self.states), batch_size)

    def tf_generator(self):
        import tensorflow as tf
        for i in range(len(self.states)):
            yield tf.cast(self.states[i], tf.float32), tf.cast(self.actions[i], tf.float32)

    def batch_generator(self, batch_size, start_idx, end_idx):
        num_batches = old_div(end_idx - start_idx, batch_size)

        for i in range(start_idx, start_idx + num_batches * batch_size, batch_size):
            states = []
            actions = []

            for idx in range(i, i + batch_size):
                states.append(self.states[idx])
                actions.append(self.actions[idx])

            yield np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)

    def batch_generator_splits(self, train_percent: float = .8, validate_percent: float = .1, test_percent: float = .1,
                               batch_size: int = 32):
        total_percent = train_percent + validate_percent + test_percent
        if total_percent != 1:
            raise ValueError(f'train_percent+validate_percent+test_percent must equal 1, but found: {train_percent}+'
                             f'{validate_percent}+{test_percent}={total_percent}')

        if self.dataset is not None:
            num_examples = len(self.dataset)
        else:
            num_examples = len(self.actions)
        num_train = int(train_percent * num_examples)
        num_validate = int(validate_percent * num_examples)
        num_test = int(test_percent * num_examples)
        validate_end = num_train + num_validate

        if self.dataset is not None:
            train_gen = self.dataset.take(num_train)
            valid_gen = self.dataset.take(num_validate)
            test_gen = self.dataset.take(num_test)
        else:
            train_gen = self.batch_generator(batch_size, 0, num_train)
            valid_gen = self.batch_generator(batch_size, num_train, validate_end)
            test_gen = self.batch_generator(batch_size, validate_end, num_examples)

        num_train = num_train - num_train % batch_size
        num_validate = num_validate - num_validate % batch_size
        num_test = num_test - num_test % batch_size

        return train_gen, num_train, valid_gen, num_validate, test_gen, num_test


    def to_tf_datasets(self, train_percent: float = .8, validate_percent: float = .1, test_percent: float = .1,
                       batch_size: int = 32):
        total_percent = train_percent + validate_percent + test_percent
        if total_percent != 1:
            raise ValueError(f'train_percent+validate_percent+test_percent must equal 1, but found: {train_percent}+'
                             f'{validate_percent}+{test_percent}={total_percent}')

        num_examples = len(self.actions)
        num_train = int(train_percent * num_examples)
        num_validate = int(validate_percent * num_examples)
        num_test = int(test_percent * num_examples)
        validate_end = num_train + num_validate

        train = tf.data.Dataset.from_tensor_slices((self.states[:num_train], self.actions[:num_train]))
        validate = tf.data.Dataset.from_tensor_slices(
            (self.states[num_train:validate_end], self.actions[num_train:validate_end]))
        test = tf.data.Dataset.from_tensor_slices((self.states[validate_end:], self.actions[validate_end:]))

        train = train.shuffle(2000).batch(batch_size, drop_remainder=True)
        validate = validate.batch(batch_size, drop_remainder=True)
        test = test.batch(batch_size, drop_remainder=True)

        return train, num_train, validate, num_validate, test, num_test

    def action_distribution(self):
        action_counts = [0 for _ in range(self.actions[0].shape[0])]

        for action in self.actions:
            action_counts[int(np.argmax(action))] += 1

        num_actions = len(self.actions)
        distribution = [count / num_actions for count in action_counts]

        return action_counts, distribution

    def state_mean_and_std(self):
        return np.mean(self.states, axis=0), np.std(self.states, axis=0)


class PPOMemory:
    def __init__(self):
        self.states = None
        self.actions = []
        self.returns = []
        self.generalized_advantage_estimate = []

    def add_lists(self, states, actions, returns, GAE):
        if self.states == None:
            self.states = [[] for _ in states]
        for i in range(len(self.states)):
            self.states[i].extend(states[i])
        self.actions.extend(actions)
        self.returns.extend(returns)
        self.generalized_advantage_estimate.extend(GAE)

    def randomize(self):
        seed = 1234

        for i in range(len(self.states)):
            random.Random(seed).shuffle(self.states[i])
        random.Random(seed).shuffle(self.actions)
        random.Random(seed).shuffle(self.returns)
        random.Random(seed).shuffle(self.generalized_advantage_estimate)

    def normalize_gae(self):
        self.generalized_advantage_estimate = (self.generalized_advantage_estimate - np.mean(self.generalized_advantage_estimate)) / (np.std(self.generalized_advantage_estimate) + 1e-8)

    def clear(self):
        self.states = None
        self.actions = []
        self.returns = []
        self.generalized_advantage_estimate = []

    def num_batches(self, batch_size):
        return old_div(len(self.states[0]), batch_size)

    def batch_generator(self, batch_size):

        num_batches = self.num_batches(batch_size)

        for i in range(num_batches):
            states = [[] for _ in self.states]
            actions = []
            returns = []
            gae = []

            for b in range(batch_size):
                idx = i * batch_size + b
                for j in range(len(states)):
                    states[j].append(self.states[j][idx])
                actions.append(self.actions[idx])
                returns.append(self.returns[idx])
                gae.append(self.generalized_advantage_estimate[idx])

            states_np = [np.array(states[j], dtype=np.float32) for j in range(len(states))]
            yield states_np,\
                  np.array(actions, dtype=np.float32),\
                  np.array(returns, dtype=np.float32),\
                  np.array(gae, dtype=np.float32)


class PPOLSTMMemory:
    def __init__(self):
        self.states = None
        self.actions = []
        self.returns = []
        self.generalized_advantage_estimate = []
        self.seq_len = 250

    def add_lists(self, states, actions, returns, GAE):
        if self.states == None:
            self.states = [[] for _ in states]
        for i in range(len(self.states)):
            self.states[i].extend(self.split_and_pad(states[i]))
        self.actions.extend(self.split_and_pad(actions))
        self.returns.extend(self.split_and_pad(returns))
        self.generalized_advantage_estimate.extend(self.split_and_pad(GAE))

    def split_and_pad(self, list):
        list_of_lists = [list[x:x + self.seq_len] for x in range(0, len(list), self.seq_len)]
        last = list_of_lists[-1]
        pad_value = np.zeros_like(last[0])
        last += [pad_value] * (self.seq_len - len(last))

        return list_of_lists

    def randomize(self):
        seed = 1234

        for i in range(len(self.states)):
            random.Random(seed).shuffle(self.states[i])
        random.Random(seed).shuffle(self.actions)
        random.Random(seed).shuffle(self.returns)
        random.Random(seed).shuffle(self.generalized_advantage_estimate)

    def normalize_gae(self):
        self.generalized_advantage_estimate = (self.generalized_advantage_estimate - np.mean(self.generalized_advantage_estimate)) / (np.std(self.generalized_advantage_estimate) + 1e-8)

    def clear(self):
        self.states = None
        self.actions = []
        self.returns = []
        self.generalized_advantage_estimate = []

    def num_batches(self, batch_size):
        return old_div(len(self.states[0]), int(batch_size/self.seq_len))

    def batch_generator(self, batch_size):
        num_batches = self.num_batches(batch_size)
        batch_size = int(batch_size/self.seq_len)

        for i in range(num_batches):
            states = [[] for _ in self.states]
            actions = []
            returns = []
            gae = []

            for b in range(batch_size):
                idx = i * batch_size + b
                for j in range(len(states)):
                    states[j].append(self.states[j][idx])
                actions.append(self.actions[idx])
                returns.append(self.returns[idx])
                gae.append(self.generalized_advantage_estimate[idx])

            states_np = [np.array(states[j], dtype=np.float32) for j in range(len(states))]
            yield states_np,\
                  np.array(actions, dtype=np.float32),\
                  np.array(returns, dtype=np.float32),\
                  np.array(gae, dtype=np.float32)


class PPOCuriosityMemory:
    def __init__(self):
        self.states = None
        self.actions = []
        self.returns = []
        self.intrinsic_returns = []
        self.next_states = None
        self.generalized_advantage_estimate = []
        self.generalized_intrinsic_advantage_estimate = []

    def add_lists(self, states, actions, returns, intrinsic_returns, next_states, GAE, GIAE):
        if self.states == None:
            self.states = [[] for _ in states]
        for i in range(len(self.states)):
            self.states[i].extend(states[i])
        self.actions.extend(actions)
        self.returns.extend(returns)
        self.intrinsic_returns.extend(intrinsic_returns)
        if self.next_states == None:
            self.next_states = [[] for _ in next_states]
        for i in range(len(self.next_states)):
            self.next_states[i].extend(next_states[i])
        self.generalized_advantage_estimate.extend(GAE)
        self.generalized_intrinsic_advantage_estimate.extend(GIAE)

    def randomize(self):
        seed = 1234

        for i in range(len(self.states)):
            random.Random(seed).shuffle(self.states[i])
        random.Random(seed).shuffle(self.actions)
        random.Random(seed).shuffle(self.returns)
        random.Random(seed).shuffle(self.intrinsic_returns)
        for i in range(len(self.next_states)):
            random.Random(seed).shuffle(self.next_states[i])
        random.Random(seed).shuffle(self.generalized_advantage_estimate)
        random.Random(seed).shuffle(self.generalized_intrinsic_advantage_estimate)

    def clear(self):
        self.states = None
        self.actions = []
        self.returns = []
        self.intrinsic_returns = []
        self.next_states = None
        self.generalized_advantage_estimate = []
        self.generalized_intrinsic_advantage_estimate = []

    def num_batches(self, batch_size):
        return old_div(len(self.states[0]), batch_size)

    def batch_generator(self, batch_size):
        # TODO needs to be updated for image+vector input

        num_batches = self.num_batches(len(self.states[0]))

        states_shape = (batch_size,) + self.states[0][0].shape
        actions_shape = (batch_size,) + self.actions[0].shape
        returns_shape = (batch_size,)

        for i in range(num_batches):
            states = [np.empty(states_shape, dtype=np.float32)]
            actions = np.empty(actions_shape, dtype=np.float32)
            returns = np.empty(returns_shape, dtype=np.float32)
            intrinsic_returns = np.empty(returns_shape, dtype=np.float32)
            next_states = [np.empty(states_shape, dtype=np.float32)]
            gae = np.empty(returns_shape, dtype=np.float32)
            giae = np.empty(returns_shape, dtype=np.float32)

            for b in range(batch_size):
                idx = i * batch_size + b

                states[b, :, :, :] = self.states[idx][0]
                actions[b, :] = self.actions[idx]
                returns[b] = self.returns[idx]
                intrinsic_returns[b] = self.intrinsic_returns[idx]
                next_states[b, :, :, :] = self.next_states[idx][0]
                gae[b] = self.generalized_advantage_estimate[idx]
                giae[b] = self.generalized_intrinsic_advantage_estimate[idx]

            yield [states], actions, returns, intrinsic_returns, [next_states], gae, giae
