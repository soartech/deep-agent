import math
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class AbstractGameEpisodes(ABC):
    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def append(self, episode):
        pass

    @abstractmethod
    def complete(self, **kwargs):
        pass


class GameEpisodes(AbstractGameEpisodes):
    def __init__(self):
        # (s, a, r)
        self.episodes = []

        # Return instead of reward
        # (s, a, R)
        self.s = None
        self.a = []
        self.R = []
        self.advantage = []

    def size(self):
        return len(self.episodes)

    def append(self, episode):
        self.episodes.append(episode)

    def complete(self, gamma):
        R = 0.0
        first = True

        for episode in reversed(self.episodes):
            s, v, a, r, v1, t = episode

            if first:
                R = v1
                first = False

            if t:
                R = 0.0

            R = r + gamma * R
            advantage = R - v

            # print('discounted reward: ', R)
            # print('advantage: ', advantage)

            if self.s == None:
                self.s = [[] for _ in s]
            for i, s0 in enumerate(s):
                self.s[i].append(s0)
            self.a.append(a)
            self.R.append(R)
            self.advantage.append(advantage)


class GameEpisodesGeneralizedAdvantage(AbstractGameEpisodes):
    def __init__(self):
        # (s, a, r)
        self.episodes = []

        self.s = None
        self.a = []
        self.R = []
        self.GAE = []

    def size(self):
        return len(self.episodes)

    def append(self, episode):
        self.episodes.append(episode)

    def complete(self, gamma, lam, episodic_reward):
        R = 0.0
        GAE = 0.0
        first = True
        for episode in reversed(self.episodes):
            s, v, a, r, v1, t = episode

            if t and episodic_reward:
                GAE = 0.0
                R = 0.0
                first = True
                final_value = 0.0
            elif first:
                R = v1
                final_value = v1

            if first:
                delta = r - (v - gamma * final_value)
                first = False
            else:
                delta = r - (v - gamma * v1)
            GAE = delta + gamma * lam * GAE
            #R = r + gamma * R
            # TD(lambda) estimate of return
            # https://www.davidsilver.uk/wp-content/uploads/2020/03/MC-TD.pdf
            # "Telescoping in TD(lambda)" slide
            R = GAE + v

            # print('Return: ', R)
            # print('generalized advantage estimate: ', GAE)

            if self.s == None:
                self.s = [[] for _ in s]
            for i, s0 in enumerate(s):
                self.s[i].append(s0)
            self.a.append(a)
            self.R.append(R)
            self.GAE.append(GAE)


class GameEpisodesGeneralizedAdvantageIntrinsicReward(AbstractGameEpisodes):
    def __init__(self):
        # (s, a, r)
        self.episodes = []

        self.s = None
        self.a = []

        self.R = []
        self.GAE = []

        self.IR = []
        self.GIAE = []

        self.s1 = None

    def size(self):
        return len(self.episodes)

    def append(self, episode):
        self.episodes.append(episode)

    def complete(self, gamma, gamma_intrinsic, lam, tensorboard_writer, step, running_mean_std):
        R = 0.0
        GAE = 0.0

        IR = 0.0
        GIAE = 0.0

        first = True

        irs = []
        irs_norm = []

        for episode in reversed(self.episodes):
            s, v, vi, a, r, ir, s1, v1, v1i, t = episode

            irs.append(ir)

            # normalize intrinsic reward
            ir = ir / math.sqrt(running_mean_std.var)

            irs_norm.append(ir)

            if t:
                GAE = 0.0
                R = 0.0

                # GIAE = 0.0

                first = True
                final_value = 0.0
                final_value_intrinsic = v1i
            elif first:
                R = v1
                final_value = v1

                final_value_intrinsic = v1i

            if first:
                delta = r - (v - gamma * final_value)
                delta_intrinsic = ir - (vi - gamma_intrinsic * final_value_intrinsic)
                IR = v1i
                first = False
            else:
                delta = r - (v - gamma * v1)
                delta_intrinsic = ir - (vi - gamma_intrinsic * v1i)
            GAE = delta + gamma * lam * GAE
            GIAE = delta_intrinsic + gamma_intrinsic * lam * GIAE
            # R = r + gamma*R
            # IR = ir + gamma_intrinsic*IR
            R = GAE + v
            IR = GIAE + vi

            if self.s == None:
                self.s = [[] for _ in s]
            for i, s0 in enumerate(s):
                self.s[i].append(s0)
            self.a.append(a)

            self.R.append(R)
            self.GAE.append(GAE)

            self.IR.append(IR)
            self.GIAE.append(GIAE)

            if self.s1 == None:
                self.s1 = [[] for _ in s1]
            for i, s11 in enumerate(s1):
                self.s1[i].append(s11)

        retintmean = np.mean(np.array(self.IR))
        retintstd = np.std(np.array(self.IR))
        #print('retintmean: ', retintmean)
        #print('retintstd: ', retintstd)

        retextmean = np.mean(np.array(self.R))
        retextstd = np.std(np.array(self.R))
        #print('retextmean: ', retextmean)
        #print('retextstd: ', retextstd)

        rewintmax_norm = np.mean(np.array(irs_norm))
        rewintmean_norm = np.max(np.array(irs_norm))
        #print('rewintmax_norm: ', rewintmax_norm)
        #print('rewintmean_norm: ', rewintmean_norm)

        rewintmax = np.mean(np.array(irs))
        rewintmean = np.max(np.array(irs))
        #print('rewintmax: ', rewintmax)
        #print('rewintmean: ', rewintmean)

        with tensorboard_writer.as_default():
            tf.summary.scalar('retintmean', retintmean, step=step)
            tf.summary.scalar('retintstd', retintstd, step=step)
            tf.summary.scalar('retextmean', retextmean, step=step)
            tf.summary.scalar('retextstd', retextstd, step=step)
            tf.summary.scalar('rewintmean_norm', rewintmean_norm, step=step)
            tf.summary.scalar('rewintmax_norm', rewintmax_norm, step=step)
            tf.summary.scalar('rewintmean', rewintmean, step=step)
            tf.summary.scalar('rewintmax', rewintmax, step=step)
            tf.summary.scalar('retintmean', retintmean, step=step)
