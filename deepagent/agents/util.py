from typing import List, Union, Iterable

import netifaces
from past.utils import old_div

import numpy as np
from tensorflow.keras.models import model_from_config
from six import iteritems


def get_my_ip():
    for interface in netifaces.interfaces():
        for v in netifaces.ifaddresses(interface).values():
            for vv in v:
                if vv['addr'].startswith('192'):
                    return vv['addr']
    return None

def check_ip(ip_list):
    for interface in netifaces.interfaces():
        for v in netifaces.ifaddresses(interface).values():
            for vv in v:
                for ip in ip_list:
                    if vv['addr'] == ip:
                        return ip
    return None

def calculate_discounted_reward(rewards, gamma):
    discounted_reward = 0.0
    for r in reversed(rewards):
        # r is a dict of unit id to reward
        discounted_reward = np.sum(list(r.values())) + gamma*discounted_reward

    print('Discounted reward for episode with ', len(rewards), ' steps: ', discounted_reward)
    return discounted_reward

def clone_model_with_new_weights(model, custom_objects={}):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    return clone

def clone_model(model, custom_objects={}):
    clone = clone_model_with_new_weights(model, custom_objects)
    clone.set_weights(model.get_weights())
    return clone


def all_terminal(terminal_or_terminal_dict):
    if type(terminal_or_terminal_dict) == dict:
        return all(terminal_or_terminal_dict.values())
    return terminal_or_terminal_dict


def normalize(probs: np.ndarray, valid_action_indices: Union[Iterable, np.ndarray]):
    probs_sum = np.sum(probs)
    if probs_sum == 0:
        if len(valid_action_indices) == 0:
            print(f'Warning: Encountered all-zero action probability.')
            probs[-1] = 1.0
        else:
            probs[valid_action_indices] = 1.0 / len(valid_action_indices)

        probs_sum = np.sum(probs)

    return old_div(probs, probs_sum)


def safemean(xs):
    """
    Avoids division error when calculating the mean.
    :param xs: Numpy array or array-like.
    :return: The mean or np.nan if it would otherwise error.
    """
    return np.nan if len(xs) == 0 else np.mean(xs)


def get_atari_epinfos(env):
    epinfos = []
    for info in iteritems(env.infos):
        envinfo = info[1]
        maybeepinfo = envinfo.get('episode')
        if maybeepinfo: epinfos.append(maybeepinfo)
    return epinfos


def is_atari(env):
    return env.env_type == 'atari'
