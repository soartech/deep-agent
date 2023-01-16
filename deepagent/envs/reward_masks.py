from __future__ import annotations

from enum import Enum

import numpy as np


class RewardTypes(Enum):
    @classmethod
    def reward_types(cls, type_string):
        ret = []
        for rt in cls:
            if rt.name.lower().startswith(type_string):
                ret.append(rt)
        return ret


class RewardMasks:
    def __init__(self, reward_types):
        self.masks = {t: 1.0 for t in reward_types}

    def randomize(self):
        for reward_type in self.masks.keys():
            # log uniform on [0.1, 10.0]
            self.masks[reward_type] = np.exp(np.random.uniform(np.log(0.1), np.log(10.0)))

    def set_masks(self, reward_masks: RewardMasks):
        for k, v in reward_masks.masks.items():
            self.masks[k] = v