from enum import Enum
from typing import Optional, NamedTuple, List, Union

from gym import spaces
import numpy as np


class FullVisibilityTypes(Enum):
    ValueFunctionState = 0
    PolicyFunctionState = 1
    Both = 2


class DeepAgentSpace:
    def __init__(self, image_spaces: Optional[Union[List[spaces.Box], spaces.Box]] = None, vector_space: Optional[spaces.Box] = None, agent_name: Optional[str] = None,
                 full_visibility_mask: List[FullVisibilityTypes] = None):
        if image_spaces is not None and type(image_spaces) != list:
            image_spaces = [image_spaces]
        if not image_spaces and not vector_space:
            raise ValueError('ImageAndVector requires an image_space or a vector_space')
        self._image_spaces = image_spaces or []
        self._vector_space = vector_space
        self._agent_name = agent_name
        self.full_visibility_mask = full_visibility_mask
        if self.full_visibility_mask is None:
            self.full_visibility_state = False
        else:
            self.full_visibility_state = True

    def get_policy_network_state(self, state):
        if not self.full_visibility_state:
            return state
        else:
            full_visibility_state_list = []
            for s, fv in zip(state, self.full_visibility_mask):
                if fv == FullVisibilityTypes.PolicyFunctionState or fv == FullVisibilityTypes.Both:
                    full_visibility_state_list.append(s)
            return full_visibility_state_list

    def get_value_function_state(self, state):
        if not self.full_visibility_state:
            return state
        else:
            full_visibility_state_list = []
            for s, fv in zip(state, self.full_visibility_mask):
                if fv == FullVisibilityTypes.ValueFunctionState or fv == FullVisibilityTypes.Both:
                    full_visibility_state_list.append(s)
            return full_visibility_state_list

    @property
    def image_spaces(self) -> List[spaces.Box]:
        return self._image_spaces

    @property
    def vector_space(self) -> Optional[spaces.Box]:
        return self._vector_space

    @property
    def image_shapes(self):
        return [s.shape for s in self._image_spaces]

    @property
    def vector_shape(self):
        return self._vector_space.shape

    @property
    def agent_name(self):
        return self._agent_name

    def __str__(self):
        return '''ImageSpaces={}
        VectorSpace={}
        '''.format(self.image_spaces, self.vector_space)


def deep_agent_observation(image: np.ndarray = None, vector: np.ndarray = None):
    """
    Returns a list with the image first and the vector second. If either is None, they will not be included.
    :param image: The image observation.
    :param vector: The vector observation.
    :return:
    """
    if vector is None and image is None:
        raise ValueError(f'Vector and image cannot both be None.')

    if vector is None:
        return [image]

    if image is None:
        return [vector]

    return [image, vector]


class SpaceError(ValueError):
    pass
