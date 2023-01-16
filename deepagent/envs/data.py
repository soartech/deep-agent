from __future__ import annotations

from typing import NamedTuple, Dict, List, Tuple

import numpy as np

from deepagent.envs.spaces import DeepAgentSpace

AgentName = str
DeepAgentSpaces = Dict[AgentName, DeepAgentSpace]

# Start section: Current Convoluted Data Structures Used in API
# TODO: These are the keys to action dictionaries envs are currently usingg, this is pretty convoluted and should
# be using something more like the DeepAgentStep named tuple below

EnvId = int
UnitId = str
UnitType = str
UnitKey = Tuple[EnvId, UnitId]
UnitKeyWithType = Tuple[UnitKey, UnitType]
UnitActions = Dict[UnitKeyWithType, np.ndarray]
UnitStates = Dict[UnitKeyWithType, List[np.ndarray]]
UnitRewards = Dict[UnitKey, float]
UnitTerminals = Dict[UnitKey, bool]
UnitMasks = Dict[
    UnitKey, np.ndarray]  # Some environments are using bool arrays where True means mask, and others are using int arrays where 0 means mask
EnvTerminals = Dict[int, bool]  # This may be completely broken/unused currently
EnvReturn = Tuple[UnitStates, UnitRewards, UnitTerminals, UnitMasks, EnvTerminals]


# End section: Current Convoluted Data Structures Used in API

