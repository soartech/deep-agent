# This file is meant for developer use, for making LOCAL-ONLY changes to experiment params, especially for path values
# Changes to this file should NEVER be pushed/committed to the repo


# To use this file, treat it the same as any other experiment file
# All experiment files should have "from . import dev_experiment_param_override" at the end of the file
# Which allows any values set in this file to override the values in the experiment file

#By default, this file doesn't contain any effective code, but shows examples of common path params to override

from deepagent.data2vec.data2vec import Data2VecPlugin
from deepagent.experiments.params.params import EnvironmentParams, FSMParams, TrainingParams, TestingParams, \
    AgentParams, TrainingFunctions, NetworkInit, MaskType, ImitationLearning

# data2vec = Data2VecPlugin(load_path=r'')

# EnvironmentParams.set(
#     environment=r'',
#     feature_extractor=data2vec.get_output,
# )
#
# FSMParams.set(
#     fsm_action_path=r'',
#     fsm_event_path=r'',
#     fsm_transprob_path=r'',
#     fsm_def_path=r'',
# )

# ImitationLearning.set(
#     pre_training_args=[
#         ImitationLearning.PreTrainArgs(
#             recording_files=[r''],
#         )
#     ]
# )
