from enum import Enum
from gym import spaces
import numpy as np
import json
from deepagent.envs.spaces import DeepAgentSpace

class EnvType(Enum):
    ride_editor = 'ride_editor'
    unity_editor = 'unity_editor'
    unity_fsm_imitation_editor = 'unity_fsm_imitation_editor'
    remote_unity_editor = 'remote_unity_editor'
    unity_binary = 'unity_binary'
    unity_fsm_imitation_binary = 'unity_fsm_imitation_binary'
    unity_fsm_imitation_fsm_trajectory_binary = 'unity_fsm_imitation_fsm_trajectory_binary'
    sorter_fsm_imitation_fsm_trajectory_binary = 'sorter_fsm_imitation_fsm_trajectory_binary'
    ride_binary = 'ride_binary'
    sorter_binary = 'sorter_binary'
    remote_unity_binary = 'remote_unity_binary'
    unity_mock = 'unity_mock'
    offset_mock = 'offset_mock'
    atari = 'atari'
    atari_rnd = 'atari_rnd'
    remote_atari = 'remote_atari'
    remote_atari_rnd = 'remote_atari_rnd'
    offset_square = 'offset_square'
    offset_square_tenth = 'offset_square_tenth'
    offset_rf = 'offset_rf'
    deepcounter_python = 'deepcounter_python'
    offset_fx2 = 'offset_fx2'
    offset_fx2_free = 'offset_fx2_free'
    adf_self_play = "adf_self_play"
    sc2_gym = 'starcraft_ii_gym'
    sc2 = 'starcraft_ii'
    remote_sc2 = 'remote_starcraft_ii'
    python_sc2 = 'python_starcraft_ii'
    python_sc2_wrapper = 'python_starcraft_ii_wrapper'
    thunderdome_sc2 = 'thunderdome_starcraft_ii'
    racer = 'racer'
    isaac = 'isaac'

    def is_atari(self):
        return self.value in [EnvType.atari.value, EnvType.atari_rnd.value]

    def is_unity_ml(self):
        return self.value in [EnvType.ride_editor.value, EnvType.ride_binary.value, EnvType.unity_editor.value, EnvType.unity_binary.value,EnvType.unity_fsm_imitation_editor.value, EnvType.unity_fsm_imitation_binary.value, EnvType.remote_unity_editor.value]

def deserialize(d, convert=True):
    # if "__enum__" in d:
    #     name, member = d["__enum__"].split(".")
    #     return getattr(PUBLIC_ENUMS[name], member)
    # else:
    try:
        if convert:
            return EnvType[d]
        else:
            raise Exception("crud")
    except:
        # print(d)
        if isinstance(d, str):
            spaceSplit = d.split("__", 1)
            # print("spacesplit", spaceSplit)
            if spaceSplit[0] == spaces.Discrete.__name__:
                return spaces.Discrete(int(spaceSplit[1]))
            elif spaceSplit[0] == spaces.Box.__name__:
                l = eval(spaceSplit[1])
                return spaces.Box(l[0], l[1], l[2], np.dtype(l[3]))
            elif spaceSplit[0] == "tuple":
                # print("tuple found")
                l = eval(spaceSplit[1])
                return tuple(deserialize(i, convert) for i in l)
            elif spaceSplit[0] == "ndarray":
                # return np.loads(spaceSplit[1])
                l = eval(spaceSplit[1])
                return np.ndarray(l)
            elif spaceSplit[0] == "DeepAgentSpace":
                spces = eval(spaceSplit[1])
                image_space = deserialize(spces[0], convert)
                vector_space = deserialize(spces[1], convert)
                return DeepAgentSpace(image_space, vector_space)
            elif spaceSplit[0] == "np.int32":
                return np.int32(eval(spaceSplit[1]))
            elif spaceSplit[0] == "np.int64":
                return np.int64(eval(spaceSplit[1]))
        elif isinstance(d, dict):
            return {deserialize(k, convert) : deserialize(v, convert) for k, v in d.items()}
        elif isinstance(d, list):
            return [deserialize(l, convert) for l in d]
        elif isinstance(d, float):
            return np.float32(d)
        elif isinstance(d, bool):
            return np.bool_(d)
        elif isinstance(d, bytes):
            return np.loads(d)
        return d

def serialize(obj):
    if isinstance(obj, DeepAgentSpace):
        ret = obj.__class__.__name__ + "__"
        return ret + str([serialize(obj.image_spaces), serialize(obj.vector_space)])
    if isinstance(obj, list):
        return [serialize(i) for i in obj]
    if isinstance(obj, tuple):
        ret = obj.__class__.__name__ + "__"
        tup_tmp = [serialize(i) for i in obj]
        return ret + str(tuple(tup_tmp))
    if isinstance(obj, np.ndarray):
        return serialize(obj.tolist())
        # return obj.dumps()
    if isinstance(obj, EnvType):
        return obj.name
    if isinstance(obj, spaces.Discrete):
        return obj.__class__.__name__ + "__" + str(obj.n)
    if isinstance(obj, spaces.Box):
        return obj.__class__.__name__ + "__" + \
               str([np.min(obj.low),
               np.max(obj.high),
               obj.shape,
               obj.dtype.str])
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.int32):
        ret = "np.int32__"
        return ret + str(obj)
    if isinstance(obj, np.int64):
        ret = "np.int64__"
        return ret + str(obj)
    if isinstance(obj, dict):
        return {serialize(k) : serialize(v) for k, v in obj.items()}
    return obj

# class EnvTypeEncoder(json.JSONEncoder):
#     def default(self, obj):
#         print(obj)
#         if type(obj) in PUBLIC_ENUMS.values():
#             return {"__enum__": str(obj)}
#         return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    # tmp = {
    #     'observations': {((0, 1), EnvType.atari): np.ndarray(shape=(3,2,1), dtype=np.uint8)},
    #     'rewards': {(0, 1): 0.0},
    #     'terminals': {(0, 1): False},
    #     'masks': {(0, 1): [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
    #     'env_terminals': {(0, 1): False}
    #     }

    tmp = {EnvType.atari: np.array([[0., 0., 1., 0., 0., 0.]], dtype=np.float32)}

    # print(convert_keys(tmp))
    js = json.dumps(serialize(tmp))
    print("json", js)
    print("")
    extracted = serialize(json.loads(js))
    print("extracted", extracted)
