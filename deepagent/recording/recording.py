# NOTE: Do not use this module directly. This is just a placeholder.
# Whenever running an experiment, use deepagent.params.utils.load_params to load in the actual params for the experiment.


import bz2
import json
import os
import pickle
from enum import Enum
from typing import NamedTuple, List, Union, Tuple, Dict

import numpy as np
from gym.utils.play import play

from deepagent.agents.memory import BCMemory
from deepagent.common.auto_dir import get_next_numbered_directory
from deepagent.experiments.params import params
from deepagent.experiments.params.utils import load_params
from deepagent.openai_addons.rnd.rnd_atari_wrappers import make_atari, wrap_deepmind

RECORDING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'recordings'))


class StepInfo(NamedTuple):
    state: List[np.ndarray]
    prev_action: np.ndarray
    mask: np.ndarray
    reward: float
    terminal: bool


class PlayerType(str, Enum):
    human = 'human'
    weights = 'weights'
    scripted = 'scripted'


class MetaData(NamedTuple):
    player_type: PlayerType
    player_id: str
    teacher_brain: str
    student_brain: str
    weights_dir: str
    step_info_file: str
    meta_data_file: str
    total_rewards: float
    total_steps: float
    total_episodes: float
    team_id: str
    state_shape: Tuple[int, int, int] = None
    action_shape: Tuple[int] = None
    match_id: str = None
    killed: List[str] = None  # deprecated
    killed_by: List[str] = None  # deprecated

    def rewards_per_episode(self):
        return self.total_rewards / self.total_episodes

    def partial(self):
        return PartialMetaData(player_id=self.player_id, step_info_file=self.step_info_file,
                               meta_data_file=self.meta_data_file)


class MultiPlayerDemoReader:
    def __init__(self, player_runs: Dict[str, List[int]]):
        self._player_runs = player_runs

    def load_meta(self):
        for player_id, run_ids in self._player_runs.items():
            reader = DemoWriterReader(player_id=player_id)
            return reader.load_meta_data(run_ids[0])

    def bc_memory(self, just_images: bool = True, use_terminals_as_start_state: bool = True, frame_stack: int = 1,
                  num_actions: int = 5, tile_first: bool = True):
        all_memory = BCMemory()
        for player_id, run_ids in self._player_runs.items():
            reader = DemoWriterReader(player_id=player_id)
            player_memory = reader.bc_memory(run_ids=run_ids, just_images=just_images,
                                             use_terminals_as_start_state=use_terminals_as_start_state,
                                             frame_stack=frame_stack, num_actions=num_actions, tile_first=tile_first)
            all_memory.add_lists(player_memory.states, player_memory.actions)
        return all_memory


class DemoWriterReader:
    def __init__(self, player_id: str, player_type: PlayerType = None, teacher_brain: str = None,
                 student_brain: str = None,
                 match_id: str = None, weights_dir: str = None, state_shape: Tuple[int] = None,
                 action_shape: Tuple[int] = None):
        env_name = os.path.basename(params.EnvironmentParams.environment)
        env_name = os.path.splitext(env_name)[0]
        self._recording_dir = os.path.join(RECORDING_DIR, env_name, player_id)
        self._create_directory()
        self._player_id = player_id
        self._total_reward = 0
        self._total_steps = 0
        self._episodes = 0
        self._player_type = player_type
        self._teacher_brain = teacher_brain
        self._student_brain = student_brain
        self._weights_dir = weights_dir
        self.match_id = match_id
        self.team_id = None
        self._state_shape = state_shape
        self._action_shape = action_shape
        if params.RuntimeParams.is_recording() and (
                self._state_shape == None or self._action_shape == None or player_type is None):
            raise ValueError(
                f'State and action space  and player type must be provided when recording, but found state_space='
                f'{self._state_shape} and action_space={self._action_shape} and player_type={player_type}')

        demo_suffix = '.demo'
        self._demo_file = get_next_numbered_directory(parent_dir=self._recording_dir, prefix=self._player_id + '_',
                                                      suffix=demo_suffix, make_dirs=False)
        meta_suffix = '.json'
        self._meta_file = get_next_numbered_directory(parent_dir=self._recording_dir, prefix=self._player_id + '_',
                                                      suffix=meta_suffix, make_dirs=False)

        num1 = self._demo_file[-(len(demo_suffix) + 1)]
        num2 = self._meta_file[-(len(meta_suffix) + 1)]
        if num1 != num2:
            raise ValueError(f'''Demo file and meta file auto-increment is off:
                            \tNext available demo file: {self._demo_file}
                            \tNext available meta file: {self._meta_file}''')

    def _create_directory(self):
        if not os.path.exists(self.recording_dir):
            os.makedirs(self.recording_dir)

    @property
    def recording_dir(self):
        return self._recording_dir

    def write_step(self, step_info: StepInfo):
        self._total_reward += step_info.reward
        self._total_steps += 1

        if step_info.terminal:
            self._episodes += 1
        with bz2.BZ2File(self._demo_file, 'ab') as f:
            pickle.dump(step_info, f)

    def finish(self):
        meta_data = MetaData(player_type=self._player_type, player_id=self._player_id,
                             teacher_brain=self._teacher_brain, student_brain=self._student_brain,
                             weights_dir=self._weights_dir, step_info_file=self._demo_file,
                             meta_data_file=self._meta_file, total_rewards=self._total_reward,
                             total_steps=self._total_steps, total_episodes=self._episodes, team_id=self.team_id,
                             match_id=self.match_id, state_shape=self._state_shape, action_shape=self._action_shape)
        write_json(meta_data._asdict(), self._meta_file)

    def delete(self):
        os.remove(self._demo_file)

    def _get_demo_file(self, run_id):
        return os.path.join(self._recording_dir, self._player_id + '_' + str(run_id) + '.demo')

    def _get_meta_file(self, run_id):
        return os.path.join(self._recording_dir, self._player_id + '_' + str(run_id) + '.json')

    def read_step_generator(self, run_id: int) -> StepInfo:
        with bz2.BZ2File(self._get_demo_file(run_id), 'rb') as f:
            while True:
                try:
                    step_info = pickle.load(f, encoding='bytes')
                    for i, state in enumerate(step_info.state):
                        state = state.astype(np.float32)
                        if len(state.shape) > 1 and np.issubdtype(step_info.state[i].dtype, np.integer):
                            # TODO: For now we're assuming that multi-dim arrays stored as integer represent unscaled
                            # image data, so we scale it to [-1, 1] here.
                            state = state / 127.5
                            step_info.state[i] = state - np.mean(state)
                    yield step_info
                except EOFError:
                    break

    def spaces(self, run_id: int):
        meta = self.load_meta_data(run_id)
        return meta.state_shape, meta.action_shape

    def bc_memory(self, run_ids: Union[int, List[int]], just_images: bool = True,
                  use_terminals_as_start_state: bool = True, frame_stack: int = 1, tile_first: bool = True,
                  num_actions: int = 5) -> BCMemory:
        """
        :param run_ids: The run id(s) of the player to use for memory generation.
        :param just_images: TODO: This is just here because unity is randomly recording extra state sometimes besides
            the feature layers.
        :param use_terminals_as_start_state: If true, the state received on terminal states will be used both for s1 of
            that state tuple, and s0 of the next.
        :param frame_stack: The number of sequential frames to use for a state observation.
        :param tile_first: If true, the first observation is tiled to fill the frame stack, otherwise, the first
            observation is zeros for all but one frame in the stakc.
        :return: A BCMemory object for the player's demo file with the specified run_id.
        """
        if type(run_ids) == int:
            run_ids = [run_ids]
        file_gens = [self.read_step_generator(rid) for rid in run_ids]

        first_step = True

        states, actions = [], []
        for file_gen in file_gens:
            for step_info in file_gen:
                image = step_info.state[0]

                process_vectors = not just_images and len(step_info.state > 1)

                if process_vectors:
                    vector = step_info.state[1]

                if first_step:
                    if tile_first:
                        image_stack = np.tile(image, frame_stack)
                    else:
                        image_stack = np.zeros((image.shape[0], image.shape[1], image.shape[2] * frame_stack))
                        image_stack[..., -image.shape[-1]:] = image

                    if process_vectors:
                        vector = step_info.state[1]

                        if tile_first:
                            vector_stack = np.tile(vector, frame_stack)
                        else:
                            vector_stack = np.zeros((len(vector) * frame_stack,))
                            vector_stack[..., vector.shape[-1]:] = vector

                    first_step = False
                    continue

                a = step_info.prev_action

                states.extend([image_stack] if not process_vectors else [image_stack, vector_stack])
                a = int(np.squeeze(a))

                np_action = np.zeros((num_actions,), dtype=np.float32)
                np_action[a] = 1.0

                actions.append(np_action)

                # roll image stack and insert new image at end
                image_stack = np.roll(image_stack, shift=-image.shape[-1], axis=-1)
                image_stack[..., -image.shape[-1]:] = image

                if process_vectors:
                    # roll vector stack and insert new vector at end
                    vector_stack = np.roll(vector_stack, shift=-len(vector))
                    vector_stack[-len(vector):] = vector

                first_step = step_info.terminal and not use_terminals_as_start_state

        bc = BCMemory()
        bc.add_lists(states, actions)
        return bc

    def load_meta_data(self, run_id: int = None):
        filename = self._meta_file if run_id is None else self._get_meta_file(run_id)
        with open(filename, 'r') as f:
            reloaded = MetaData(**json.loads(f.read()))
        return reloaded

    def partial_meta_data(self):
        return PartialMetaData(player_id=self._player_id, step_info_file=self._demo_file,
                               meta_data_file=self._meta_file)


def write_json(dictionary, file_name):
    with open(file_name, 'w') as f:
        json_data = json.dumps(dictionary, default=lambda o: o.__dict__, indent=2, sort_keys=True)
        f.write(json_data)


class PartialMetaData(NamedTuple):
    player_id: str
    step_info_file: str
    meta_data_file: str


def _get_atari_step_callback(recorder: DemoWriterReader):
    def callback(prev_obs, obs, prev_action, rew, env_done, info):
        recorder.write_step(step_info=StepInfo(state=[obs], prev_action=prev_action, mask=None,
                                               reward=rew,
                                               terminal=env_done))
        pass

    return callback


def record_atari(player_id):
    env = make_atari(params.EnvironmentParams.environment)
    env = wrap_deepmind(env)
    # print(f'state_space={env.observation_space.shape} action_space={dir(env.action_space)}')
    state_space = env.observation_space.shape
    action_space = (env.action_space.n,)
    demo_write_read = DemoWriterReader(player_id=player_id, player_type=PlayerType.human, teacher_brain='atari',
                                       student_brain='atari', state_shape=state_space, action_shape=action_space)

    play(env, fps=24, zoom=4.0, callback=_get_atari_step_callback(recorder=demo_write_read))

    demo_write_read.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--player_id', '-pid', help="The id of the player demonstration.", required=True)
    parser.add_argument('--params', '-p', help="The params to use: ex: 'params1', 'params2', etc.", required=True)
    parser.add_argument('--run_ids', '-rids', type=int, default=[1], nargs="*")
    args = parser.parse_args()

    load_params(name=args.params)

    reader_writer = DemoWriterReader(player_id=args.player_id, player_type=PlayerType.human)
    generator = reader_writer.read_step_generator(args.run_ids[0])
    for i, step in enumerate(generator):
        if i % 100 == 0:
            print(
                f'step {i}: state={step.state} pa={step.prev_action} m={step.mask} t={step.terminal} r={step.reward}')
        else:
            print(f'step {i}: state_shape={[s.shape for s in step.state]} pa={step.prev_action} m={step.mask} '
                  f't={step.terminal} r={step.reward}')
    print(reader_writer.load_meta_data(args.run_ids[0]))

    env_type = params.EnvironmentParams.env_type
    if env_type.is_atari():
        frame_stack = 4
    elif env_type.is_unity_ml():
        frame_stack = params.UnityParams.unity_frame_stack
    else:
        raise ValueError(f"Don't know how to frame stack for env={env_type}")

    i = 0
    bc_memory = reader_writer.bc_memory(args.run_ids, just_images=True, frame_stack=frame_stack,
                                        use_terminals_as_start_state=params.EnvironmentParams.env_type.is_atari(),
                                        tile_first=params.EnvironmentParams.env_type.is_unity_ml(),
                                        num_actions=5)
                                        # num_actions=18)
    # for state, action in bc_memory.batch_generator(32):
    #     i += 1
    #     if i > 100: break
    #     print(f's0={state.shape} a={action.shape}')
    train, num_train, validate, num_validate, test, num_test = bc_memory.batch_generator_splits()
    # print(f'num_train={num_train} num_validate={num_validate} num_test={num_test}')
    for x, y in train:
        state = x
        import scipy.misc

        num_batch_indexes = state.shape[0]
        num_layers = state.shape[-1]
        print(f'nbi={num_batch_indexes} nl={num_layers}')
        for batch_idx in range(num_batch_indexes):
            for i in range(num_layers):
                tmp = state[batch_idx, :, :, :]
                print('calling scipy')
                scipy.misc.toimage(tmp[:, :, i], cmax=1.0, cmin=-1.0).save(
                    'BatchIdx-{}_LayerIndex-{}.png'.format(batch_idx, i))
            if batch_idx == 2: break
        break
        # print(f'x={x} y={y}')
        # break

    # for x, y in validate:
    #     print(f'x={x} y={y}')
    #     break
    #
    # for x, y in test:
    #     print(f'x={x} y={y}')
    #     break
    #
    # print(f'num_train={num_train} num_validate={num_validate} num_test={num_test}')
