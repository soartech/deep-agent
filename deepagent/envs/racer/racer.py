import copy
import random
from collections import deque
from typing import List, Tuple
import os
import sys
import math

import numpy as np
import pygame as pg
from gym import spaces

from deepagent.envs.data import DeepAgentSpaces, EnvReturn, UnitActions
from deepagent.envs.deepagent_env import AbstractDeepAgentDictionaryEnv
from deepagent.envs.racer.teams import Team, Teams
from deepagent.envs.racer.units import Unit, RacerUgv, TeamNumber
from deepagent.envs.racer.speed_maps import make_initial_map, get_altered_map, make_initial_satellite_map, get_altered_satellite_map
from deepagent.envs.racer.shapes import Point, PlayableArea
from deepagent.envs.racer.terrain import Terrain, GoalArea
from deepagent.envs.spaces import DeepAgentSpace, FullVisibilityTypes
from deepagent.experiments.params import params
from deepagent.envs.racer.utils import Camera, BLACK

SEED_OVERRIDE = 5

class RacerEnv(AbstractDeepAgentDictionaryEnv):
    def __init__(self, random_seed=1234):
        self.terrain = None
        self.slowest = 2.0
        self.fastest = 31.0

        np.random.seed(SEED_OVERRIDE)
        random.seed(SEED_OVERRIDE)
        map_gen_config = params.Racer.map_gen_config
        self.playable_area = PlayableArea(t=0, l=0, b=map_gen_config.y, r=map_gen_config.x)
        # self.goal_areas = [GoalArea(location=pg.Vector2(random.randint(0, self.playable_area.r - 11),
        #                                                 random.randint(0, self.playable_area.b - 11)),
        #                             size=pg.Vector2(random.randint(5, 10), random.randint(5, 10))) for _ in range(1)]
        self.goal_areas = [GoalArea(location=pg.Vector2(2, 2), size=pg.Vector2(2, 2)) for _ in range(1)]
        ga = self.goal_areas[0]
        goal_rect = [int(ga.location.x), int(ga.location.y), int(ga.location.x + ga.size.x), int(ga.location.y + ga.size.y)]

        if params.Racer.real_a_priori_map:
            self.goal_areas = [GoalArea(location=pg.Vector2(44, 2), size=pg.Vector2(6, 6)) for _ in range(1)]
            ga = self.goal_areas[0]
            goal_rect = [int(ga.location.x), int(ga.location.y), int(ga.location.x + ga.size.x), int(ga.location.y + ga.size.y)]
            self.base_map = make_initial_satellite_map(h=map_gen_config.y, w=map_gen_config.x, map_octaves=params.Racer.map_octaves, patch_octaves=params.Racer.patch_octaves,
                                                       nspeeds=params.Racer.nspeeds, slowest=self.slowest, fastest=self.fastest, max_delta=params.Racer.max_delta,
                                                       wall_octaves=params.Racer.wall_octaves, nadded_walls=params.Racer.nadded_walls, ndeleted_walls=params.Racer.ndeleted_walls,
                                                       goal_rect=goal_rect)
        else:
            self.base_map = make_initial_map(h=map_gen_config.y, w=map_gen_config.x, map_octaves=params.Racer.map_octaves, patch_octaves=params.Racer.patch_octaves,
                                             nspeeds=params.Racer.nspeeds, slowest=self.slowest, fastest=self.fastest, max_delta=params.Racer.max_delta,
                                             wall_octaves=params.Racer.wall_octaves, nadded_walls=params.Racer.nadded_walls, ndeleted_walls=params.Racer.ndeleted_walls,
                                             goal_rect=goal_rect)

        if params.Racer.randomize_maps:
            random.seed(random_seed)
            np.random.seed(random_seed)
        else:
            random.seed(SEED_OVERRIDE)
            np.random.seed(SEED_OVERRIDE)
        if params.RuntimeParams.is_testing():
            random.seed(1241565)
            np.random.seed(1241565)
        self.spawn_random = random.Random(random_seed)

        self.num_frames = params.Racer.frame_stack
        self.max_episode_steps = 250
        self.distance_normalization = 50.0

        self.total_steps = 0
        self.episodes = 0
        self._obs_value_names = None

        team1 = Team(ally_unit_names=['racerugv'],
                     team_number=TeamNumber.ONE,
                     agent_controlled=True)
        team2 = Team(ally_unit_names=[],
                     team_number=TeamNumber.TWO,
                     agent_controlled=False)

        self.teams = Teams(team1, team2)

        self.team1_win_rankings = [(0, 1, unit_type) for unit_type in team1.ally_unit_type_strings] + [(1, 2, unit_type) for unit_type in team2.ally_unit_type_strings]
        self.team2_win_rankings = [(0, 2, unit_type) for unit_type in team2.ally_unit_type_strings] + [(1, 1, unit_type) for unit_type in team1.ally_unit_type_strings]
        self.draw_win_rankings = [(0, 1, unit_type) for unit_type in team1.ally_unit_type_strings] + [(0, 2, unit_type) for unit_type in team2.ally_unit_type_strings]

        if not params.EnvironmentParams.env_render:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'

        # pygame params
        pg.init()
        pg.mixer.quit()
        width, height = 1027, 768
        self.screen = pg.display.set_mode((width, height), pg.RESIZABLE)
        pg.display.set_caption('Racer')

        # Create The Background
        self.background = pg.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill(BLACK)

        # Calculate pixels per unit so the entire map fits within the screen viewport to start
        max_pixel_width = self.screen.get_width() // self.playable_area.width
        max_pixel_height = self.screen.get_height() // self.playable_area.height
        pixels_per_game_unit = max(min(max_pixel_width, max_pixel_height) - 1, 1)
        self.camera = Camera(screen=self.screen, playable_area=self.playable_area,
                             pixels_per_game_unit=pixels_per_game_unit)

        self.background_sprites = pg.sprite.Group()

        self._init_map_sprites()
        self._reset()

    def _reset(self, new_map=False):
        self.episodes += 1
        if new_map:
            self._init_map_sprites()
        self.episode_steps = 0
        self.obs_history = deque(maxlen=self.num_frames)
        self.foreground_sprites = pg.sprite.Group()
        self._spawn_units()
        # terrain reset must be called AFTER spawn units, since it updates vision based on unit position
        self.terrain.reset(self.teams.get_team(TeamNumber.ONE).units[0])
        self.reset_default_masks()

    def reset_default_masks(self):
        masks = dict()
        for team in self.teams.agent_controlled():
            for u in team.units:
                unit_type = team.get_unit_type_string(u.__class__.__name__)
                masks[u.tag] = np.ones((self.action_space[unit_type].vector_space.shape[0],), dtype=np.float32)
        self.default_masks = masks

    def get_random_point(self, center: Tuple[int, int], size: Tuple[int, int], edge_buffer: int = 5):
        """
        @param center: (x, y)
        @param size: (x-range, by y-range)
        @param edge_buffer: Buffer between random point and edge of map
        @return: A random point (x, y) within the region defined by center and size, guaranteed to be edge_buffer
            distance inside the edge of the map.
        """
        x, y = size[0], size[1]
        half_y, half_x = y / 2.0, x / 2.0

        buffer = float(edge_buffer)
        miny, minx = edge_buffer, edge_buffer
        maxy, maxx = self.playable_area.b - edge_buffer, self.playable_area.r - edge_buffer

        min_selection_x, max_selection_x = center[0] - half_x, center[0] + half_x
        min_selection_y, max_selection_y = center[1] - half_y, center[1] + half_y

        return self.spawn_random.uniform(max(min_selection_x, minx), min(max_selection_x, maxx)), \
               self.spawn_random.uniform(max(min_selection_y, miny), min(max_selection_y, maxy))

    def _get_spawn_location(self, target_x, target_y):
        ux, uy = target_x + self.spawn_random.uniform(-5.0, 5.0), target_y + self.spawn_random.uniform(-5.0, 5.0)

        ux = round(ux)
        uy = round(uy)

        # Snap to map if off map
        if ux < 0:
            ux = 0
        r = self.playable_area.r
        if ux > r:
            ux = r

        if uy < 0:
            uy = 0
        b = self.playable_area.b
        if uy > b:
            uy = b

        # Search playable area indices in outward rings around the unit to find the nearest playable index
        #max_diff = int(max(abs(r - ux), abs(b - uy)))
        max_diff = 20
        for n in range(max_diff):
            x = ux - n
            for y in range(max(-n, 0), min(n, b)):
                if self.terrain.traversable(Point(x, uy + y)):
                    return x, uy + y

            y = uy + n
            for x in range(max(-n, 0), min(n, r)):
                if self.terrain.traversable(Point(ux + x, y)):
                    return ux + x, y

            x = ux + n
            for y in range(max(-n, 0), min(n + 1, b)):
                if self.terrain.traversable(Point(x, uy + y)):
                    return x, uy + y

            y = uy - n
            for x in range(max(-n + 1, 0), min(n, r)):
                if self.terrain.traversable(Point(ux + x, y)):
                    return ux + x, y

        raise Exception('Failed to spawn unit')

    def _spawn_units(self):
        id = 0

        if params.RuntimeParams.is_testing():
            # don't spawn near goal areas
            near_goal = True
            centroid = self.goal_areas[0].centroid
            goal_x = centroid.x
            goal_y = centroid.y
            while near_goal:
                x, y = self.get_random_point(self.playable_area.center, [self.playable_area.width, self.playable_area.height])

                dx = goal_x - x
                dy = goal_y - y

                near_goal = (dx*dx + dy*dy < 400)

        else:
            x, y = self.get_random_point(self.playable_area.center, [self.playable_area.width, self.playable_area.height])

        try:
            ux, uy = self._get_spawn_location(x, y)
            scv = RacerUgv(id, ux, uy, TeamNumber.ONE, self.terrain, self.camera, [self.foreground_sprites])
            self._post_spawn(scv)
            id += 1
        except Exception:
            print('failed to spawn unit')

    def _post_spawn(self, unit):
        #print('Starting location: ', unit.x, unit.y)
        self.teams.get_team(unit.team_number).add_unit(unit)

    def _init_map_sprites(self):
        if params.Racer.real_a_priori_map:
            map_data = get_altered_satellite_map(self.base_map)
        else:
            map_data = get_altered_map(self.base_map)

        if self.terrain is None:
            self.terrain = Terrain(map_data=map_data,
                                   origin_offset=self.playable_area.origin_offset,
                                   goal_areas=self.goal_areas,
                                   render_groups=[self.background_sprites],
                                   camera=self.camera,
                                   slowest=self.slowest,
                                   fastest=self.fastest)
        else:
            self.terrain.new_map(map_data, self.goal_areas)

        self.foreground_sprites = None
        self._map_with_unit_obs = None

        # goal info
        self.num_goals = 1
        self.num_visible_goals = 1

    # compute the best dijkstra action based on currently visible map
    def get_best_action(self):
        unit = self.teams.get_team(TeamNumber.ONE).units[0]
        return self.terrain.get_best_action(unit)

    def step(self, actions: UnitActions) -> EnvReturn:
        self.episode_steps += 1
        self.total_steps += 1

        # reset every episodes per map steps when training
        if params.RuntimeParams.is_training() and self.total_steps % params.Racer.map_gen_config.episodes_per_map == 0:
            return self.reset(new_map=True)

        # reset map after every game when testing
        if params.RuntimeParams.is_testing() and self.episode_steps > self.max_episode_steps:
            return self.reset(new_map=True)

        if self.episode_steps > self.max_episode_steps:
            return self.reset(new_map=False)

        rewards = self.create_empty_rewards()

        for key, action in actions.items():
            unit_id = key[0]
            action_id = np.argmax(action)
            unit = self.teams.get_unit(unit_id)
            reward = unit.step(action_id)

            rewards[unit_id] += reward

        masks, states, terminal, terminals, end_of_game_ranking = self.post_step(rewards)

        return states, rewards, terminals, masks, end_of_game_ranking if terminal else terminal

    def post_step(self, rewards):
        team1 = self.teams.get_team(TeamNumber.ONE)

        terminal = False
        end_of_game_ranking = None

        # create states dict
        states = self.create_states()
        unit_ids = [id for id, _ in states.keys()]
        masks = dict()
        for id, mask in self.create_default_masks().items():
            if id in unit_ids:
                masks[id] = mask
                # unit = self.teams.get_unit(id)
                # masks[id] = self.terrain.create_action_mask(unit, masks[id])
                # if np.sum(masks[id]) == 0.0:
                #     self._reset(new_map=False)
        terminals = self.create_terminals(rewards)

        # reset if an scv reached a goal
        if not terminal:
            reached_goal = False
            for u in team1.units:
                if self.terrain.in_goal(u):
                    reached_goal = True
            if reached_goal:
                for u in team1.units:
                    rewards[u.tag] += 10.0
                terminal = True
                end_of_game_ranking = self.team1_win_rankings
                terminals = self.set_all_terminal(terminals)
                if params.RuntimeParams.is_training():
                    self._reset(new_map=False)
                else:
                    # reset map at the end of a game when testing
                    self._reset(new_map=True)
        return masks, states, terminal, terminals, end_of_game_ranking

    def set_all_terminal(self, terminals):
        for k in terminals.keys():
            terminals[k] = True
        return terminals

    def create_states(self):
        # create states dict
        states = dict()

        self.obs_history.append(copy.deepcopy(self.teams))

        for team in self.teams.agent_controlled():
            for u in team.units:
                unit_type = team.get_unit_type_string(u.__class__.__name__)
                key = (u.tag, unit_type)

                map_state = self.terrain.get_state()
                ground_truth_map_state = self.terrain.get_ground_truth_state()
                if params.Racer.include_vector_state:
                    vector_state = self.terrain.get_heuristic_vector()
                    states[key] = [ground_truth_map_state, map_state, vector_state]
                else:
                    states[key] = [ground_truth_map_state, map_state]

        return states

    def reset(self, new_map=False) -> EnvReturn:
        s, r, t, m, tt = self.create_states(), self.create_empty_rewards(), self.create_all_terminal(), self.create_default_masks(), True
        self._reset(new_map=new_map)
        return s, r, t, m, self.draw_win_rankings

    def create_default_masks(self):
        if self.default_masks:
            return self.default_masks
        else:
            self.reset_default_masks()
        return self.default_masks

    def reset_default_masks(self):
        masks = dict()
        for team in self.teams.agent_controlled():
            for u in team.units:
                unit_type = team.get_unit_type_string(u.__class__.__name__)
                masks[u.tag] = np.ones((self.action_space[unit_type].vector_space.shape[0],), dtype=np.float32)
        self.default_masks = masks

    def create_empty_rewards(self):
        rewards = dict()
        for team in self.teams:
            for u in team.units:
                rewards[u.tag] = 0.0
        return rewards

    def create_terminals(self, rewards):
        terminals = dict()
        for team in self.teams:
            for u in team.units:
                if u.is_dead:
                    terminals[u.tag] = True
                    rewards[u.tag] -= 1.0
                else:
                    terminals[u.tag] = False
        return terminals

    def create_all_terminal(self):
        terminals = dict()
        for team in self.teams:
            for u in team.units:
                terminals[u.tag] = True
        return terminals

    @property
    def observation_space(self) -> DeepAgentSpaces:
        observation_space = {}
        image_space = spaces.Box(low=-1, high=1, shape=self.terrain.get_state().shape, dtype=np.float32)
        if params.Racer.include_vector_state:
            vector_space = spaces.Box(low=-1, high=1, shape=self.terrain.get_heuristic_vector().shape, dtype=np.float32)
            space = DeepAgentSpace(image_spaces=image_space, vector_space=vector_space,
                                   full_visibility_mask=[FullVisibilityTypes.ValueFunctionState, FullVisibilityTypes.PolicyFunctionState, FullVisibilityTypes.Both])
        else:
            vector_space = None
            space = DeepAgentSpace(image_spaces=image_space, vector_space=vector_space,
                                   full_visibility_mask=[FullVisibilityTypes.ValueFunctionState, FullVisibilityTypes.PolicyFunctionState])

        for team in self.teams.agent_controlled():
            for cls in team.unit_classes:
                unit_type = team.get_unit_type_string(cls.__name__)
                observation_space[unit_type] = space

        return observation_space

    @property
    def obs_value_names(self):
        if self._obs_value_names:
            return self._obs_value_names
        else:
            obs_value_names_dict = {}
            for team in self.teams.agent_controlled():
                for unit_type in team.ally_unit_type_strings:
                    obs_value_names = []

                    # graph info
                    for goal_idx in range(self.num_visible_goals):
                        obs_value_names.append(f'goal_{goal_idx}_unit_x')
                        obs_value_names.append(f'goal_{goal_idx}_unit_y')

                        obs_value_names.append(f'goal_{goal_idx}_dist')

                        obs_value_names.append(f'goal_{goal_idx}_ul_x')
                        obs_value_names.append(f'goal_{goal_idx}_ul_y')

                        obs_value_names.append(f'goal_{goal_idx}_ur_x')
                        obs_value_names.append(f'goal_{goal_idx}_ur_y')

                        obs_value_names.append(f'goal_{goal_idx}_lr_x')
                        obs_value_names.append(f'goal_{goal_idx}_lr_y')

                        obs_value_names.append(f'goal_{goal_idx}_ll_x')
                        obs_value_names.append(f'goal_{goal_idx}_ll_y')

                        obs_value_names.append(f'goal_{goal_idx}_x')
                        obs_value_names.append(f'goal_{goal_idx}_y')

                    print(self.action_space.keys())

                    if self.action_space[unit_type].image_spaces:
                        raise ValueError('image_space not supported by this environment')
                    for i in range(self.action_space[unit_type].vector_space.shape[0]):
                        obs_value_names.append(f'action_{i}')
                    obs_value_names_dict[unit_type] = obs_value_names
            self._obs_value_names = obs_value_names_dict
            return self._obs_value_names

    @property
    def action_space(self) -> DeepAgentSpaces:
        action_space = {}

        for team in self.teams.agent_controlled():
            for cls in team.unit_classes:
                for u in team.units:
                    if isinstance(u, cls):
                        action_space[team.get_unit_type_string(cls.__name__)] = u.action_space()
                        break

        return action_space

    @property
    def use_terminals_as_start_states(self) -> bool:
        False

    def close(self):
        print('Num games: ', len(self.terrain.dijkstra_times))
        average_time = sum(self.terrain.dijkstra_times) / len(self.terrain.dijkstra_times)
        print('Average dijkstra time: ', average_time)
        average_time = sum(self.terrain.agent_times) / len(self.terrain.agent_times)
        print('Average agent time: ', average_time)
        print('Incomplete maps: ', self.terrain.incomplete_maps)

    @property
    def obs_index(self):
        self.obs_i += 1
        return self.obs_i

    def render(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                print('pygame QUIT event received.')
                sys.exit()
            elif event.type == pg.VIDEORESIZE:
                self.camera.on_screen_resize(self.screen)
                self.background = pg.transform.scale(self.background, self.screen.get_size())
            elif event.type == pg.MOUSEBUTTONDOWN:
                zoom = 1 if event.button == 4 else -1 if event.button == 5 else 0
                if zoom != 0:
                    self.camera.zoom(zoom)

        keystate = pg.key.get_pressed()
        camera_dir = pg.Vector2(keystate[pg.K_RIGHT] - keystate[pg.K_LEFT], keystate[pg.K_DOWN] - keystate[pg.K_UP])
        self.camera.move(camera_dir)

        self.background_sprites.update()
        self.foreground_sprites.update()
        self.screen.blit(self.background, (0, 0))
        self.background_sprites.draw(self.screen)
        self.foreground_sprites.draw(self.screen)
        pg.display.flip()
