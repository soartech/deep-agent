from typing import List, Tuple, Type, Union

import pygame as pg
from gym import spaces
from enum import Enum

from deepagent.envs.racer.terrain import Terrain
from deepagent.envs.racer.utils import RED, BLUE, Camera, GREEN, BLACK, ZoomListener, MoveListener, WHITE, YELLOW
from deepagent.envs.spaces import DeepAgentSpace
from deepagent.envs.racer.shapes import Point


class TeamNumber(Enum):
    ONE = 1
    TWO = 2

    def color(self):
        if self == TeamNumber.ONE:
            return pg.Color(BLUE)
        return pg.Color(RED)

    def enemy(self):
        return TeamNumber.ONE if self == TeamNumber.TWO else TeamNumber.TWO


class Unit:
    flying_height = 500
    state_attributes = ['id', 'x', 'y', 'size', 'unit_type', 'flying', 'speed', 'health', 'vision_range', 'team', 'time_per_sim_step']

    def __init__(self, tag, x, y, size, unit_type, speed, health, vision_range, team_number: TeamNumber, terrain: Terrain,
                 flying=False, time_per_sim_step=1.0):
        self.team_number = team_number

        self.tag = tag
        self.x = x
        self.y = y

        self.health = health
        self.max_health = self.health
        self.speed = speed
        self.vision_range = vision_range
        self.time_per_sim_step = time_per_sim_step
        self.unit_type = unit_type
        self.flying = flying
        self.size = size
        self.terrain = terrain
        self._visible = True

        self.num_goals = 1

        self.goal_idx_order = [num for num in range(self.num_goals)]
        self.goal_positions = [None for _ in range(self.num_goals)]

    def __deepcopy__(self, memo):
        """
        Image and other pygame classes are not serializable. We deepcopy everything except for the sprite related
        attributes.
        """
        copy = Unit(self.tag, self.x, self.y, self.size, self.unit_type, self.speed, self.health, self.vision_range, self.team_number, terrain=None, camera=None,
                    flying=self.flying, time_per_sim_step=self.time_per_sim_step, render_groups=None)
        copy._visible = self._visible
        return copy

    def action_space(self):
        # move n, s, e, w
        vector_space = spaces.Box(0.0, 1.0, shape=(8,))
        # UnityML does not support anything besides 1-d vectors for action spaces out of the box
        space = DeepAgentSpace(vector_space=vector_space)

        return space

    def step(self, action_id: int):
        self.update_state()
        reward = self.execute_action(action_id)
        return reward

    def update_state(self):
        pass

    @property
    def health_ratio(self):
        return self.health / self.max_health

    def update_lidar(self):
        self.lidar = self.terrain.lidar(Point(self.x, self.y))

    def action_logic(self, action_id: int):
        if action_id == 0:
            return self.move_n()
        elif action_id == 1:
            return self.move_s()
        elif action_id == 2:
            return self.move_w()
        elif action_id == 3:
            return self.move_e()
        elif action_id == 4:
            return self.move_nw()
        elif action_id == 5:
            return self.move_ne()
        elif action_id == 6:
            return self.move_sw()
        elif action_id == 7:
            return self.move_se()
        else:
            print('Error, Unit unrecognized action id', action_id)
        return 0.0

    def move_n(self):
        return self.terrain.move_unit_one_cell(self, True, None)

    def move_s(self):
        return self.terrain.move_unit_one_cell(self, False, None)

    def move_w(self):
        return self.terrain.move_unit_one_cell(self, None, True)

    def move_e(self):
        return self.terrain.move_unit_one_cell(self, None, False)

    def move_nw(self):
        return self.terrain.move_unit_one_cell(self, True, True)

    def move_ne(self):
        return self.terrain.move_unit_one_cell(self, True, False)

    def move_sw(self):
        return self.terrain.move_unit_one_cell(self, False, True)

    def move_se(self):
        return self.terrain.move_unit_one_cell(self, False, False)

    @property
    def is_dead(self):
        return self.health <= 0

    @property
    def location(self) -> Point:
        return Point(self.x, self.y)

    def squared_distance_to(self, other_unit: 'Unit'):
        return self.location.squared_distance_to(other_unit.location)

    def is_near_unit(self, units, dist2=100.0):
        for unit in units:
            if self.squared_distance_to(unit) < dist2:
                return True
        return False

    def can_see(self, unit: 'Unit'):
        if not self.in_vision_range(unit):
            return False

        if self.higher_than(unit):
            return True

        if not unit.flying and self.lower_than(unit):
            return False

        return self.check_line_of_sight(unit)

    def lower_than(self, unit):
        return self.height <= unit.height - Terrain.half_height_difference

    def higher_than(self, unit):
        return self.height >= unit.height + Terrain.half_height_difference

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, visible: bool):
        if visible == self._visible:
            return

        self._visible = visible
        self._set_alpha()

    def _set_alpha(self):
        if self.image is not None:
            self.image.set_alpha(255 if self.visible else 63)

    @property
    def height(self):
        if self.flying:
            return Unit.flying_height
        return self.terrain.get_height(self.location)

    def check_line_of_sight(self, unit: Union['Unit', Point]):
        """
        Check's Bresenham's line algorithm line of sight check. Checks the unit cells on a line from this unit to another
        unit to see if any walls are in the way. This check assumes that both units are not flying units and are at
        the same height level. Note: This algorithm is not necessarily symmetrical. Unit x can have line of sight
        on unit y even though unit y doesn't have line of sight on unit x due to the way Bresenham's algorithm
        picks cells from point x to point y versus point y to point x.
        """

        x0, y0 = int(self.x), int(self.y)
        x1, y1 = int(unit.x), int(unit.y)

        dx = x1 - x0
        dy = y1 - y0

        xsign = 1 if dx > 0 else -1
        ysign = 1 if dy > 0 else -1

        dx = abs(dx)
        dy = abs(dy)

        if dx > dy:
            xx, xy, yx, yy = xsign, 0, 0, ysign
        else:
            dx, dy = dy, dx
            xx, xy, yx, yy = 0, ysign, xsign, 0

        D = 2 * dy - dx
        y = 0

        points = []
        for x in range(dx + 1):
            point = Point(x0 + x * xx + y * yx, y0 + x * xy + y * yy)
            if not self.terrain.traversable(point):
                return False
            points.append(point)
            if D >= 0:
                y += 1
                D -= 2 * dx
            D += 2 * dy

        return True

    def in_vision_range(self, unit):
        return self.squared_distance_to(unit) <= (self.vision_range ** 2)

    def execute_action(self, action_id):
        return self.action_logic(action_id)


class RacerUgv(Unit):
    def __init__(self, tag, x, y, team_number, terrain: Terrain, camera: Camera, render_groups=None):
        unit_type = 'RacerUgv'
        speed = 0.3
        health = 100.0
        vision_range = 2
        size = 1

        super().__init__(tag=tag, x=x, y=y, size=size, unit_type=unit_type, speed=speed, health=health,
                         vision_range=vision_range, team_number=team_number, terrain=terrain,
                         flying=False, time_per_sim_step=1.0)

    def __deepcopy__(self, memo):
        """
        Image and other pygame classes are not serializable. We deepcopy everything except for the sprite related
        attributes.
        """
        copy = RacerUgv(tag=self.tag, x=self.x, y=self.y, team_number=self.team_number, terrain=None, camera=None, render_groups=None)
        copy._visible = self._visible
        return copy
