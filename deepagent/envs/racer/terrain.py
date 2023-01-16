import networkx
import pygame as pg
import numpy as np
import math
import PIL

from abc import ABC, abstractmethod
from typing import List, Tuple, Type, Union

from PIL import ImageDraw, Image
from pygame.sprite import Group
from PIL.Image import composite

from deepagent.envs.racer.utils import RED, BLUE, Camera, GREEN, BLACK, ZoomListener, MoveListener, WHITE, YELLOW, ZoomListener, MoveListener
from deepagent.envs.racer.shapes import Point, Line, Route, point_along_route
from deepagent.experiments.params import params
from deepagent.envs.racer.speed_maps import compute_current_cost_array

class RouteStrategy(ABC):
    @abstractmethod
    def create_route(self, terrain: 'Terrain', start: Point, end: Point, flying: bool = False) -> Route:
        pass


class Bresenham(RouteStrategy):
    """
    Creates routes using Bresenham's line algorithm. Terminates route at end coordinate or when it hits a wall.
    Adapted from https://github.com/encukou/bresenham
    """
    def create_route(self, terrain: 'Terrain', start: Point, end: Point, flying: bool = False) -> Route:
        x0, y0 = start.x, start.y
        x1, y1 = end.x, end.y

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
        for x in range(int(dx + 1)):
            point = Point(x0 + x * xx + y * yx, y0 + x * xy + y * yy)
            if terrain.off_map(point) or (not terrain.traversable(point) and not flying):
                # When route hits a dead end, make a final point at the boundary of that dead end (wall or map edge)
                epsilon = 1e-4
                if len(points) > 0:
                    prev_point = points[-1]
                    x_dir = point.x - prev_point.x
                    y_dir = point.y - prev_point.y
                    x_boundary = float(math.floor(prev_point.x)) if x_dir < 0 - epsilon else float(math.ceil(prev_point.x)) - epsilon if x_dir > 0 + epsilon else prev_point.x
                    y_boundary = float(math.floor(prev_point.y)) if y_dir < 0 - epsilon else float(math.ceil(prev_point.y)) - epsilon if y_dir > 0 + epsilon else prev_point.y
                    points.append(Point(x_boundary, y_boundary))
                break
            points.append(point)
            if D >= 0:
                y += 1
                D -= 2 * dx
            D += 2 * dy

        return points


class GoalArea:
    def __init__(self, location: pg.Vector2, size: pg.Vector2):
        self.location = location
        self.size = size

    @property
    def centroid(self) -> pg.Vector2:
        return pg.Vector2(self.location.x + self.size.x / 2, self.location.y + self.size.y / 2)


class Terrain(pg.sprite.Sprite, ZoomListener, MoveListener):
    elevation_height_difference = 16  # Height between different terrain levels in sc2
    half_height_difference = elevation_height_difference // 2.0  # Half way up a ramp is a vision distinction in sc2

    def __init__(self, map_data, origin_offset: pg.Vector2,
                 render_groups: List[Group], camera: Camera, goal_areas: List[GoalArea], fastest, slowest,
                 route_strategy: RouteStrategy = Bresenham()):
        """
        @param move_goal_areas_to_nodes:
        @param heights_image: Image is assumed to be 1 pixel per 1 starcraft 2 world unit.
        @param render_groups: The pygame render groups to register this sprite with.
        @param goal_areas: The GoalArea objects for the game. Note: The goal areas are separate sprites and have their
            own rendering code.
        @param move_goal_areas_to_nodes: Will move the goal areas to ensure that they cover nodes in the path grid.
        """
        pg.sprite.Sprite.__init__(self, *render_groups)
        self.camera = camera
        self.camera.register_zoom_listener(self)
        self.camera.register_move_listener(self)

        self._origin_offset = origin_offset
        self._goal_areas = goal_areas

        self.slowest = slowest
        self.fastest = fastest

        self.new_map(map_data, goal_areas)

        self._route_strategy = route_strategy
        self._set_location()

        self.dijkstra_route_time = 0.0
        self.agent_route_time = 0.0

        self.dijkstra_times = []
        self.agent_times = []
        self.incomplete_maps = 0

    def new_map(self, map_data, goal_areas):
        self._goal_areas = goal_areas
        ga = self._goal_areas[0]
        self.goal_rect = [int(ga.location.x), int(ga.location.y), int(ga.location.x + ga.size.x), int(ga.location.y + ga.size.y)]
        self.unit_x = 0
        self.unit_y = 0

        self._heights_array_a_priori = map_data.apriori_height_array
        self._obstacle_array_a_priori = map_data.apriori_obstacles
        self._speed_array_a_priori = map_data.apriori_speed_array
        self._total_cost_array = map_data.true_cost_array

        self._a_priori_cost_array = map_data.apriori_cost_array

        self._heights_array = map_data.true_height_array
        self._obstacle_array = map_data.true_obstacles
        self._speed_array = map_data.true_speed_array

        self._obstacle_image = self.np_float_to_pil(np.array(self._obstacle_array, dtype=np.float32))
        self._obstacle_surface = pg.image.fromstring(self._obstacle_image.convert('RGB').tobytes(), self._obstacle_image.size, 'RGB')
        self._obstacle_surface.set_colorkey(WHITE)

        self._heights_image = self.np_float_to_pil(self._heights_array)
        self._heights_surface = pg.image.fromstring(self._heights_image.convert('RGB').tobytes(), self._heights_image.size, 'RGB')
        self.size = self._heights_image.size

        self._current_speed = np.copy(self._speed_array_a_priori)
        self._current_obstacle = np.copy(self._obstacle_array_a_priori)
        self._current_height = np.copy(self._heights_array_a_priori)

        self._obstacle_counts = np.zeros_like(self._speed_array)
        self._no_obstacle_counts = np.zeros_like(self._speed_array)
        self._num_obs = np.zeros_like(self._speed_array)

        self.state = self.create_state()
        self.ground_truth_state = self.create_ground_truth_state()

        self.draw_map()

    def get_state(self):
        return self.state

    def get_ground_truth_state(self):
        return self.ground_truth_state

    def draw_map(self):
        state = self.ground_truth_state[:,:,0:3]
        image = self.np_float_to_pil(state)
        surface = pg.image.fromstring(image.convert('RGB').tobytes(), image.size, 'RGB')

        self.image = pg.Surface((self._heights_surface.get_width(), self._heights_surface.get_height()))
        self.image.blit(surface, (0, 0))

        self.image = self._rescale_image()
        self.rect = self.image.get_rect()

    def reset(self, unit):
        self.unit_x = unit.x
        self.unit_y = unit.y

        self._current_speed = np.copy(self._speed_array_a_priori)
        self._current_obstacle = np.copy(self._obstacle_array_a_priori)
        self._current_height = np.copy(self._heights_array_a_priori)

        self._obstacle_counts = np.zeros_like(self._speed_array)
        self._no_obstacle_counts = np.zeros_like(self._speed_array)
        self._num_obs = np.zeros_like(self._speed_array)

        if params.RuntimeParams.is_testing() and self.agent_route_time != 0.0:
            print('Agent route time: ', self.agent_route_time)
            print('Dijkstra route time: ', self.dijkstra_route_time)
            if self.dijkstra_route_time > 0.0:
                if self.agent_route_time > 1000.0:
                    self.incomplete_maps += 1
                    self.agent_times.append(1000.0)
                else:
                    self.agent_times.append(self.agent_route_time)
                self.dijkstra_times.append(self.dijkstra_route_time)

        self.dijkstra_route_time = self._total_cost_array[self.unit_y, self.unit_x]
        self.agent_route_time = 0.0

        self.observe(unit)

    # debugging for viewing cost map
    # def create_state(self):
    #     cost = np.copy(self._total_cost_array)
    #     cost = np.interp(cost, (cost.min(), cost.max()), (-1,1))
    #
    #     return np.stack([cost, cost, cost], axis=-1)

    def compute_new_dijkstra(self):
        mask = np.invert(self._current_obstacle)
        speed_with_obstacle = np.copy(self._current_speed)
        np.putmask(speed_with_obstacle, mask, 0.0)
        cost_array = compute_current_cost_array(speed_with_obstacle, self.goal_rect)
        return cost_array

    def get_best_action(self, unit):
        cost_array = self.compute_new_dijkstra()

        min_cost = 9999999
        min_dx = 0
        min_dy = 0

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                point = Point(self.unit_x + dx, self.unit_y + dy)
                if self.traversable(point):
                    cost = cost_array[int(point.y - self._origin_offset.y), int(point.x - self._origin_offset.x)]
                    speed = self._current_speed[int(point.y - self._origin_offset.y), int(point.x - self._origin_offset.x)]
                    cell_delta = math.sqrt(dx ** 2 + dy ** 2)
                    time_to_reach = cell_delta / speed
                    cost += time_to_reach
                    if cost > 0.0001 and cost < min_cost:
                        min_cost = cost
                        min_dx = dx
                        min_dy = dy

        action_idx = -1

        if min_dy == 1 and min_dx == 0:
            action_idx = 0
        elif min_dy == -1 and min_dx == 0:
            action_idx = 1
        elif min_dy == 0 and min_dx == 1:
            action_idx = 2
        elif min_dy == 0 and min_dx == -1:
            action_idx = 3
        elif min_dy == 1 and min_dx == 1:
            action_idx = 4
        elif min_dy == 1 and min_dx == -1:
            action_idx = 5
        elif min_dy == -1 and min_dx == 1:
            action_idx = 6
        elif min_dy == -1 and min_dx == -1:
            action_idx = 7

        return action_idx

    def create_ground_truth_state(self):
        mask = np.invert(self._obstacle_array)
        speed_with_obstacle = np.copy(self._speed_array)
        speed_with_obstacle[self.unit_y, self.unit_x] = self.fastest * 1.1
        self.fill_goal(speed_with_obstacle, self.fastest * 1.1)
        np.putmask(speed_with_obstacle, mask, 0.0)
        speed_with_obstacle = np.interp(speed_with_obstacle, (0, self.fastest * 1.1), (-1, +1))

        height_with_obstacle = np.copy(self._heights_array)
        height_with_obstacle[self.unit_y, self.unit_x] = height_with_obstacle.max() * 1.1
        self.fill_goal(height_with_obstacle, height_with_obstacle.max())
        np.putmask(height_with_obstacle, mask, 0.0)
        height_with_obstacle = np.interp(height_with_obstacle, (height_with_obstacle.min(), height_with_obstacle.max()), (-1, +1))

        confidence = np.copy(self._num_obs)
        confidence[self.unit_y, self.unit_x] = 0
        confidence = np.clip(confidence, 0, 10)

        if params.Racer.alternate_confidence:
            mask2 = np.not_equal(self._obstacle_array, self._obstacle_array_a_priori)
            confidence = np.subtract(confidence, 20, out=confidence, where=mask2)
            confidence = np.interp(confidence, (-10, 10), (-1, +1))
        else:
            confidence = np.interp(confidence, (0, 10), (-1, +1))

        if params.Racer.include_a_priori_cost:
            a_priori_cost = np.copy(self._cost_array)
            a_priori_cost = np.interp(a_priori_cost, (a_priori_cost.min(), a_priori_cost.max()), (-1, +1))

            return np.stack([height_with_obstacle, speed_with_obstacle, confidence, a_priori_cost], axis=-1)
        else:
            return np.stack([height_with_obstacle, speed_with_obstacle, confidence], axis=-1)

    def create_state(self):
        mask = np.invert(self._current_obstacle)
        speed_with_obstacle = np.copy(self._current_speed)
        speed_with_obstacle[self.unit_y, self.unit_x] = self.fastest*1.1
        self.fill_goal(speed_with_obstacle, self.fastest*1.1)
        np.putmask(speed_with_obstacle, mask, 0.0)
        speed_with_obstacle = np.interp(speed_with_obstacle, (0, self.fastest*1.1), (-1, +1))

        height_with_obstacle = np.copy(self._current_height)
        height_with_obstacle[self.unit_y, self.unit_x] = height_with_obstacle.max()*1.1
        self.fill_goal(height_with_obstacle, height_with_obstacle.max())
        np.putmask(height_with_obstacle, mask, 0.0)
        height_with_obstacle = np.interp(height_with_obstacle, (height_with_obstacle.min(), height_with_obstacle.max()), (-1, +1))

        confidence = np.copy(self._num_obs)
        confidence[self.unit_y, self.unit_x] = 0
        confidence = np.clip(confidence, 0, 10)


        if params.Racer.alternate_confidence:
            mask2 = np.not_equal(self._current_obstacle, self._obstacle_array_a_priori)
            confidence = np.multiply(confidence, -1, out=confidence, where=mask2)
            confidence = np.interp(confidence, (-10, 10), (-1, +1))
        else:
            confidence = np.interp(confidence, (0, 10), (-1, +1))

        if params.Racer.include_a_priori_cost:
            a_priori_cost = np.copy(self._a_priori_cost_array)
            a_priori_cost = np.interp(a_priori_cost, (a_priori_cost.min(), a_priori_cost.max()), (-1, +1))

            return np.stack([height_with_obstacle, speed_with_obstacle, confidence, a_priori_cost], axis=-1)
        else:
            return np.stack([height_with_obstacle, speed_with_obstacle, confidence], axis=-1)

    def get_heuristic_vector(self):
        current_cost = self._a_priori_cost_array[self.unit_y, self.unit_x]
        vector = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                if not self.traversable(Point(self.unit_x + dx, self.unit_y + dy)):
                    vector.append(-1.0)
                else:
                    cost = self._a_priori_cost_array[self.unit_y + dy, self.unit_x + dx]
                    vector.append((current_cost - cost)/10.0)

        return np.array(vector, dtype=np.float32)

    def fill_goal(self, array, value):
        ga = self._goal_areas[0]

        x = int(ga.location.x)
        y = int(ga.location.y)
        for dx in range(int(ga.size.x+1)):
            for dy in range(int(ga.size.y+1)):
                array[y + dy, x + dx] = value

    def observe(self, unit: 'Unit'):
        height = self._heights_array[unit.y, unit.x]

        for dx in range(-unit.vision_range, unit.vision_range+1):
            for dy in range(-unit.vision_range, unit.vision_range+1):
                point = Point(unit.x + dx, unit.y + dy)
                if self.off_map(point):
                    continue
                if dx*dx + dy*dy <= unit.vision_range*unit.vision_range and self._heights_array[unit.y+dy, unit.x+dx] <= height:
                    self._num_obs[unit.y+dy, unit.x+dx] += 1
                    self._current_height[unit.y+dy, unit.x+dx] = self._heights_array[unit.y+dy, unit.x+dx]
                    # TODO handle uncertainty in observation
                    obstacle = not self._obstacle_array[unit.y+dy, unit.x+dx]
                    if obstacle:
                        self._obstacle_counts[unit.y+dy, unit.x+dx] += 1
                    else:
                        self._no_obstacle_counts[unit.y+dy, unit.x+dx] += 1
                    if self._obstacle_counts[unit.y+dy, unit.x+dx] > self._no_obstacle_counts[unit.y+dy, unit.x+dx]:
                        self._current_obstacle[unit.y+dy, unit.x+dx] = False
                    else:
                        self._current_obstacle[unit.y+dy, unit.x+dx] = True
                    self._current_speed[unit.y+dy, unit.x+dx] = self._speed_array[unit.y+dy, unit.x+dx]

        self.state = self.create_state()
        self.ground_truth_state = self.create_ground_truth_state()
        self.draw_map()

    def np_float_to_pil(self, np_array: np.array):
        np_array = np.copy(np_array)
        max = np.amax(np_array)
        min = np.amin(np_array)
        np_array -= min
        np_array *= (255.0/(max-min))
        int_array = np.uint8(np_array)
        return Image.fromarray(np.uint8(int_array)).convert('RGB')

    def _set_location(self):
        self.rect.x, self.rect.y = self.camera.translate(self._origin_offset.x, self._origin_offset.y)

    def on_move(self):
        self._set_location()

    def on_zoom(self):
        self.image = self._rescale_image()

    @property
    def goal_node_locations(self):
        return [pg.Vector2(goal_node[1]['o'][1], goal_node[1]['o'][0]) for goal_node in self._goal_nodes]

    def _rescale_image(self):
        return pg.transform.scale(self.image, [int(s * self.camera.pixels_per_game_unit) for s in self.size])

    def off_map(self, point: Point):
        x, y = point.x - self._origin_offset.x, point.y - self._origin_offset.y
        return x < 0 or x >= self._obstacle_surface.get_width() or y < 0 or y >= self._obstacle_surface.get_height()

    def create_action_mask(self, unit, mask):
        mask = np.ones_like(mask)
        # action order is n, s, w, e, nw, ne, sw, se
        action_idx = 0
        action_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dy, dx in action_offsets:
            if not self.traversable(Point(unit.x+dx, unit.y+dy)):
                mask[action_idx] = 0.0
            action_idx += 1
        return mask

    def move_unit_one_cell(self, unit: 'Unit', north: bool, west: bool):
        x = 0
        y = 0
        starting_cost = self._total_cost_array[unit.y, unit.x]

        if west is not None:
            x = 1 if west else -1
        if north is not None:
            y = 1 if north else -1

        location = Point(unit.x, unit.y)
        destination = Point(unit.x+x, unit.y+y)

        if x != 0 and y != 0:
            move_cost = 10.0 * math.sqrt(2.0) / self._speed_array[unit.y, unit.x]
        else:
            move_cost = 10.0 / self._speed_array[unit.y, unit.x]
        self.agent_route_time += move_cost

        if self.off_map(location):
            raise ValueError(f'Starting location {location} is out of bounds for unit {unit}')
        elif not self.traversable(destination):
            self.observe(unit)
            # penalty for choosing invalid action
            return -1.0/4.0

        unit.x += x
        unit.y += y

        self.unit_x = unit.x
        self.unit_y = unit.y

        ending_cost = self._total_cost_array[unit.y, unit.x]

        self.observe(unit)

        return ((starting_cost-ending_cost)-move_cost)/100.0

    def in_goal(self, unit: 'Unit'):
        x = unit.x
        y = unit.y

        ga = self._goal_areas[0]
        if x >= ga.location.x and x <= ga.location.x + ga.size.x and y >= ga.location.y and y <= ga.location.y + ga.size.y:
            return True
        return False

    def nearest_goal_dist(self, units: List['Unit']):
        dist = 9999999999999999999999.9
        for u in units:
            for ga in self._goal_areas:
                dx = u.x - ga.centroid.x
                dy = u.y - ga.centroid.y
                d = math.sqrt(dx*dx + dy*dy)
                if d < dist:
                    dist = d
        return dist

    def lidar(self, point: Point):
        height = self.get_height(point)

        positive_x = 0
        positive_x_height = 0
        for i in range(10):
            positive_x = i
            if not self.traversable(Point(point.x + i, point.y)):
                positive_x_height = self.compute_height_diff(point.x + i + 3, point.y, height)
                break

        negative_x = 0
        negative_x_height = 0
        for i in range(10):
            negative_x = i
            if not self.traversable(Point(point.x - i, point.y)):
                negative_x_height = self.compute_height_diff(point.x - i - 3, point.y, height)
                break

        positive_y = 0
        positive_y_height = 0
        for i in range(10):
            positive_y = i
            if not self.traversable(Point(point.x, point.y + i)):
                positive_y_height = self.compute_height_diff(point.x, point.y + i + 3, height)
                break

        negative_y = 0
        negative_y_height = 0
        for i in range(10):
            negative_y = i
            if not self.traversable(Point(point.x, point.y - i)):
                negative_y_height = self.compute_height_diff(point.x, point.y - i - 3, height)
                break

        return (10 - positive_x, -(10 - negative_x), 10 - positive_y, -(10 - negative_y), positive_x_height, negative_x_height, positive_y_height, negative_y_height)

    def compute_height_diff(self, x, y, height):
        p = Point(x, y)
        if self.off_map(p):
            return 0.0
        h = self.get_height(p)
        if h > height:
            return 1.0
        elif h < height:
            return -1.0
        else:
            return 0.0

    def traversable(self, point: Point):
        return not self.off_map(point) and self._obstacle_array[int(point.y - self._origin_offset.y), int(point.x - self._origin_offset.x)]

    def get_speed(self, point: Point):
        return float(self._speed_array[int(point.y - self._origin_offset.y), int(point.x-self._origin_offset.x)])

    def get_height(self, point: Point):
        if self.off_map(point):
            return 0.0
        return self._heights_array[int(point.y - self._origin_offset.y), int(point.x - self._origin_offset.x)]

    def to_terrain_vec(self, vec: pg.Vector2):
        return pg.Vector2(vec.x - self._origin_offset.x, vec.y - self._origin_offset.y)

    def to_terrain_point(self, point: Point):
        return Point(point.x - self._origin_offset.x, point.y - self._origin_offset.y)

    def to_world_point(self, point: Point):
        return Point(point.x + self._origin_offset.x, point.y + self._origin_offset.y)


