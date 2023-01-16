from __future__ import annotations

import math
from typing import NamedTuple, List

import numpy as np
import pygame as pg
from PIL import Image

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Point: x={self.x} y={self.y}'

    def distance_to(self, p2):
        return math.sqrt((p2.x - self.x) ** 2 + (p2.y - self.y) ** 2)

    def squared_distance_to(self, p2):
        return (p2.x - self.x) ** 2 + (p2.y - self.y) ** 2

    @staticmethod
    def from_vec(vec: pg.math.Vector2) -> Point:
        return Point(vec.x, vec.y)


class Line(NamedTuple):
    p1: Point
    p2: Point

    def get_distance(self) -> float:
        return self.p1.distance_to(self.p2)

    def get_squared_distance(self) -> float:
        return self.p1.squared_distance_to(self.p2)

    def get_point_along(self, distance) -> Point:
        v = np.array([self.p2.x - self.p1.x, self.p2.y - self.p1.y])
        u = v / np.sqrt(v.dot(v))
        point = np.array([self.p1.x, self.p1.y]) + distance * u
        return Point(point[0], point[1])


Route = List[Point]


def point_along_route(route: Route, move_distance: float) -> Point:
    distance = 0
    for i in range(len(route) - 1):
        line = Line(route[i], route[i + 1])
        line_distance = line.get_distance()
        if distance + line_distance > move_distance:
            remaining_distance = move_distance - distance
            return line.get_point_along(remaining_distance)
    return route[-1]


class PlayableArea(NamedTuple):
    t: int
    b: int
    r: int
    l: int

    @property
    def origin_offset(self):
        return pg.Vector2((self.l, self.t))

    @property
    def center(self):
        center_x = (self.r + self.l) / 2
        center_y = (self.t + self.b) / 2
        return pg.Vector2((center_x, center_y))

    @property
    def width(self):
        return self.r - self.l

    @property
    def height(self):
        return self.b - self.t

    @staticmethod
    def from_center_width_height(center: pg.Vector2, width: int, height: int):
        half_height = height / 2.0
        half_width = width / 2.0
        t = int(round(center.y - half_height))
        b = int(round(center.y + half_height))
        r = int(round(center.x + half_width))
        l = int(round(center.x - half_width))
        return PlayableArea(t, b, r, l)

    @staticmethod
    def from_sc2_playable_area(playable_area, map_height) -> PlayableArea:
        l = playable_area.p0.x
        r = playable_area.p1.x

        # Y is upside down
        t = map_height - playable_area.p1.y
        b = map_height - playable_area.p0.y

        return PlayableArea(t, b, r, l)

    def crop_image(self, image: Image):
        return image.crop((self.l, self.t, self.r + 1, self.b + 1))