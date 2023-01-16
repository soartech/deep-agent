from abc import abstractmethod, ABC

import pygame as pg

from deepagent.envs.racer.shapes import PlayableArea

RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)


class ZoomListener(ABC):
    @abstractmethod
    def on_zoom(self):
        pass


class MoveListener(ABC):
    @abstractmethod
    def on_move(self):
        pass


class Camera:
    def __init__(self, screen, playable_area: PlayableArea, pixels_per_game_unit: float, move_speed: float = 6,
                 min_pixels_per_game_unit: int = 1, max_pixels_per_game_unit: float = 64, zoom_speed: float = 1):
        self.screen = screen
        self.playable_area = playable_area
        self.pixels_per_game_unit = pixels_per_game_unit
        self.screen_center = self.get_center()
        self.location = self.screen_center
        self.move_speed = move_speed
        self.min_pixels_per_game_unit = min_pixels_per_game_unit
        self.max_pixels_per_game_unit = max_pixels_per_game_unit
        self.zoom_speed = zoom_speed
        self.zoom_listeners = set()
        self.move_listeners = set()

    def translate(self, x, y):
        return x * self.pixels_per_game_unit + self.location.x, y * self.pixels_per_game_unit + self.location.y

    def move(self, vec: pg.Vector2):
        if vec.length() != 0:
            vec.scale_to_length(self.pixels_per_game_unit * self.move_speed)
        self.location -= vec
        self.notify_move()

    def on_screen_resize(self, screen):
        self.screen = screen
        new_screen_center = self.get_center()
        difference = new_screen_center - self.screen_center
        self.screen_center = new_screen_center
        self.move(pg.Vector2((difference.x, difference.y)))

    def get_center(self):
        width, height = self.screen.get_size()
        game_width = self.playable_area.r - self.playable_area.l
        game_height = self.playable_area.b - self.playable_area.t
        x = (width - game_width * self.pixels_per_game_unit) / 2.0 - self.playable_area.l * self.pixels_per_game_unit
        y = (height - game_height * self.pixels_per_game_unit) / 2.0 - self.playable_area.t * self.pixels_per_game_unit
        return pg.Vector2((x, y))

    def zoom(self, zoom):
        new = self.pixels_per_game_unit + zoom * self.zoom_speed
        self.pixels_per_game_unit = min(self.max_pixels_per_game_unit, max(self.min_pixels_per_game_unit, new))
        new_center = self.get_center()
        self.location += new_center - self.screen_center
        self.screen_center = self.get_center()
        self.notify_zoom()

    def notify_zoom(self):
        for listener in self.zoom_listeners:
            listener.on_zoom()

    def register_zoom_listener(self, zoom_listener: ZoomListener):
        self.zoom_listeners.add(zoom_listener)

    def unregister_zoom_listener(self, zoom_listener: ZoomListener):
        if zoom_listener in self.zoom_listeners:
            self.zoom_listeners.remove(zoom_listener)

    def notify_move(self):
        for listener in self.move_listeners:
            listener.on_move()

    def register_move_listener(self, move_listener: MoveListener):
        self.move_listeners.add(move_listener)

    def unregister_move_listener(self, move_listener: MoveListener):
        if move_listener in self.move_listeners:
            self.move_listeners.remove(move_listener)
