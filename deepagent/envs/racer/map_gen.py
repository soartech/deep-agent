import math
import random
from typing import Set, Tuple, Dict, Union
from PIL import Image, ImageOps
import numpy as np
from perlin_noise import PerlinNoise
from s2clientprotocol.common_pb2 import ImageData
from skimage import measure
from skimage.morphology import rectangle, dilation, octagon

from deepagent.envs.racer.shapes import PlayableArea


def to_image(image_data: Union[ImageData, np.ndarray]) -> Image:
    if isinstance(image_data, ImageData):
        if image_data.bits_per_pixel == 8:
            mode = 'L'
        elif image_data.bits_per_pixel == 1:
            mode = '1'
        else:
            raise ValueError(f"Can't handle SC2 image_data that has {image_data.bits_per_piexl} bits per pixel.")
        image = Image.frombytes(mode=mode, size=(image_data.size.x, image_data.size.y), data=image_data.data)
        return ImageOps.flip(image)  # sc2 images are upsidedown

    if image_data.dtype not in [np.uint8, np.bool]:
        raise ValueError(f"Can't handle np.ndarray image_data that has dtype {image_data.dtype}")
    return Image.fromarray(image_data)


def rotations(array):
    rot90 = np.rot90(array, k=1)
    rot180 = np.rot90(array, k=2)
    rot270 = np.rot90(array, k=3)
    return array, rot90, rot180, rot270


ORTHOGONAL_RAMP_1_TO_2 = np.array(
    [
        [1, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1],
        [1, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1],
        [1, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1],
        [0, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 0],
        [0, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 0],
        [0, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 0],
        [2, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2],
        [2, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2],
        [2, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2]
    ]
)
ORTHOGONAL_RAMP_2_TO_3 = np.copy(ORTHOGONAL_RAMP_1_TO_2)
ORTHOGONAL_RAMP_2_TO_3[ORTHOGONAL_RAMP_2_TO_3 > 0] += 1

NS12, WE12, SN12, EW12 = rotations(ORTHOGONAL_RAMP_1_TO_2)
NS23, WE23, SN23, EW23 = rotations(ORTHOGONAL_RAMP_2_TO_3)

DIAGONAL_RAMP_1_TO_2 = np.array(
    [
        [6, 6, 6, 6, 6, 6, 0, 0, 0],
        [6, 6, 6, 6, 6, 7, 8, 0, 0],
        [6, 6, 6, 6, 7, 8, 9, 10, 0],
        [6, 6, 6, 7, 8, 9, 10, 11, 12],
        [6, 6, 7, 8, 9, 10, 11, 12, 12],
        [6, 7, 8, 9, 10, 11, 12, 12, 12],
        [0, 8, 9, 10, 11, 12, 12, 12, 12],
        [0, 0, 10, 11, 12, 12, 12, 12, 12],
        [0, 0, 0, 12, 12, 12, 12, 12, 12],
    ]
) / 6.0
DIAGONAL_RAMP_2_TO_3 = np.copy(DIAGONAL_RAMP_1_TO_2)
DIAGONAL_RAMP_2_TO_3[DIAGONAL_RAMP_2_TO_3 > 0] += 1

NWSE12, SWNE12, SENW12, NESW12 = rotations(DIAGONAL_RAMP_1_TO_2)
NWSE23, SWNE23, SENW23, NESW23 = rotations(DIAGONAL_RAMP_2_TO_3)

ANTI_CLOCKWISE_1_TO_2 = [WE12, SWNE12, SN12, SENW12, WE12, NESW12, NS12, NWSE12]
ANTI_CLOCKWISE_2_TO_3 = [WE23, SWNE23, SN23, SENW23, WE23, NESW23, NS23, NWSE23]


def surrounding_coords(y, x, max_y, max_x):
    """
    @param y: The y component is the row value of the coordinate..
    @param x: The x component is the column value of the coordinate.
    @param max_y: The highest possible y index.
    @param max_x: The highest possible x index.
    @return: A list of all of the valid coordinates directly surrounding the coordinate (y,x)
    """
    surrounding = [(y + 1, x + 1), (y, x + 1), (y - 1, x + 1), (y - 1, x), (y - 1, x - 1), (y, x - 1),
                   (y + 1, x - 1), (y + 1, x)]

    for i in range(len(surrounding) - 1, -1, -1):
        coord = surrounding[i]
        if coord[0] < 0 or coord[1] < 0 or coord[0] > max_y or coord[1] > max_x:
            surrounding.pop(i)

    return surrounding


def create_perlin_array(octaves: int, num_y: int, num_x: int):
    """
    Creates a 2D array with values [0.0,1.0] using perlin noise.
    @param octaves: The number of levels of detail you want your Perlin noise to have.
        It's kind of like zooming in and out.
    @param num_y: Height if array.
    @param num_x: Width of array.
    @return: A 2D Perlin Noise array.
    """
    noise = PerlinNoise(octaves=octaves)
    array = np.zeros((num_y, num_x))
    for y in range(num_y):
        for x in range(num_x):
            array[y, x] = noise([y / num_y, x / num_x])
    return array


def add_walls(digitized):
    """
    Add walls (values of 0) to a digitized 2D array.
    @param digitized: A digitized 2D array where 0=wall, 1=ground-level-1, 2=ground-level-2, 3=ground-level-3
    @return: A digitized array with walls between ground level areas.
    """
    walled_array = np.copy(digitized)
    shape = walled_array.shape
    for y in range(0, shape[0]):
        for x in range(0, shape[1]):
            terrain_val = digitized[y, x]
            adjacent_coords = surrounding_coords(y, x, shape[0] - 1, shape[1] - 1)
            surrounding_vals = [digitized[sc] for sc in adjacent_coords]
            if any((val > terrain_val + .99 for val in surrounding_vals)):
                walled_array[y, x] = 0
    return walled_array


def expand_top_level(digitized):
    """
    The Perlin noise generation creates bright areas (areas that get digitized to ground-level-3) that are too small.
    We also want the bright areas (level-3) to sometimes go all the way to the edge of the level below it (level-2),
    and to have shapes that are a little less Perlin and more like what we see in SC2.
    @param digitized: A digitized 2D array where 0=wall, 1=ground-level-1, 2=ground-level-2, 3=ground-level-3
    @return: A digitized array with the ground-level-3 areas expanded.
    """
    shape = digitized.shape
    top_level = np.zeros(shape, dtype=np.int)
    top_level[digitized == 3] = 3

    y_index = 0
    while y_index < shape[0]:
        x_index = 0
        y_inc = random.randint(20, 40)
        while x_index < shape[0]:
            x_inc = random.randint(20, 40)
            region = top_level[y_index:y_index + y_inc, x_index:x_index + x_inc]
            if random.random() > .7:
                region = dilation(region, rectangle(random.randint(1, 12), random.randint(1, 12)))
            else:
                region = dilation(region, octagon(random.randint(1, 10), random.randint(1, 10)))
            top_level[y_index:y_index + y_inc, x_index:x_index + x_inc] = region

            x_index += x_inc
        y_index += y_inc
    expanded_top_level = digitized.copy()
    expanded_top_level[top_level == 3] = 3
    return expanded_top_level


def dist2(vec1, vec2):
    """
    Returns squared distance.
    @param vec1: An array-like of numbers.
    @param vec2: An array-like of numbers.
    @return: The squared distance between the two vectors.
    """
    d1 = vec1[0] - vec2[0]
    d2 = vec1[1] - vec2[1]
    return d1 * d1 + d2 * d2


def connect_regions(digitized, region_boundaries, sc2_ramps=False):
    """
    Creates
    @param digitized: A digitized 2D array where 0=wall, 1=ground-level-1, 2=ground-level-2, 3=ground-level-3
    @param region_boundaries: A dict mapping region id pairs to region boundary coordinates.
    @param sc2_ramps: If True, try to connect regions with sc2-like ramps, if False, then use simple 1-pixel wide ramps.
    @return: A floating point array with transition areas between the values of the digitized ground level values,
        representing ramps between the sections.
    """
    shape = digitized.shape
    connected = np.copy(digitized).astype(np.float)
    for boundary_id, boundary_coords in region_boundaries.items():
        if len(boundary_coords) < 10:
            continue
        boundary_coords_list = list(boundary_coords)
        coord = boundary_coords_list[random.randint(0, len(boundary_coords_list)) - 1]
        place_two = len(boundary_coords_list) > 40

        closest_coords = sorted(boundary_coords_list, key=lambda c: dist2(coord, c))
        if sc2_ramps:
            # find 9 closest (including self-point)
            edge_groups = [closest_coords[:9]]
            if place_two:
                # get a 2nd group of points on the edge that are farthest away from the first group
                edge_groups += [closest_coords[-9:]]

            for closest in edge_groups:
                # Try all ramps and choose the one that is the least destructive
                min_array_diff = math.inf
                y_start_final, x_start_final = 0, 0
                ramp_slice_final = None

                mid_coord = closest[0]
                for ramp in ANTI_CLOCKWISE_1_TO_2 + ANTI_CLOCKWISE_2_TO_3:
                    ramp_y = ramp.shape[0]
                    ramp_x = ramp.shape[1]
                    min_y = max(mid_coord[0] - ramp_y, 0)
                    min_x = max(mid_coord[1] - ramp_x, 0)

                    # Try each ramp in each location around the coordinate that covers the coordinate
                    for start_y in range(min_y, min(shape[0], mid_coord[0] + ramp_y)):
                        for start_x in range(min_x, min(shape[1], mid_coord[1] + ramp_x)):
                            # Make sure that the tile will cover several coordinates that are actually between these
                            # two regions, as opposed just like one pixel in the top-left cornerr
                            indices = [(y, x) for y in range(start_y, start_y + ramp_y) for x in
                                       range(start_x, start_x + ramp_x)]
                            num_boundary_coords_involved = len([index for index in indices if index in boundary_coords])
                            if num_boundary_coords_involved < min(ramp_x, ramp_y):
                                continue

                            # Calculate the absolute difference of placing the ramp at this index
                            difference = np.sum(
                                np.absolute(
                                    ramp[:shape[0] - start_y, :shape[1] - start_x] -
                                    digitized[start_y: start_y + ramp_y, start_x:start_x + ramp_x]
                                )
                            )
                            if difference < min_array_diff and difference:
                                ramp_slice_final = ramp[:shape[0] - start_y, :shape[1] - start_x]
                                y_start_final = start_y
                                x_start_final = start_x
                                min_array_diff = difference

                # Assign the best (least destructive) ramp slice to the output array
                if ramp_slice_final is not None:
                    connected[y_start_final: y_start_final + ramp_slice_final.shape[0],
                    x_start_final:x_start_final + ramp_slice_final.shape[1]] = ramp_slice_final
        else:
            for cc in closest_coords[:10]:
                connected[cc[0], cc[1]] = mean_val = _mean_val(cc, digitized)
            if place_two:
                for cc in closest_coords[-10:]:
                    connected[cc[0], cc[1]] = _mean_val(cc, digitized)

    return connected


def _mean_val(coord, array, exclude=0):
    """
    Finds the mean value of all coordinates immediately surrounding a value in an array.
    @param coord: Coordinate into numpy array.
    @param array: Numpy array.
    @param exclude: Value to ignore in calculations.
    @return: The mean value of the array surrounding the coord, excluding the exclude value in the calculation.
    """
    shape = array.shape
    scoords = surrounding_coords(coord[0], coord[1], shape[0] - 1, shape[1] - 1)
    counts = {}
    for sc in scoords:
        if sc != exclude:
            val = array[sc]
            if val not in counts:
                counts[val] = 0
            counts[val] += 1
    mean_val = np.mean(sorted(counts, reverse=True)[:2])
    return mean_val


def make_map(y, x):
    """
    Generates a map for use in a StarcraftPythonEnv
    @param y: Height of map.
    @param x: Width of map..
    @return: An array
    """
    map_array = create_perlin_array(octaves=7, num_y=y, num_x=x)
    shape = map_array.shape

    map_array = digitize(map_array)
    map_array = expand_top_level(map_array)
    map_array = add_walls(map_array)

    labels, return_num = measure.label(map_array, background=0, return_num=True)
    region_boundaries = get_region_boundaries(map_array, labels)

    map_array = connect_regions(map_array, region_boundaries, sc2_ramps=True)
    # Fix broken walls from ramp placement
    map_array = add_walls(map_array)

    # Create and return MapData object, convert array values to sc2 height values
    heights_array = map_array

    # this map array has : 0 = walls,    1=level_1,  2=level_2,   3=level_3
    # example sc2 map has: 0 = walls, 150=level_1, 166=level_2, 182=level_3
    # convert the range
    heights_array[:] *= 16
    heights_array[heights_array > 0] += 134
    heights_array = heights_array.astype(np.uint8)

    # create a binary pathing array
    pathing_array = heights_array != 0

    # playable area of generated map is the entire map
    playable_area = PlayableArea(0, y - 1, x - 1, 0)

    return MapData(heights_array, pathing_array, playable_area)


def get_region_boundaries(digitized, labels):
    """
    @param digitized: A digitized 2D array where 0=wall, 1=ground-level-1, 2=ground-level-2, 3=ground-level-3
    @param labels: An array where the cells for each region are marked with the same region-id value.
    @return: A dict mapping tuples of region ids to the wall coordinates between those two regions
    """
    region_boundaries = {}  # type: Dict[Tuple, Set]
    zeros = np.where(digitized == 0)
    wall_locations = zip(zeros[0], zeros[1])
    for wall_location in wall_locations:
        region_edge_coords = surrounding_coords(wall_location[0], wall_location[1], labels.shape[0] - 1,
                                                labels.shape[1] - 1)
        region_edge_labels = [labels[rec[0], rec[1]] for rec in region_edge_coords]
        wall_coords = []
        boundary_ids = []
        terrain_vals = []

        for label_id, coord in zip(region_edge_labels, region_edge_coords):
            if label_id != 0:
                boundary_ids.append(label_id)
                terrain_vals.append(digitized[coord])
            else:
                wall_coords.append(coord)

        # Valid region junctions will be walls that are in-between two regions
        boundary_id = tuple(sorted(set(boundary_ids)))
        if len(boundary_id) != 2:
            continue

        # Valid region junctions will be between regions that are 1 elevation apart, not two, i.e.
        # regions are adjacent if there is a wall between levels 1 and 2, or a wall between levels 2 and 3.s
        first_terrain_val = terrain_vals[0]
        if first_terrain_val + 1 not in terrain_vals[1:] and first_terrain_val - 1 not in terrain_vals[1:]:
            continue

        if boundary_id not in region_boundaries:
            region_boundaries[boundary_id] = set()
        for coord in wall_coords:
            region_boundaries[boundary_id].add(coord)

    return region_boundaries


def digitize(array):
    """
    Takes an array and digitizes the values to different integer values, representing terrain heights.
    @param array: A floating point array.
    @return: A digitized 2D array where 0=wall, 1=ground-level-1, 2=ground-level-2, 3=ground-level-3
    """
    bins = 3
    density, edges = np.histogram(array, bins=bins, range=(0, 1))
    digitized = np.digitize(array, edges)
    digitized[digitized == bins] = bins - 1  # squash slightly too high value back to top level
    digitized = digitized + 1  # make room for a wall/untraversable zero value
    return digitized


class MapData:
    def __init__(self, terrain_height: Union[ImageData, np.ndarray], pathing_grid: Union[ImageData, np.ndarray],
                 playable_area: PlayableArea):
        self.playable_area = playable_area
        self.pathing_image = self.playable_area.crop_image(to_image(pathing_grid))
        self.terrain_image = self.playable_area.crop_image(to_image(terrain_height))


if __name__ == '__main__':
    random.seed(2125)
    map_data = make_map(y=133, x=141)
    map_data.terrain_image.show()
