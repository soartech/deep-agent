import math, os
import random
import time
import numpy as np
from collections import deque
from heapq import heappush,heappop

from skimage import measure
import deepagent.envs.racer.map_gen as mg
from deepagent.envs.racer.map_gen import make_map, MapData, create_perlin_array


class Ob(object):
    """Utility class can be used for just an empty object, or with attributes initialized by a dictionary"""
    def __init__(self,dict=None):
        if dict is not None:
            for k,v in dict.items():
                setattr(self,k,v)


## note result and hdr are incompatible
def splitTF(seq,key=lambda x:x,result=lambda x:x,hdr=False):
    T = []
    F = []
    start = 0
    if hdr:
        start = 1
        T.append(seq[0])
        F.append(seq[0])
    for item in seq[start:]:
        if key(item):
            T.append(result(item))
        else:
            F.append(result(item))
    return T,F


def digitize_ob(array):
    """
    Takes an array and digitizes the values to different integer values, representing terrain heights.
    @param array: A floating point array.
    @return: A digitized 2D array where 0=wall, 1=ground-level-1, 2=ground-level-2, 3=ground-level-3
    """
    bins = 3
    density, edges = np.histogram(array, bins=bins, range=(0, 1))
    digitized = np.digitize(array, edges)
    digitized1 = digitized.copy()
    digitized[digitized == bins] = bins - 1  # squash slightly too high value back to top level
    digitized = digitized + 1  # make room for a wall/untraversable zero value
    # return Ob(locals())
    return digitized


def make_map_ob(y, x, octaves=7):
    map_array1 = mg.create_perlin_array(octaves, num_y=y, num_x=x)
    map_array2 = digitize_ob(map_array1)
    map_array3 = mg.expand_top_level(map_array2)
    map_array4 = mg.add_walls(map_array3)
    labels, return_num = measure.label(map_array4, background=0, return_num=True)
    region_boundaries = mg.get_region_boundaries(map_array4, labels)
    map_array5 = mg.connect_regions(map_array4, region_boundaries, sc2_ramps=True)
    map_array6 = mg.add_walls(map_array5)
    heights_array = map_array6
    heights_array[:] *= 16
    heights_array[heights_array > 0] += 134
    heights_array = heights_array.astype(np.uint8)
    pathing_array = heights_array != 0
    playable_area = mg.PlayableArea(0, y - 1, x - 1, 0)
    map_data = mg.MapData(heights_array, pathing_array, playable_area)
    map_data.heights_only1 = map_array3
    map_data.heights_only2 = map_array5
    return map_data


def peaks(arr):
    points = []
    h,w = arr.shape
    for y in range(1,h-2):
        for x in range(1,w-2):
            c1 = arr[y,x] > arr[y-1,x] 
            c2 = arr[y,x] > arr[y+1,x] 
            c3 = arr[y,x] > arr[y,x-1] 
            c4 = arr[y,x] > arr[y,x+1] 
            if c1 and c2 and c3 and c4:
                points.append([x,y,arr[y,x]])
    return points


def segment_peaks(arr):
    seg = np.zeros(arr.shape,dtype=np.int)
    pks = peaks(arr)
    random.shuffle(pks)  # shuffle indices so display looks better
    q = deque()
    lookup = {}
    for i,tup in enumerate(pks):
        k = i + 1
        x,y,z = tup
        seg[y,x] = k
        item = (k,x,y,z)
        q.append(item)
        lookup[k] = tup
    delta = [(0,-1),(0,1),(-1,0),(1,0)]
    h,w = arr.shape
    bails = []
    while q:
        k,x,y,z = q.popleft()
        for dx,dy in delta:
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < w):
                bails.append(1)
                continue
            if not (0 <= ny < h):
                bails.append(2)
                continue
            if seg[ny,nx] > 0:
                bails.append(3)
                continue
            nz = arr[ny,nx]
            if nz > z:
                bails.append(4)
                continue
            seg[ny,nx] = k
            item = (k,nx,ny,nz)
            q.append(item)
    o = Ob()
    o.seg = seg
    o.pks = pks
    o.bails = bails
    return o


def make_zlookup(pks,nbins=10,pad=5):
    zvals = np.array([x[2] for x in pks])
    counts,edges = np.histogram(zvals,bins=nbins)
    levels = np.digitize(zvals,edges)
    zlookup = {}
    for i,level in enumerate(levels):
        k = i + 1
        zlookup[k] = level + pad
    return zlookup


def do_zlookup(seg_array,zlookup):
    out_array = np.zeros(seg_array.shape,dtype=np.int)
    h,w = seg_array.shape
    for y in range(h):
        for x  in range(w):
            seg_id = seg_array[y,x]
            level = zlookup.get(seg_id,seg_id)
            out_array[y,x] = level
    return out_array


def walls_from_terrain(levels,heights_array):
    h,w = levels.shape
    out = levels.copy()
    for y in range(h):
        for x in range(w):
            level2 = heights_array[y,x]
            if level2 == 0:
                out[y,x] = 0
    return out


def add_traversability(seg,nbins=10,pad=5):
    seg.zlookup = make_zlookup(seg.pks,nbins,pad)
    seg.levels = do_zlookup(seg.seg,seg.zlookup)
    h,w = seg.seg.shape
    o = make_map_ob(h,w)
    seg.heights_array = o.heights_array
    seg.traverse = walls_from_terrain(seg.levels,seg.heights_array)


def make_speed_lookup(pks, nspeeds=5, slowest=2.0, fastest=31.0):
    zvals = np.array([x[2] for x in pks])
    if nspeeds <= 1:
        v1 = float(np.min(zvals) - 1)
        v2 = float(np.max(zvals) + 1)
        edges = [v1,v2]
    else:
        counts,edges = np.histogram(zvals,bins=nspeeds)
    levels = np.digitize(zvals,edges)
    zlookup = {}
    speeds = np.linspace(slowest,fastest,nspeeds+1)
    for i,level in enumerate(levels):
        k = i + 1
        zlookup[k] = speeds[level-1]
    return zlookup


def perturb_speed_lookup(zlookup, max_delta, slowest, fastest):
    perturbed_zlookup = {}
    for key,val in zlookup.items():
        delta = random.uniform(-max_delta, max_delta)
        newval = val + delta
        newval = min(fastest,max(slowest,newval))
        perturbed_zlookup[key] = newval
    return perturbed_zlookup


def alter_walls(speed_array_with_walls, wall_octaves=40, nadded_walls=5, ndeleted_walls=5):
    h,w = speed_array_with_walls.shape
    perlin_array = create_perlin_array(wall_octaves, h, w)
    seg = segment_peaks(perlin_array)


def add_walls(speed_array,seg,seg_numbers):
    h,w = speed_array.shape
    seg_array = seg.seg
    previous_speed = 10
    for y in range(h):
        for x in range(w):
            speed = speed_array[y,x]
            seg_num = seg_array[y,x]
            if seg_num in seg_numbers:
                speed_array[y,x] = 0


def delete_walls(speed_array,seg,seg_numbers):
    h,w = speed_array.shape
    seg_array = seg.seg
    previous_speed = 10
    for y in range(h):
        for x in range(w):
            speed = speed_array[y,x]
            if speed == 0:
                seg_num = seg_array[y,x]
                if seg_num in seg_numbers:
                    speed_array[y,x] = previous_speed
            else:
                previous_speed = speed


def add_walls_sat(speed_array,seg,seg_numbers):
    h,w = speed_array.shape
    seg_array = seg.seg
    for y in range(h):
        for x in range(w):
            seg_num = seg_array[y,x]
            if seg_num in seg_numbers:
                speed_array[y,x] = 0


def delete_walls_sat(speed_array,seg,seg_numbers,base_speed=2):
    h,w = speed_array.shape
    seg_array = seg.seg
    for y in range(h):
        for x in range(w):
            seg_num = seg_array[y,x]
            if seg_num in seg_numbers:
                speed_array[y,x] = base_speed


def find_segs_with_walls(speed_array,seg):
    wall_segs = set()
    h,w = speed_array.shape
    seg_array = seg.seg
    for y in range(h):
        for x in range(w):
            speed = speed_array[y,x]
            if speed == 0:
                seg_num = seg_array[y,x]
                if seg_num > 0:
                    wall_segs.add(seg_num)
    return wall_segs


def find_segs_without_walls(speed_array,seg):
    nowall_segs = set()
    h,w = speed_array.shape
    seg_array = seg.seg
    for y in range(h):
        for x in range(w):
            speed = speed_array[y,x]
            if speed != 0:
                seg_num = seg_array[y,x]
                if seg_num > 0:
                    nowall_segs.add(seg_num)
    return nowall_segs


def cover(arr,src,dest):
    h,w = arr.shape
    for y in range(h):
        for x in range(w):
            if arr[y,x] == src:
                arr[y,x] = dest


def pick_segs_to_flip_sat(speed_array,seg, nadded_walls=5, ndeleted_walls=5):
    wall_segs = find_segs_with_walls(speed_array,seg)
    nowall_segs = find_segs_with_walls(speed_array,seg)
    nsegs = len(seg.pks)
    with_walls = [x for x in range(1,nsegs+1) if x in wall_segs]
    without = [x for x in range(1,nsegs+1) if x in nowall_segs]
    try:
        to_delete = random.sample(with_walls,ndeleted_walls)
    except Exception as e:
        print(e)
        to_delete = []
    try:
        to_add = random.sample(without,nadded_walls)
    except Exception as e:
        print(e)
        to_add = []
    return to_add, to_delete


def pick_segs_to_flip(speed_array,seg, nadded_walls=5, ndeleted_walls=5):
    wall_segs = find_segs_with_walls(speed_array,seg)
    nsegs = len(seg.pks)
    with_walls,without = splitTF(range(1,nsegs+1),lambda x:x in wall_segs)
    try:
        to_delete = random.sample(with_walls,ndeleted_walls)
    except Exception as e:
        print(e)
        to_delete = []
    try:
        to_add = random.sample(without,nadded_walls)
    except Exception as e:
        print(e)
        to_add = []
    return to_add, to_delete


def temp_data(map_data, seg_array, speed_array):
    out = Ob()
    out.playable_area = map_data.playable_area
    out.pathing_image = map_data.pathing_image
    out.terrain_image = map_data.terrain_image
    out.seg_array = seg_array       # array of segment indices
    out.speed_array = speed_array   # array of speed values
    return out


def add_apriori_map(smap, slowest, fastest, max_delta, wall_octaves, nadded_walls, ndeleted_walls):
    smap.perturbed_zlookup = perturb_speed_lookup(smap.zlookup, max_delta, slowest, fastest)
    smap.apriori_speed_array = do_zlookup(smap.seg_array,smap.perturbed_zlookup)
    cover(smap.apriori_speed_array,0,slowest)
    #heights_array = np.array(smap.terrain_image)
    obstacles_array = np.array(smap.pathing_image)
    smap.apriori_speed_array_with_walls = walls_from_terrain(smap.apriori_speed_array,obstacles_array)
    h,w = smap.apriori_speed_array_with_walls.shape
    perlin_array = create_perlin_array(wall_octaves, h, w)
    smap.wallflip_seg = segment_peaks(perlin_array)
    to_add, to_delete = pick_segs_to_flip(smap.apriori_speed_array_with_walls, smap.wallflip_seg, nadded_walls, ndeleted_walls)
    add_walls(smap.apriori_speed_array_with_walls,smap.wallflip_seg,to_add)
    delete_walls(smap.apriori_speed_array_with_walls,smap.wallflip_seg,to_delete)


# old make_speed_map(h=133, w=141, map_octaves=7,patch_octaves=20, nspeeds=5, slowest=2.0, fastest=31.0, ...
def make_speed_map(h=64, w=64, map_octaves=4,patch_octaves=17, nspeeds=5, slowest=2.0, fastest=31.0, 
                max_delta=12, wall_octaves=20, nadded_walls=5, ndeleted_walls=8, goal_rect=[5,5,16,8]):
    t1 = time.time()
    #map_data = make_map(h, w)
    map_data = make_map_ob(h, w,map_octaves)
    t2 = time.time()
    patch_perlin_array = create_perlin_array(patch_octaves, h, w)
    seg = segment_peaks(patch_perlin_array)
    seg_array = seg.seg   # patches numbered 1 to N (0 for remnants at borders)
    zlookup = make_speed_lookup(seg.pks, nspeeds, slowest, fastest)
    true_speed_array = do_zlookup(seg_array,zlookup)
    cover(true_speed_array,0,slowest)
    out = temp_data(map_data, seg_array, true_speed_array)
    heights_array = np.array(map_data.terrain_image)
    obstacles_array = np.array(map_data.pathing_image)
    out.true_speed_array_with_walls = walls_from_terrain(true_speed_array,obstacles_array)
    out.patch_perlin_array = patch_perlin_array
    out.zlookup = zlookup
    add_apriori_map(out, slowest, fastest, max_delta, wall_octaves, nadded_walls, ndeleted_walls)
    true_speed_with_walls = np.array(out.true_speed_array_with_walls, np.float32)
    apriori_speed_with_walls = np.array(out.apriori_speed_array_with_walls, np.float32)
    true_dijkstra = dijkstra(true_speed_with_walls,goal_rect,cell_size=10,delta=delta_8way)
    apriori_dijkstra = dijkstra(apriori_speed_with_walls,goal_rect,cell_size=10,delta=delta_8way)
    true_obs = np.array(out.true_speed_array_with_walls, np.bool)
    apriori_obs = np.array(out.apriori_speed_array_with_walls, np.bool)
    true_speed = np.array(out.speed_array,np.float32)
    apriori_speed = np.array(out.apriori_speed_array,np.float32)
    heights = np.array(heights_array,np.float32)
    heights_only = np.array(map_data.heights_only1,np.float32)
    heights_only *= 16
    heights_only += 134
    # speed_data = SpeedData(true_obs,apriori_obs,true_speed,apriori_speed,heights_only,heights_only,
    #             true_speed_with_walls,true_dijkstra.cost_array,apriori_speed_with_walls,apriori_dijkstra.cost_array,true_dijkstra.traceback,goal_rect)
    speed_data = SpeedData(apriori_obs,true_obs,apriori_speed,true_speed,heights_only,heights_only,
                apriori_speed_with_walls,apriori_dijkstra.cost_array,true_speed_with_walls,true_dijkstra.cost_array,apriori_dijkstra.traceback,goal_rect)
    # speed_data.terrain = heights
    t3 = time.time()
    print(f'make_speed_map took {t3-t1:.1f} secs make_map was {t2-t1:.1f} secs')
    return speed_data
    # return Ob(locals())


def add_altered_map(inmap):
    inmap.perturbed_zlookup = perturb_speed_lookup(inmap.zlookup, inmap.max_delta, inmap.slowest, inmap.fastest)
    inmap.altered_speed_array = do_zlookup(inmap.seg_array,inmap.perturbed_zlookup)
    cover(inmap.altered_speed_array,0,inmap.slowest)
    #heights_array = np.array(inmap.terrain_image)
    obstacles_array = np.array(inmap.pathing_image)
    inmap.altered_speed_array_with_walls = walls_from_terrain(inmap.altered_speed_array,obstacles_array)
    h,w = inmap.altered_speed_array_with_walls.shape
    perlin_array = create_perlin_array(inmap.wall_octaves, h, w)
    inmap.wallflip_seg = segment_peaks(perlin_array)
    to_add, to_delete = pick_segs_to_flip(inmap.altered_speed_array_with_walls, inmap.wallflip_seg, inmap.nadded_walls, inmap.ndeleted_walls)
    add_walls(inmap.altered_speed_array_with_walls,inmap.wallflip_seg,to_add)
    delete_walls(inmap.altered_speed_array_with_walls,inmap.wallflip_seg,to_delete)


### Call this first and one time only
def make_initial_map(h=64, w=64, map_octaves=4,patch_octaves=17, nspeeds=5, slowest=2.0, fastest=31.0, 
                max_delta=12, wall_octaves=20, nadded_walls=5, ndeleted_walls=8, goal_rect=[5,5,16,8]):
    t1 = time.time()
    map_data = make_map_ob(h, w,map_octaves)
    t2 = time.time()
    patch_perlin_array = create_perlin_array(patch_octaves, h, w)
    seg = segment_peaks(patch_perlin_array)
    seg_array = seg.seg   # patches numbered 1 to N (0 for remnants at borders)
    zlookup = make_speed_lookup(seg.pks, nspeeds, slowest, fastest)
    initial_speed_array = do_zlookup(seg_array,zlookup)
    cover(initial_speed_array,0,slowest)
    inmap = temp_data(map_data, seg_array, initial_speed_array)
    inmap.map_data = map_data
    obstacles_array = np.array(map_data.pathing_image)
    inmap.initial_speed_array_with_walls = walls_from_terrain(initial_speed_array,obstacles_array)
    inmap.patch_perlin_array = patch_perlin_array
    inmap.zlookup = zlookup
    inmap.slowest = slowest
    inmap.fastest = fastest
    inmap.max_delta= max_delta
    inmap.wall_octaves = wall_octaves
    inmap.nadded_walls = nadded_walls
    inmap.ndeleted_walls = ndeleted_walls
    inmap.goal_rect = goal_rect
    return inmap


### Call this many times to get the SpeedData structure with apriori and randomized true maps (same structure as before)
def get_altered_map(inmap):
    add_altered_map(inmap)
    initial_speed_with_walls = np.array(inmap.initial_speed_array_with_walls, np.float32)
    altered_speed_with_walls = np.array(inmap.altered_speed_array_with_walls, np.float32)
    initial_dijkstra = dijkstra(initial_speed_with_walls,inmap.goal_rect,cell_size=10,delta=delta_8way)
    altered_dijkstra = dijkstra(altered_speed_with_walls,inmap.goal_rect,cell_size=10,delta=delta_8way)
    initial_obs = np.array(inmap.initial_speed_array_with_walls, np.bool)
    altered_obs = np.array(inmap.altered_speed_array_with_walls, np.bool)
    initial_speed = np.array(inmap.speed_array,np.float32)
    altered_speed = np.array(inmap.altered_speed_array,np.float32)
    heights_only = np.array(inmap.map_data.heights_only1,np.float32)
    heights_only *= 16
    heights_only += 134
    speed_data = SpeedData(altered_obs,initial_obs,altered_speed,initial_speed,heights_only,heights_only,
                altered_speed_with_walls,altered_dijkstra.cost_array,initial_speed_with_walls,initial_dijkstra.cost_array,altered_dijkstra.traceback,inmap.goal_rect)
    return speed_data


def load_stored_map():
    from deepagent.envs.racer import terrain
    racer_dir = os.path.dirname(terrain.__file__)
    path = os.path.join(racer_dir,'sentinal_maps','TRAV64.npy')
    arr = np.load(path)
    return arr


### Call this first and one time only - for satellite map
def make_initial_satellite_map(h=64, w=64, map_octaves=4,patch_octaves=17, nspeeds=5, slowest=2.0, fastest=2.0, 
                max_delta=12, wall_octaves=20, nadded_walls=5, ndeleted_walls=8, goal_rect=[44,2,50,8]):
    inmap = Ob()
    inmap.initial_speed_array_with_walls = load_stored_map()
    inmap.speed_array = np.ones_like(inmap.initial_speed_array_with_walls,dtype=np.float32) * slowest
    inmap.slowest = slowest
    inmap.fastest = fastest
    inmap.max_delta= max_delta
    inmap.wall_octaves = wall_octaves
    inmap.nadded_walls = nadded_walls
    inmap.ndeleted_walls = ndeleted_walls
    inmap.goal_rect = goal_rect
    return inmap


def add_altered_sattelite_map(inmap):
    inmap.altered_speed_array_with_walls = np.copy(inmap.initial_speed_array_with_walls)
    h,w = inmap.altered_speed_array_with_walls.shape
    perlin_array = create_perlin_array(inmap.wall_octaves, h, w)
    inmap.wallflip_seg = segment_peaks(perlin_array)
    to_add, to_delete = pick_segs_to_flip_sat(inmap.altered_speed_array_with_walls, inmap.wallflip_seg, inmap.nadded_walls, inmap.ndeleted_walls)
    #print('to_add,to_delete',to_add,to_delete)
    add_walls_sat(inmap.altered_speed_array_with_walls,inmap.wallflip_seg,to_add)
    delete_walls_sat(inmap.altered_speed_array_with_walls,inmap.wallflip_seg,to_delete)


### Call this many times to get the SpeedData structure with apriori and randomized true maps (same structure as before)
def get_altered_satellite_map(inmap):
    add_altered_sattelite_map(inmap)
    initial_speed_with_walls = np.array(inmap.initial_speed_array_with_walls, np.float32)
    altered_speed_with_walls = np.array(inmap.altered_speed_array_with_walls, np.float32)
    initial_dijkstra = dijkstra(initial_speed_with_walls,inmap.goal_rect,cell_size=10,delta=delta_8way)
    altered_dijkstra = dijkstra(altered_speed_with_walls,inmap.goal_rect,cell_size=10,delta=delta_8way)
    initial_obs = np.array(inmap.initial_speed_array_with_walls, np.bool)
    altered_obs = np.array(inmap.altered_speed_array_with_walls, np.bool)
    initial_speed = np.array(inmap.speed_array,np.float32)
    altered_speed = np.copy(initial_speed)
    heights_only= np.ones_like(inmap.initial_speed_array_with_walls,dtype=np.float32)
    heights_only *= 150
    speed_data = SpeedData(altered_obs,initial_obs,altered_speed,initial_speed,heights_only,heights_only,
                altered_speed_with_walls,altered_dijkstra.cost_array,initial_speed_with_walls,initial_dijkstra.cost_array,altered_dijkstra.traceback,inmap.goal_rect)
    return speed_data


def test_for_leaks(n=1000):
    for i in range(n):
        import os, psutil
        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, ' MB ')
        sm = make_speed_map()


class SpeedData:
    def __init__(self, true_obstacles, apriori_obstacles, true_speed_array, apriori_speed_array, true_height_array, 
                apriori_height_array,true_speed_with_walls,true_cost_array,apriori_speed_with_walls,apriori_cost_array,traceback,goal_rect):
        self.true_obstacles = true_obstacles
        self.apriori_obstacles = apriori_obstacles
        self.true_speed_array = true_speed_array
        self.apriori_speed_array = apriori_speed_array
        self.true_height_array = true_height_array
        self.apriori_height_array = apriori_height_array
        self.true_speed_with_walls = true_speed_with_walls
        self.true_cost_array = true_cost_array
        self.apriori_speed_with_walls = apriori_speed_with_walls
        self.apriori_cost_array = apriori_cost_array
        self.traceback = traceback
        self.goal_rect = goal_rect


def fastest_path(speed_data,point=(138,128)):
    x,y = point
    path_cost = speed_data.true_cost_array[y,x]
    path = []
    while point:
        path.append(point)
        point = speed_data.traceback.get(point)
    return path_cost,path

delta_4way = [(0,-1),(0,1),(-1,0),(1,0)]
delta_8way = [(0,-1),(0,1),(-1,0),(1,0),(-1,-1),(-1,1),(1,-1),(1,1)]


def dijkstra(speed_array,goal_rect=[10,10,20,20],cell_size=10,delta=delta_8way,limit=float('inf')):
    delta_length = [(x,y,math.sqrt(x**2+y**2)) for x,y in delta]
    h,w = speed_array.shape
    traceback = {}
    cost_array = np.zeros(speed_array.shape,dtype=np.float)  ## bug fix, not float32
    q = []
    x1,y1,x2,y2 = goal_rect
    for starty in range(y1,y2+1):
        for startx in range(x1,x2+1):
            speed = speed_array[starty,startx]
            if speed <= 0:                     ## bug fix, don't overwrite obstacles
                continue
            initial_cost = 1.0
            cost_array[starty,startx] = initial_cost
            initial_length = 0
            item = (initial_cost,startx,starty,initial_length)
            heappush(q,item)
    counter = 0
    while q:
        counter += 1
        if counter >= limit:
            break
        path_cost,x,y,path_length = heappop(q)
        for dx,dy,cell_length in delta_length:
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < w):
                continue
            if not (0 <= ny < h):
                continue
            if speed_array[ny,nx] <= 0:  # obstacle
                continue
            old_cost = cost_array[ny,nx]
            speed = speed_array[ny,nx]
            cell_cost = cell_size * cell_length / speed
            new_cost = path_cost + cell_cost
            if old_cost == 0 or new_cost < old_cost:
                cost_array[ny,nx] = new_cost
                traceback[(nx,ny)] = (x,y)
                item = (new_cost,nx,ny,path_length+1)
                heappush(q,item)
    o = Ob()
    o.goal_rect = goal_rect
    o.speed_array = speed_array
    o.cost_array = cost_array
    o.traceback = traceback
    o.counter = counter
    o.fast = 'Dijkstra'
    return o


def compute_current_cost_array(speed_array_with_obstacles,goal_rect,cell_size=10,delta=delta_8way):
    "Call this for making moves using repeated Dijkstra"
    result = dijkstra(speed_array_with_obstacles,goal_rect,cell_size,delta)
    return result.cost_array


def test_dijkstra(sm):
    t1 = time.time()
    sm.dij1 = dijkstra(sm.true_speed_with_walls,sm.goal_rect)
    t2 = time.time()
    sm.dij2 = dijkstra2(sm.true_speed_with_walls,sm.goal_rect)
    t3 = time.time()
    sm.time1 = t2 - t1
    sm.time2 = t3 - t2
    print('dij1 time =',sm.time1,'dij2 time =',sm.time2)
    sm.sub = sm.dij1.cost_array[0:20,0:20]


def dijkstra2(speed_array,goal_rect=[10,10,20,20],cell_size=10,delta=delta_8way,limit=float('inf')):
    delta_length = [(x,y,math.sqrt(x**2+y**2)) for x,y in delta]
    h,w = speed_array.shape
    traceback = {}
    cost_array = np.zeros(speed_array.shape,dtype=np.float)
    q = []
    x1,y1,x2,y2 = goal_rect
    back = None
    for starty in range(y1,y2+1):
        for startx in range(x1,x2+1):
            speed = speed_array[starty,startx]
            if speed <= 0:
                continue
            initial_cost = 1.0
            # cost_array[starty,startx] = initial_cost
            initial_length = 0
            item = (initial_cost,startx,starty,back,initial_length)
            heappush(q,item)
    counter = 0
    misses = []
    items = []
    while q:
        counter += 1
        if counter >= limit:
            break
        popped_item = heappop(q)
        items.append(popped_item)
        path_cost,x,y,back,path_length = popped_item
        old_cost = cost_array[y,x]
        if old_cost > 0:
            if (path_cost > old_cost):
                misses.append([counter,old_cost,path_cost,x,y,back,path_length])
            continue
        cost_array[y,x] = path_cost
        if back:
            traceback[(x,y)] = back
        for dx,dy,cell_length in delta_length:
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < w):
                continue
            if not (0 <= ny < h):
                continue
            if speed_array[ny,nx] <= 0:  # obstacle
                continue
            old_cost = cost_array[ny,nx]
            speed = speed_array[ny,nx]
            cell_cost = cell_size * cell_length / speed
            new_cost = path_cost + cell_cost
            item = (new_cost,nx,ny,(x,y),path_length+1)
            heappush(q,item)
            # if old_cost == 0 or new_cost < old_cost:
            #     cost_array[ny,nx] = new_cost
            #     traceback[(nx,ny)] = (x,y)
            #     item = (new_cost,nx,ny,path_length+1)
            #     heappush(q,item)
    o = Ob()
    o.goal_rect = goal_rect
    o.speed_array = speed_array
    o.cost_array = cost_array
    o.traceback = traceback
    o.counter = counter
    o.fast = 'Dijkstra2'
    o.misses = misses
    o.items = items
    return o


def point_in_rect(x,y,rect):
    x1,y1,x2,y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def find_a_path(speed_array,startxy,goal_rect,cell_size=10,limit=float('inf'),fast=True):
    delta = [(0,-1),(0,1),(-1,0),(1,0)]
    h,w = speed_array.shape
    traceback = {}
    cost_array = np.zeros(speed_array.shape)
    q = deque()
    startx,starty = startxy
    initial_cost = speed_array[starty,startx]
    cost_array[starty,startx] = initial_cost
    initial_length = 0
    item = (startx,starty,initial_cost,initial_length)
    q.append(item)
    gx = gy = None
    counter = 0
    while q:
        counter += 1
        if counter >= limit:
            break
        x,y,path_cost,path_length = q.popleft()
        for dx,dy in delta:
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < w):
                continue
            if not (0 <= ny < h):
                continue
            if speed_array[ny,nx] <= 0:  # obstacle
                continue
            if point_in_rect(nx,ny,goal_rect):
                traceback[(nx,ny)] = (x,y)
                gx,gy = nx,ny
                break
            old_cost = cost_array[ny,nx]
            if fast and old_cost > 0:
                continue
            speed = speed_array[ny,nx]
            cell_cost = cell_size / speed
            new_cost = path_cost + cell_cost
            if old_cost == 0 or new_cost < old_cost:
                cost_array[ny,nx] = new_cost
                traceback[(nx,ny)] = (x,y)
                item = (nx,ny,new_cost,path_length+1)
                q.append(item)
    o = Ob()
    o.startxy = startxy
    o.goal_rect = goal_rect
    o.speed_array = speed_array
    o.cost_array = cost_array
    o.traceback = traceback
    o.goalxy = (gx,gy)
    o.counter = counter
    o.path_cost = path_cost
    o.path_length = path_length
    o.fast = fast
    return o


def make_total_cost(speed_array,goal_rect=[10,10,20,20],cell_size=10,delta=delta_8way,limit=float('inf'),fast=True):
    delta_length = [(x,y,math.sqrt(x**2+y**2)) for x,y in delta]
    h,w = speed_array.shape
    traceback = {}
    cost_array = np.zeros(speed_array.shape,dtype=np.float32)
    q = deque()
    x1,y1,x2,y2 = goal_rect
    for starty in range(y1,y2+1):
        for startx in range(x1,x2+1):
            initial_cost = 1.0
            cost_array[starty,startx] = initial_cost
            initial_length = 0
            item = (starty,startx,initial_cost,initial_length)
            q.append(item)
    counter = 0
    while q:
        counter += 1
        if counter >= limit:
            break
        x,y,path_cost,path_length = q.popleft()
        for dx,dy,cell_length in delta_length:
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < w):
                continue
            if not (0 <= ny < h):
                continue
            if speed_array[ny,nx] <= 0:  # obstacle
                continue
            old_cost = cost_array[ny,nx]
            if fast and old_cost > 0:
                continue
            speed = speed_array[ny,nx]
            cell_cost = cell_size * cell_length / speed
            new_cost = path_cost + cell_cost
            if old_cost == 0 or new_cost < old_cost:
                cost_array[ny,nx] = new_cost
                traceback[(nx,ny)] = (x,y)
                item = (nx,ny,new_cost,path_length+1)
                q.append(item)
    o = Ob()
    o.goal_rect = goal_rect
    o.speed_array = speed_array
    o.cost_array = cost_array
    o.traceback = traceback
    o.counter = counter
    o.fast = 'Fast' if fast else 'Slow'
    return o


def get_path(result,point=(140,130),limit=10000):
    if point is None:
        point = result.goalxy
    path = []
    counter = 0
    while point is not None:
        counter += 1
        if counter > limit:
            break
        x,y = point
        cost = result.speed_array[y,x]
        path.append((point,cost))
        point = result.traceback.get(point)
    return path

