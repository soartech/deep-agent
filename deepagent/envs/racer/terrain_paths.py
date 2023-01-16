import collections
import copy
import sys
from enum import Enum
from typing import List, Dict, Tuple, Hashable, Any

import networkx
import networkx as nx
import numpy as np
from networkx import Graph, shortest_path
from rtree import index
from skimage.morphology import skeletonize
from sknw import sknw

from deepagent.envs.racer.shapes import Point, Route


class GraphSimplificationMethod(Enum):
    ONE = 1
    TWO = 2


NodeId = int
Edge = Tuple[NodeId, NodeId, Dict]


class TerrainPaths:
    """
    Uses skeletonization to create paths through different height levels of terrain images or between the walls of a
    traversability map.
    """

    def __init__(self, terrain: np.ndarray, num_levels: int = 2,
                 method: GraphSimplificationMethod = GraphSimplificationMethod.TWO):
        """
        @param terrain: A numpy array of a terrain image.
        @param num_levels: The number of height levels to make. The algorithm groups the height values in the image by
            frequency and then will consider the num_levels most frequent values as the height levels to make paths for.
        @param method: Which graph simplification method to use. TWO works better.
        """
        self._terrain = terrain
        hist = hister(flatten(self._terrain))
        self._levels = [x[0] for x in hist[0:num_levels]]
        self._levels.sort()
        self._level_images = []
        self._skel_images = []
        self._graphs = []
        self._method = method
        for level in self._levels[1:]:
            img = (self._terrain == level).astype(np.uint)
            self._level_images.append(img)
            skel = skeletonize(img).astype(np.uint16)
            self._skel_images.append(skel)
            graph = sknw.build_sknw(skel)
            self._graphs.append(graph)

        self._rems = []
        self._prune_skels = []
        self._simple_graphs = []  # type: List[Graph]
        self._simple_skels = []
        self._prune_terrain()
        self._simple_rtrees = self._create_rtrees()
        self.temp_node_id = "temp"

    def _prune_terrain(self):
        nlevels = len(self._level_images)
        for i in range(nlevels):
            skel = self._skel_images[i]
            graph = self._graphs[i]
            if self._method == GraphSimplificationMethod.ONE:
                rems = simplify_graph(graph)
                oskel = remove_branches(skel, graph, rems, 0)
            else:  # method == TWO
                rems = prune_leaves(graph)
                oskel = remove_paths(skel, graph, rems, 0)
            simpskel = skeletonize(oskel).astype(np.uint16)
            simpgraph = sknw.build_sknw(simpskel)
            self._rems.append(rems)
            self._prune_skels.append(oskel)
            self._simple_skels.append(simpskel)
            self._simple_graphs.append(simpgraph)
        self._paints = [self._paint_edges(x)[0] for x in range(nlevels)]
        self._zones = [self._fill_edges(x) for x in range(nlevels)]

    def _paint_edges(self, n=0, start=10):
        plist = []
        img = np.copy(self._level_images[n])
        g = self._simple_graphs[n]
        eds = list(g.edges())
        for i, edge in enumerate(eds):
            val = i + start
            points = paint_edge(g, edge, img, val)
            for pt in points:
                plist.append((pt, val))
        return img, plist

    def _fill_edges(self, n=0):
        img, plist = self._paint_edges(n)
        q = collections.deque(plist)
        while q:
            pt, val = q.popleft()
            y, x = pt
            neighbors = [(y, x + 1), (y, x - 1), (y + 1, x), (y - 1, x)]
            for p1 in neighbors:
                yy, xx = p1
                if yy >= img.shape[0] or xx >= img.shape[1]:
                    continue
                v = img[yy, xx]
                if v == 1:
                    img[yy, xx] = val
                    q.append((p1, val))
        return img

    def get_level(self, n):
        return self._level_images[n]

    def get_level_height(self, n):
        return self._levels[n + 1]

    def get_skel(self, n):
        return self._skel_images[n]

    def get_zones(self, n):
        return self._zones[n]

    def get_graph(self, n):
        return self._graphs[n]

    def get_simple_skel(self, n):
        return self._simple_skels[n]

    def get_simple_graph(self, n):
        return self._simple_graphs[n]

    def get_paint(self, n):
        return self._paints[n]

    def get_closest_node(self, point: Point):
        graph = self._simple_graphs[0]
        min_dist = sys.maxsize
        closest_node = None
        for node_num, node_data, in graph.nodes(data=True):
            node_centroid = Point(node_data['o'][1], node_data['o'][0])
            dist = point.squared_distance_to(node_centroid)
            if dist < min_dist:
                min_dist = dist
                closest_node = (node_num, node_data)
        return closest_node

    def get_closest_n(self, point: Point, n: int = 2, i=0):
        """
        Finds the closest edge or node and returns information about it.
        @param point: any point
        @param n: the number of closest objects to return.
        @param i: The simple rtree floor layer to use.
        @return: A list of closest edges, each edge object has an extra 'pt_idx' field saying which pt in the 'pts' array is the closest.
        """
        rtree = self._simple_rtrees[i]
        return list(rtree.nearest([point.y, point.x, point.y, point.x], num_results=n, bbox=False))

    @staticmethod
    def is_edge(edge_or_node):
        return len(edge_or_node) == 3

    def add_temp_node_to_graph(self, start_point: Point):
        graph = self._simple_graphs[0]
        closest = self.get_closest_n(start_point)

        graph.add_node(self.temp_node_id, pts=[], o=[start_point.y, start_point.x])

        for closest_edge in closest:
            node_a_id = closest_edge[0]
            node_a = graph.nodes[node_a_id]
            node_a_pt = Point(node_a['o'][1], node_a['o'][0])

            node_b_id = closest_edge[1]
            node_b = graph.nodes[node_b_id]
            node_b_pt = Point(node_b['o'][1], node_b['o'][0])

            pt_array = closest_edge[2]['pts']
            pt_idx = closest_edge[2]['pt_idx']

            last_point = Point(pt_array[-1][1], pt_array[-1][0])
            if node_a_pt.squared_distance_to(last_point) < node_b_pt.squared_distance_to(last_point):
                pt_array = np.flip(pt_array)
                pt_idx = len(pt_array) - pt_idx - 1

            closest_pt = Point(pt_array[pt_idx][1], pt_array[pt_idx][0])
            closest_pt_minus1 = Point(pt_array[pt_idx - 1][1],
                                      pt_array[pt_idx - 1][0]) if pt_idx - 1 >= 0 else node_a_pt

            if closest_pt_minus1.squared_distance_to(closest_pt) > closest_pt_minus1.squared_distance_to(start_point):
                split_index = pt_idx
            else:
                split_index = pt_idx + 1

            edge_points_to_node_a = pt_array[:split_index]

            if len(edge_points_to_node_a > 0):
                pt = edge_points_to_node_a[-1]
                dist = start_point.distance_to(Point(pt[1], pt[0]))
            else:
                dist = start_point.distance_to(node_a_pt)

            dist_to_node_a = split_index + dist
            graph.add_edge(node_a_id, self.temp_node_id, pts=edge_points_to_node_a, weight=dist_to_node_a)

            node_b_id = closest_edge[1]
            edge_points_to_node_b = pt_array[split_index:]
            if len(edge_points_to_node_b > 0):
                pt = edge_points_to_node_b[0]
                dist = start_point.distance_to(Point(pt[1], pt[0]))
            else:
                dist = start_point.distance_to(node_b_pt)

            dist_to_node_b = pt_array.shape[0] - split_index + dist
            # Only add the shorter of the two edges if we're inserting the temp node on an edge that is a self loop
            if node_a_id != node_b_id or dist_to_node_b < dist_to_node_a:
                graph.add_edge(self.temp_node_id, node_b_id, pts=pt_array[split_index:], weight=dist_to_node_b)

    def get_shortest_path(self, start_point: Point, end_node: Tuple[Hashable, Any], num_points: int = 6) -> Route:
        """
        Finds the shortest path from the start_point to the end_node.
        @param start_point: Any point.
        @param end_node: The goal node.
        @param num_points: The desired number of points. The length of the path will be:
            less then num_points if length from start_point to end_node is less than num_points, else num_points

        @return: A list of pts towards the end node from the start point, the length of the path, the next node in the path
        """
        graph = self._simple_graphs[0]

        try:
            shortest = shortest_path(graph, self.temp_node_id, end_node[0], weight='weight')
        except networkx.exception.NetworkXNoPath as e:
            # no path to target
            path = None
            length = 1000.0
            next_node_position = (start_point.x, start_point.y)
            return path, length, next_node_position

        shortest = [(node_id, graph.nodes[node_id]) for node_id in shortest]
        length = 0.0
        for i in range(len(shortest) - 1):
            node = shortest[i]
            next_node = shortest[i + 1]
            edge_data = graph.get_edge_data(node[0], next_node[0])
            length += edge_data['weight']

        pts = []
        for i in range(len(shortest) - 1):
            node = shortest[i]
            next_node = shortest[i + 1]

            start_node_pt = Point(node[1]['o'][1], node[1]['o'][0])
            end_node_pt = Point(next_node[1]['o'][1], next_node[1]['o'][0])

            edge_data = graph.get_edge_data(node[0], next_node[0])
            edge_pts = [Point(ep[1], ep[0]) for ep in edge_data['pts']]
            if len(edge_pts) > 0 and edge_pts[-1].squared_distance_to(start_node_pt) < edge_pts[-1].squared_distance_to(
                    end_node_pt):
                edge_pts = reversed(edge_pts)

            for edge_pt in edge_pts:
                pts.append(edge_pt)
                if len(pts) > num_points:
                    break

            pts.append(end_node_pt)
            if len(pts) > num_points:
                break

        # attempt to fix oscillations back and forth over the starting point
        if len(pts) > 2:
            pts = pts[2:]
        if len(shortest) > 1:
            next_node = (shortest[1][1]['o'][0], shortest[1][1]['o'][1])
        else:
            next_node = (start_point.x, start_point.y)

        if len(pts) == 0:
            pts = None

        return pts, length, next_node

    def remove_temp_node(self):
        graph = self._simple_graphs[0]
        graph.remove_node(self.temp_node_id)

    def _create_rtrees(self):
        rtrees = []
        for graph in self._simple_graphs:
            rtree = index.RtreeContainer()
            for edge in graph.edges(data=True):
                for i, point in enumerate(edge[2]['pts']):
                    edge_copy = copy.deepcopy(edge)
                    edge_copy[2]['pt_idx'] = i
                    rtree.insert(edge_copy, np.concatenate([point, point]))
            rtrees.append(rtree)
        return rtrees

    def get_neighbors(self, node, level=0):
        graph = self.get_simple_graph(level)
        return [graph.nodes[node_id] for node_id in graph.neighbors(node[0])]

    def get_neighbors_tuple(self, node):
        graph = self.get_simple_graph(0)
        return [(node_id, graph.nodes[node_id]) for node_id in graph.neighbors(node[0])]


def prune_node(g: nx.Graph, d, node, alist, verbose=False):
    # we assume that incoming node is not a leaf
    leafs = []
    inners = []
    to_remove = []
    for x in alist:
        if len(d[x]) == 1:
            leafs.append(x)
        else:
            inners.append(x)
    weights = []
    if verbose:
        print('nleafs', len(leafs), 'ninner', len(inners))
    for a in leafs:
        w = g[node][a]['weight']
        weights.append([w, a])
    weights.sort()
    if len(inners) == 0:
        # everything is a leaf, keep 2
        for w, a in weights[:-2]:
            leaf = a
            to_remove.append([leaf, (node, a)])
    elif len(inners) == 1:
        # one inner neighbor, keep 1 leaf
        for w, a in weights[:-1]:
            leaf = a
            to_remove.append([leaf, (node, a)])
    elif len(inners) > 1:
        # multiple inner neighbors, remove all leaves
        for w, a in weights:
            leaf = a
            to_remove.append([leaf, (node, a)])
    if verbose:
        print('removing', len(to_remove))
    return to_remove


def simplify_graph(g, verbose=False):
    d = nx.to_dict_of_lists(g)  # d: node -> list_of_adjacent_nodes
    to_remove = []
    for n, alist in d.items():
        if len(alist) > 1:
            rems = prune_node(g, d, n, alist, verbose)
            to_remove.extend(rems)
    return to_remove


def get_leaf_nodelist(n, d):
    nodes = []
    while True:
        nodes.append(n)
        alist = d[n]
        num = len(alist)
        if num == 1:
            if len(nodes) > 1:
                return None  # ended at another leaf
            n = alist[0]
        if num == 2:
            prev = nodes[-2]
            if alist[0] == prev:
                n = alist[1]
            else:
                assert (alist[1] == prev)
                n = alist[0]
        elif num > 2:
            # terminus
            return nodes


def path_from_nodelist(nodes):
    path = []
    for i in range(len(nodes) - 1):
        path.append((nodes[i], nodes[i + 1]))
    return path


def get_d(graph):
    d = nx.to_dict_of_lists(graph)  # d: node -> list_of_adjacent_nodes
    return d


def get_path_weight(path, graph):
    total = 0
    for (s, e) in path:
        w = graph[s][e]['weight']
        total += w
    return total


def select_removals(nodelists, g, thresh):
    rems = []
    for nodelist in nodelists:
        path = path_from_nodelist(nodelist)
        weight = get_path_weight(path, g)
        if weight <= thresh:
            rems.append(path)
    return rems


def prune_leaves(g, thresh=15):
    d = get_d(g)  # d: node -> list_of_adjacent_nodes
    branches = {}
    for n, alist in d.items():
        if len(alist) == 1:
            nodelist = get_leaf_nodelist(n, d)
            if nodelist:
                terminus = nodelist[-1]
                nodelists = branches.get(terminus, [])
                nodelists.append(nodelist)
                branches[terminus] = nodelists
    to_remove = []
    for terminus, nodelists in branches.items():
        rems = select_removals(nodelists, g, thresh)
        to_remove.extend(rems)
    return to_remove


def remove_branches(skel, graph, to_remove, val=0):
    oskel = np.copy(skel)
    for leaf, edge in to_remove:
        s, e = edge
        epoints = graph[s][e]['pts']
        for point in epoints:
            y, x = point
            oskel[y, x] = val
        npoints = graph.nodes()[s]['pts']
        for point in npoints:
            y, x = point
            oskel[y, x] = 1
    return oskel


def remove_paths(skel, graph, to_remove, val=0):
    oskel = np.copy(skel)
    for path in to_remove:
        nedges = len(path)
        for i, edge in enumerate(path):
            s, e = edge
            epoints = graph[s][e]['pts']
            for point in epoints:
                y, x = point
                oskel[y, x] = val
            nval = 0
            if i == nedges - 1:
                nval = 1
            npoints = graph.nodes()[e]['pts']
            for point in npoints:
                y, x = point
                oskel[y, x] = nval
    return oskel


def paint_edge(graph, edge, img, val):
    s, e = edge
    epoints = graph[s][e]['pts']
    for point in epoints:
        y, x = point
        img[y, x] = val
    return epoints


def histit(vec, ht=None, key=lambda x: x):
    """histogram of vec returned as a hashtable -- add to existing if ht is supplied"""
    if ht is None:
        out = {}
    else:
        out = ht
    for x in vec:
        out[key(x)] = 1 + out.get(key(x), 0)
    return out


def hister(vec, ht=None, key=lambda x: x):
    """histogram of vec returned as pairs sorted in decreasing frequency"""
    items = list(histit(vec, ht, key).items())
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def flatten(lists):
    """Combine list of lists into one big list."""
    out = []
    for x in lists:
        out.extend(x)
    return out
