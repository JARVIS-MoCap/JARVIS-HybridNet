"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import numpy as np
import matplotlib

from jarvis.config.project_manager import ProjectManager

def get_skeleton(cfg):
    if len(cfg.SKELETON) > 0:
        base_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
                    (255,0,255), (0,255,255), (0,140,255), (140,255,0),
                    (255,140,0), (0,255,140), (255,140,140), (140,255,140),
                    (140,140,255), (140,140,140)]
        gray_color = (100,100,100)
        color_idx = 0
        colors = []
        connections = np.zeros(len(cfg.KEYPOINT_NAMES), dtype=int)
        for keypoint in cfg.KEYPOINT_NAMES:
            colors.append(gray_color)

        line_idxs = []
        starting_idxs = []

        for bone in cfg.SKELETON:
            index_start = cfg.KEYPOINT_NAMES.index(bone[0])
            starting_idxs.append(index_start)
            index_stop = cfg.KEYPOINT_NAMES.index(bone[1])
            line_idxs.append([index_start, index_stop])
            connections[index_start] += 1
            connections[index_stop] += 1

        seeds = np.nonzero(connections == 1)[0]

        unconnected = np.nonzero(connections == 0)[0]
        graph = Graph(line_idxs)
        cycles = graph.get_cycles()

        accounted_for = []

        for cycle in cycles:
            for point in cycle:
                colors[point] = base_colors[color_idx]
            color_idx = (color_idx + 1) % len(base_colors)

        for seed in seeds:
            if seed in starting_idxs:
                idx = seed
                colors[idx] = base_colors[color_idx]
                accounted_for.append(idx)
                conn_idxs = [line[1] for line in line_idxs if line[0] == idx]
                backward_idx = [line[0] for line in line_idxs if line[1] == idx]
                while len(conn_idxs) == 1 and len(backward_idx) < 2:
                    idx = conn_idxs[0]
                    if connections[idx] < 3 or part_of_cycle(cycles, idx):
                        if idx in accounted_for:
                            colors[idx] = gray_color
                        else:
                            colors[idx] = base_colors[color_idx]
                            accounted_for.append(idx)
                    conn_idxs = [line[1] for line in line_idxs if line[0] == idx]
                    backward_idx = [line[0] for line in line_idxs if line[1] == idx]
                color_idx = (color_idx + 1) % len(base_colors)

        for point in unconnected:
            colors[point] = base_colors[color_idx]
            color_idx = (color_idx + 1) % len(base_colors)

    else:
        colors = []
        line_idxs = []
        cmap = matplotlib.cm.get_cmap('jet')
        for i in range(cfg.KEYPOINTDETECT.NUM_JOINTS):
            colors.append(((np.array(
                    cmap(float(i)/cfg.KEYPOINTDETECT.NUM_JOINTS)) *
                    255).astype(int)[:3]).tolist())

    return colors, line_idxs


def part_of_cycle(cycles, idx):
    for cycle in cycles:
        if idx in cycle:
            return True
    return False


class Graph:
    def __init__(self,graph):
        self.graph = graph
        self.cycles = []
        self.max_len = 0

    def get_cycles(self):
        for edge in self.graph:
            for node in edge:
                self.findNewCycles([node])
        return self.cycles

    def findNewCycles(self, path):
        start_node = path[0]
        next_node= None
        sub = []
        #visit each edge and each node of each edge
        for edge in self.graph:
            node1, node2 = edge
            if start_node in edge:
                    if node1 == start_node:
                        next_node = node2
                    else:
                        next_node = node1
                    if not self.visited(next_node, path):
                            # neighbor node not on path yet
                            sub = [next_node]
                            sub.extend(path)
                            # explore extended path
                            self.findNewCycles(sub);
                    elif len(path) > 2  and next_node == path[-1]:
                            # cycle found
                            p = self.rotate_to_smallest(path);
                            inv = self.invert(p)
                            if self.isNew(p) and self.isNew(inv):
                                overlaps = self.overlapping(p)
                                if len(overlaps) > 0:
                                    max_len = 0
                                    for overlap in overlaps:
                                        if len(overlap) > max_len:
                                            max_len = len(overlap)
                                    if len(p) > max_len:
                                        self.cycles.append(p)
                                        for overlap in overlaps:
                                            self.cycles.remove(overlap)
                                else:
                                    self.cycles.append(p)

    def invert(self,path):
        return self.rotate_to_smallest(path[::-1])

    def rotate_to_smallest(self,path):
        n = path.index(min(path))
        return path[n:]+path[:n]

    def isNew(self,path):
        return not path in self.cycles

    def overlapping(self, path):
        overlaps = []
        for cycle in self.cycles:
            for point in path:
                if point in cycle:
                    overlaps.append(cycle)
                    break
        return overlaps

    def visited(self,node, path):
        return node in path
