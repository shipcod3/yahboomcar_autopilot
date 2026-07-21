"""
Frontier-detection helpers used by roam_node to autonomously explore
and map the surroundings (Tesla-style "self map" behaviour).

A frontier cell is a FREE cell that has at least one UNKNOWN neighbour.
Frontier cells are grouped into clusters (simple flood fill / BFS) and
each cluster is reduced to its centroid in world coordinates. The
caller then picks the "best" centroid (usually nearest large frontier)
and sends it to Nav2 as the next exploration goal.
"""

import math
from collections import deque

import numpy as np

UNKNOWN = -1
FREE_MAX = 49          # cells with value 0..49 are considered free/traversable
OCC_MIN = 65            # cells with value >= 65 are considered occupied


def grid_to_world(mx, my, origin_x, origin_y, resolution):
    wx = origin_x + (mx + 0.5) * resolution
    wy = origin_y + (my + 0.5) * resolution
    return wx, wy


def world_to_grid(wx, wy, origin_x, origin_y, resolution):
    mx = int((wx - origin_x) / resolution)
    my = int((wy - origin_y) / resolution)
    return mx, my


def find_frontiers(grid, width, height, origin_x, origin_y, resolution,
                    min_cluster_size=6):
    """
    grid: 1D list/array (row major, size = width*height) of int8 occupancy values
    Returns list of dicts: {'x':world_x, 'y':world_y, 'size':cluster_size}
    """
    arr = np.asarray(grid, dtype=np.int16).reshape((height, width))

    is_free = (arr >= 0) & (arr <= FREE_MAX)
    is_unknown = (arr == UNKNOWN)

    # A free cell touching an unknown cell (4-neighbourhood) is a frontier cell
    frontier_mask = np.zeros_like(is_free, dtype=bool)
    frontier_mask[1:, :] |= is_free[1:, :] & is_unknown[:-1, :]
    frontier_mask[:-1, :] |= is_free[:-1, :] & is_unknown[1:, :]
    frontier_mask[:, 1:] |= is_free[:, 1:] & is_unknown[:, :-1]
    frontier_mask[:, :-1] |= is_free[:, :-1] & is_unknown[:, 1:]

    visited = np.zeros_like(frontier_mask, dtype=bool)
    clusters = []

    ys, xs = np.nonzero(frontier_mask)
    frontier_cells = set(zip(ys.tolist(), xs.tolist()))

    for (y0, x0) in list(frontier_cells):
        if visited[y0, x0]:
            continue
        # BFS to gather the connected cluster
        q = deque([(y0, x0)])
        visited[y0, x0] = True
        cluster_pts = []
        while q:
            y, x = q.popleft()
            cluster_pts.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if frontier_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))

        if len(cluster_pts) < min_cluster_size:
            continue

        mean_y = sum(p[0] for p in cluster_pts) / len(cluster_pts)
        mean_x = sum(p[1] for p in cluster_pts) / len(cluster_pts)
        wx, wy = grid_to_world(mean_x, mean_y, origin_x, origin_y, resolution)
        clusters.append({'x': wx, 'y': wy, 'size': len(cluster_pts)})

    return clusters


def pick_best_frontier(clusters, robot_x, robot_y, blacklist, blacklist_radius=0.6,
                        min_travel=0.4):
    """
    Choose the frontier with the best score = size / (distance^0.6)
    while ignoring points close to a previously-failed goal (blacklist)
    or too close to the robot itself.
    """
    best = None
    best_score = -1.0
    for c in clusters:
        dx = c['x'] - robot_x
        dy = c['y'] - robot_y
        dist = math.hypot(dx, dy)
        if dist < min_travel:
            continue

        skip = False
        for bx, by in blacklist:
            if math.hypot(c['x'] - bx, c['y'] - by) < blacklist_radius:
                skip = True
                break
        if skip:
            continue

        score = c['size'] / (dist ** 0.6)
        if score > best_score:
            best_score = score
            best = c

    return best
