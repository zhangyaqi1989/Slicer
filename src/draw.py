#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
"""
This module defines some plot functions
"""


# 3rd party library
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# local library
from slicer import compute_rectangle_border

# use seaborn plot style
sns.set()


def plot_path(path_lst, ox, oy, length, width, colors, plot_border=True,
              ax=None):
    """
    plot list of subpaths and rectangle border is defined by length X width
    The left bottom corner of the border is @ (ox, oy)

    Args:
        path_lst: [path1, path2, ...], path = (xs, ys)
        ox, oy: left bottom corner of the border
        length: length of border
        width: width of border
        colors: colors of each subpath
        plot_border: plot border or not
        ax: axes object

    Returns:
        ax: an axes object
    """
    # 0. create axes
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))
    # 1. plot rectangle border
    if plot_border:
        xs, ys = compute_rectangle_border(ox, oy, length, width)
        ax.plot(xs, ys, 'k-')
    # 2. plot path defined by path_lst
    assert(len(path_lst) == len(colors))
    for path, color in zip(path_lst, colors):
        ax.scatter(path[0][0], path[1][0])
        ax.plot(path[0], path[1], color)
    return ax


def plot_checkerboard2D(checker_lst, ox, oy, length, width, grid_length,
                        grid_width, colors_lst, checker_plot_lst, ax=None):
    """
    plot 2D checkerboard2D pattern

    Args:
        ox, oy: left bottom corner of the border
        length: length of border
        width: width of border
        grid_length: length of grid
        grid_width: width of grid
        colors_lst: color of each path
        checker_plot_lst: whether plot each checker or not
        ax: axes object

    Returns:
        ax: an axes object
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))
    nrows = width // grid_width
    ncols = length // grid_length
    for row in range(nrows):
        for col in range(ncols):
            curr_x = ox + col * grid_length
            curr_y = oy + row * grid_width
            idx = col + row * ncols
            if not checker_plot_lst[idx]:
                continue
            checker = checker_lst[idx]
            colors = colors_lst[idx]
            plot_path(checker, curr_x, curr_y, grid_length,
                      grid_width, colors, plot_border=True, ax=ax)


def plot_gcode_roads3D(road_lst):
    """
    plot road list read from Gcode file

    Args:
        roads: road list

    Returns:
        None
    """
    n = len(road_lst)
    print("Number of roads: " + str(n))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    road = (0, 0, 0, 0, 0, 0)
    xs = []
    ys = []
    zs = []
    k = 0.1
    segs = -1
    tmp = []
    for i in range(n):
        if i > k * n:
            print(str(int(100 * k)) + "%")
            k += 0.1
        old_road = road
        road = road_lst[i]
        if road[0] != old_road[2] or road[1] != old_road[3] or road[4] != old_road[4]:
            segs += 1
            if len(xs) >= 1:
                xs.append(old_road[2])
                ys.append(old_road[3])
                zs.append(old_road[4])
                ax.plot(xs, ys, zs)
                ax.scatter(xs[0], ys[0], zs[0])
                plt.draw()
            xs = [road[0]]
            ys = [road[1]]
            zs = [road[4]]
        else:
            xs.append(road[0])
            ys.append(road[1])
            zs.append(road[4])
    if len(xs) != 0:
        xs.append(road[2])
        ys.append(road[3])
        zs.append(road[4])
        ax.plot(xs, ys, zs)
        ax.scatter(xs[0], ys[0], zs[0])
        plt.draw()
    print("100%")


if __name__ == "__main__":
    print("Hello World")
