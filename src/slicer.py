#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
"""
This module contains some basic tool path
generating functions for rectangle layer
"""

# standard library
import math
import sys

# 3rd party library
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_path(path_lst, ox, oy, length, width, colors, plot_border=True,
              ax=False):
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
        plt.plot(xs, ys, 'k-')
    # 2. plot path defined by path_lst
    assert(len(path_lst) == len(colors))
    for path, color in zip(path_lst, colors):
        plt.scatter(path[0][0], path[1][0])
        plt.plot(path[0], path[1], color)
    return ax


def compute_rectangle_border(ox, oy, length, width, start_loc="LL"):
    """
    compute rectangle border

    Args:
        ox, oy: left bottom corner of the border
        length: length of border
        width: width of border
        start_loc: position of the first vertex

    Returns:
        (xs, ys): vertices on the border
    """
    order = ['LL', 'LR', 'UR', 'UL']
    xs = [0, length, length, 0]
    ys = [0, 0, width, width]
    assert(start_loc in order)
    index = order.index(start_loc)
    xs = xs[index:] + xs[:index]
    ys = ys[index:] + ys[:index]
    xs.append(xs[0])
    ys.append(ys[0])
    xs = [x + ox for x in xs]
    ys = [y + oy for y in ys]
    return (xs, ys)


def compute_raster_path2D(ox, oy, length, width, road_width, air_gap, angle,
                          start_loc='LL'):
    """
    create 2D path of 90 degree or 0 degree raster

    Args:
        ox, oy: left bottom corner of the border
        length: length of border
        width: width of border
        road_width: width of the road
        air_gap: air gap between adjacent roads
        angle: raster angle (only support 0 and 90)
        start_loc: LL, LR, UR, UL

    Returns:
        [(xs, ys)]: a path list
    """
    assert(start_loc in ['LL', 'LR', 'UL', 'UR'])
    assert(angle in [0, 90])
    x_start_func = min
    y_start_func = min
    if start_loc in ['LR', 'UR']:
        x_start_func = max
    if start_loc in ['UL', 'UR']:
        y_start_func = max

    xlims = [0.5 * road_width, length - 0.5 * road_width]
    ylims = [0.5 * road_width, width - 0.5 * road_width]

    if angle == 0:
        x1 = x_start_func(xlims)
        x2 = length - x1
        ystart = y_start_func(ylims)
        yend = width - y_start_func(0, width)
        if yend > ystart:
            step = road_width + air_gap
        else:
            step = -(road_width + air_gap)
        temp = np.arange(ystart, yend, step)
        ys = []
        for v in temp:
            ys.extend([v, v])
        xs = [x1]
        for i in range(1, len(temp) * 2):
            if ys[i] == ys[i - 1]:
                xs.append(x1 + x2 - xs[-1])
            else:
                xs.append(xs[-1])
    if angle == 90:
        y1 = y_start_func(ylims)
        y2 = width - y1
        xstart = x_start_func(xlims)
        xend = length - x_start_func(0, length)
        if xend > xstart:
            step = road_width + air_gap
        else:
            step = -(road_width + air_gap)
        temp = np.arange(xstart, xend, step)
        xs = []
        for v in temp:
            xs.extend([v, v])
        ys = [y1]
        for i in range(1, len(temp) * 2):
            if xs[i] == xs[i - 1]:
                ys.append(y1 + y2 - ys[-1])
            else:
                ys.append(ys[-1])
    # linear transform
    xs = [x + ox for x in xs]
    ys = [y + oy for y in ys]
    return [(xs, ys)]
