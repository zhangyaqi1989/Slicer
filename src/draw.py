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

# local library
from slicer import compute_rectangle_border


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


if __name__ == "__main__":
    print("Hello World")
