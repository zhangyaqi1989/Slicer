#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
"""
This module contains test functions for slicer
"""

# standard library
import sys


# 3rd party library
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# local library
from slicer import (compute_raster_path2D, compute_contour_path2D,
                    compute_path2D, compute_checkerboard2D,
                    )

from draw import plot_path, plot_checkerboard2D


def test_compute_raster_path2D():
    """Test compute_raster_path2D()"""
    ox = 1
    oy = 2
    length = 30
    width = 18
    road_width = 1
    angle = 0
    start_loc = 'LR'
    air_gap = 0  # 0.2*road_width
    path_lst = compute_raster_path2D(ox, oy, length, width, road_width,
                                     air_gap, angle, start_loc)
    colors = ['b-'] * len(path_lst)
    ax = plot_path(path_lst, ox, oy, length, width, colors)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def test_compute_contour_path2D():
    """Test compute_contour_path2D() """
    ox = 1
    oy = 2
    length = 30
    width = 18
    road_width = 1
    air_gap = 0  # 0.2*road_width
    num_contours = 2
    start_locs = ['LL', 'UR']
    contour_lst = compute_contour_path2D(ox, oy, length, width, road_width,
                                         air_gap, num_contours, start_locs)
    colors = ['b-'] * len(contour_lst)
    ax = plot_path(contour_lst, ox, oy, length, width, colors)
    plt.show()


def test_compute_path2D():
    ox = 1
    oy = 2
    length = 30
    width = 18
    road_width = 1
    angle = 0
    start_loc = 'LR'
    air_gap = 0  # 0.2*road_width
    contour_air_gap = 0
    raster_air_gap = 0
    num_contours = 3
    raster_start_loc = 'LR'
    contour_start_locs = ['LL'] * num_contours
    points_lst = compute_path2D(ox, oy, length, width, road_width,
                                contour_air_gap, raster_air_gap, num_contours,
                                contour_start_locs, raster_start_loc, angle)
    colors = ['r-'] * num_contours + ['b-']
    ax = plot_path(points_lst, ox, oy, length, width, colors)
    plt.show()


def test_compute_checkerboard2D():
    ox = 1
    oy = 2
    grid_length = 30
    grid_width = 18
    nrows = 3
    ncols = 4
    road_width = 1
    contour_air_gap = 0
    raster_air_gap = 0
    num_contours = 3
    length = ncols * grid_length
    width = nrows * grid_width
    checker_lst = compute_checkerboard2D(ox, oy, grid_length, grid_width,
                                         nrows, ncols, num_contours, road_width, contour_air_gap,
                                         raster_air_gap)
    colors = ['r-'] * num_contours + ['b-']
    colors_lst = []
    for i in range(len(checker_lst)):
        colors_lst.append(colors)
    checker_plot_lst = [True] * len(checker_lst)
    # checker_plot_lst = np.random.choice([True, False], len(checker_lst))
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_checkerboard2D(checker_lst, ox, oy, length, width, grid_length,
                        grid_width, colors_lst, checker_plot_lst, ax)
    plt.show()


def main():
    """command line runner"""
    # funcs_names = [item for item in dir() if item.startswith('test_')]
    funcs_names = ['test_compute_raster_path2D',
                   'test_compute_contour_path2D',
                   'test_compute_path2D',
                   'test_compute_checkerboard2D',
                   ]
    funcs = [globals()[item] for item in funcs_names]
    if len(sys.argv) != 2:
        print("Usage: >> python {} <{}-{}>".format(sys.argv[0], 0,
                                                   len(funcs_names) - 1))
        sys.exit(1)
    n = int(sys.argv[1])
    assert(0 <= n < len(funcs_names))
    funcs[n]()


if __name__ == "__main__":
    main()
