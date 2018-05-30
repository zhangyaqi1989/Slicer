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

# third party library
import numpy as np
import matplotlib.pyplot as plt

# local library
from slicer import compute_raster_path2D, plot_path


def test_compute_raster_path2D():
    ox = 1
    oy = 2
    length = 30
    width = 18
    road_width = 1
    angle = 0
    start_loc = 'LR'
    air_gap = 0  # 0.2*road_width
    path_lst = compute_raster_path2D(ox, oy, length, width, road_width, air_gap,
                                   angle, start_loc)
    colors = ['b-'] * len(path_lst)
    ax = plot_path(path_lst, ox, oy, length, width, colors)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


if __name__ == "__main__":
    test_compute_raster_path2D()
