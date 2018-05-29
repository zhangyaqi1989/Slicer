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


def plot_path(points_lst, ox, oy, length, width, colors, plot_border=True):
    """ points = (xs, ys)
        points_lst = [points1, points2, ...]
        canvas is (ox, oy) --> (ox + length, oy + width)
    """
    # 1. plot rectangle border
    if plot_border:
        xs, ys = rectangle_border(ox, oy, length, width)
        plt.plot(xs, ys, 'k-')
    # 2. plot path defined by path_lst
    for points, color in zip(points_lst, colors):
        plt.scatter(points[0][0], points[1][0])
        plt.plot(points[0], points[1], color)
    # plt.show()


def rectangle_border(ox, oy, length, width, start_loc="LL"):
    """ return border points of the rectangle defined by (ox, oy) --> 
        (ox + length, oy + width)
    """
    order = ['LL', 'LR', 'UR', 'UL']
    xs = [0, length, length, 0]
    ys = [0, 0, width, width]
    index = order.index(start_loc)
    xs = xs[index:] + xs[:index]
    ys = ys[index:] + ys[:index]
    xs.append(xs[0])
    ys.append(ys[0])
    xs = [x + ox for x in xs]
    ys = [y + oy for y in ys]
    return (xs, ys)

