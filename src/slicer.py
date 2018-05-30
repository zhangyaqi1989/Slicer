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
    create 2D raster path of 90 degree or 0 degree on a rectangle layer

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


def compute_contour_path2D(ox, oy, length, width, road_width, air_gap,
                           num_contours, start_locs):
    """
    compute 2D contour path of a rectangle layer

    Args:
        ox, oy: left bottom corner of the border
        length: length of border
        width: width of border
        road_width: width of the road
        air_gap: air gap between adjacent contours
        num_contours: number of contours
        start_locs: defines start position of each countour
                    each start_loc is in [LL, LR, UR, UL]

    Returns:
        contour list
    """
    assert(len(start_locs) == num_contours)
    contour_lst = []
    curr_x = ox + 0.5 * road_width
    curr_y = oy + 0.5 * road_width
    curr_length = length - road_width
    curr_width = width - road_width
    gap = road_width + air_gap
    for i in range(num_contours):
        start_loc = start_locs[i]
        contour = compute_rectangle_border(curr_x, curr_y, curr_length,
                                           curr_width, start_loc)
        contour_lst.append(contour)
        # update curr_x, curr_y, curr_length and curr_width
        curr_x += gap
        curr_y += gap
        assert(curr_length >= 2 * gap)
        assert(curr_width >= 2 * gap)
        curr_length -= 2 * gap
        curr_width -= 2 * gap
    return contour_lst


def compute_path2D(ox, oy, length, width, road_width, contour_air_gap,
                   raster_air_gap, num_contours, contour_start_locs,
                   raster_start_loc, angle):
    """
    create 2D contour + raster path of rectangle layer

    Args:
        ox, oy: left bottom corner of the border
        length: length of border
        width: width of border
        road_width: width of the road
        contour_air_gap: air gap between adjacent contours
        raster_air_gap: air gap between adjacent roads
        num_contours: number of contours
        contour_start_locs: defines start position of each contour
                    each start_loc is in [LL, LR, UR, UL]
        raster_start_loc: LL, LR, UR, UL
        angle: raster angle (only support 0 and 90)

    Returns:
        path list
    """
    path_lst = compute_contour_path2D(ox, oy, length, width, road_width,
                                      contour_air_gap, num_contours, contour_start_locs)
    gap = road_width + contour_air_gap
    # this update is questionable
    raster_ox = ox + 0.5 * road_width + num_contours * gap
    raster_oy = oy + 0.5 * road_width + num_contours * gap
    raster_length = length - road_width - num_contours * 2 * gap
    raster_width = width - road_width - num_contours * 2 * gap
    path_lst.extend(compute_raster_path2D(raster_ox, raster_oy, raster_length,
                                          raster_width, road_width, raster_air_gap, angle, raster_start_loc))
    return path_lst


def _rand_start_locs(n):
    """
    generate start_locs randomly

    Args:
        n: number of start_locs

    Returns:
        randomly picked start_locs
    """
    return np.random.choice(['LL', 'LR', 'UL', 'UR'], n, replace=True)


def compute_checkerboard2D(ox, oy, grid_length, grid_width, nrows, ncols,
                           num_contours, road_width, contour_air_gap, raster_air_gap):
    """
    compute 2D checkerboard path (contour + raster) of a rectangle layer

    Args:
        ox, oy: left bottom corner of the border
        grid_length: length of grid
        grid_width: width of grid
        nrows: number of cells in row
        ncols: number of cells in column
        num_contours: number of contours
        road_width: width of the road
        contour_air_gap: air gap between adjacent contours
        raster_air_gap: air gap between adjacent roads

    Returns:
        path list
    """
    num_checkers = nrows * ncols
    length = grid_length * ncols
    width = grid_width * nrows
    # # fixed value
    # raster_start_loc_lst = ['LR']*num_checkers
    # contour_start_locs = ['LL']*num_contours
    # angle_lst = [0]*num_checkers
    # # random pick
    raster_start_loc_lst = _rand_start_locs(num_checkers)
    contour_start_locs_lst = []
    angle_lst = np.random.choice([0, 90], num_checkers, replace=True)
    for i in range(num_checkers):
        contour_start_locs_lst.append(_rand_start_locs(num_contours))
    checker_lst = _make_checkerboard2D(ox, oy, length, width, road_width,
                                       contour_air_gap, raster_air_gap,
                                       num_contours, contour_start_locs_lst, raster_start_loc_lst,
                                       angle_lst, grid_length, grid_width)
    return checker_lst


def _make_checkerboard2D(ox, oy, length, width, road_width, contour_air_gap,
                         raster_air_gap, num_contours, contour_start_locs_lst,
                         raster_start_loc_lst, angle_lst, grid_length, grid_width):
    """
    make 2D checkerboard path (contour + raster) of a rectangle layer

    Args:
        ox, oy: left bottom corner of the border
        length: length of border
        width: width of border
        road_width: width of the road
        contour_air_gap: air gap between adjacent contours
        raster_air_gap: air gap between adjacent roads
        num_contours: number of contours
        contour_start_locs_lst: defines start position of each contour
                    each start_loc is in [LL, LR, UR, UL]
        raster_start_loc_lst: LL, LR, UR, UL
        angle_lst: raster angle (only support 0 and 90)
        grid_length: length of grid
        grid_width: width of grid

    Returns:
        path list
    """
    ncols = length // grid_length
    nrows = width // grid_width
    checker_lst = []
    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            raster_start_loc = raster_start_loc_lst[idx]
            angle = angle_lst[idx]
            contour_start_locs = contour_start_locs_lst[idx]
            curr_x = ox + grid_length * col
            curr_y = oy + grid_width * row
            checker = compute_path2D(curr_x, curr_y, grid_length, grid_width,
                                     road_width, contour_air_gap, raster_air_gap,
                                     num_contours, contour_start_locs, raster_start_loc,
                                     angle)
            checker_lst.append(checker)
    return checker_lst


def compute_checkerboard3D(ox, oy, oz, length, width, height, road_width,
                           layer_height, contour_air_gap_lst, raster_air_gap_lst, num_contours_lst,
                           contour_start_locs_lsts, raster_start_loc_lsts, angle_lsts, grid_length,
                           grid_width):
    """ create checkerboard3D checker_lst
    """
    num_layers = int(height // layer_height)
    hs = np.linspace(oz + 0.5 * layer_height, oz + 0.5 * layer_height +
                     (num_layers - 1) * layer_height, num_layers)
    points_lst = []
    for i in range(num_layers):
        z = hs[i]
        contour_start_locs_lst = contour_start_locs_lsts[i]
        raster_start_loc_lst = raster_start_loc_lsts[i]
        angle_lst = angle_lsts[i]
        contour_air_gap = contour_air_gap_lst[i]
        raster_air_gap = raster_air_gap_lst[i]
        num_contours = num_contours_lst[i]
        temp_checker_lst = compute_checkerboard2D(ox, oy, length, width,
                                                  road_width, contour_air_gap, raster_air_gap,
                                                  num_contours, contour_start_locs_lst,
                                                  raster_start_loc_lst, angle_lst, grid_length, grid_width)
        temp_points_lst = _insertZ(temp_checker_lst, z)
        points_lst.extend(temp_points_lst)
    return points_lst


def _insertZ(checker_lst, z):
    """ insert z value to checker_lst
        return (xs, ys, zs)
    """
    # xs = []
    # ys = []
    points_lst = []
    for checker in checker_lst:
        for points in checker:
            tempxs, tempys = points
            # xs.extend(tempxs)
            # ys.extend(tempys)
            points = (tempxs, tempys, [z] * len(tempxs))
            points_lst.append(points)

    # n = len(xs)
    # zs = [z]*n
    # return (xs, ys, zs)
    return points_lst
