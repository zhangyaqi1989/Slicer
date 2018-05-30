#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
"""
This module contains some basic slicer functions
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
    if plot_border:
        xs, ys = rectangle_border(ox, oy, length, width)
        plt.plot(xs, ys, 'k-')
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


def checkerboard2D(ox, oy, length, width, road_width, contour_air_gap, \
        raster_air_gap, num_contours, contour_start_locs_lst, \
        raster_start_loc_lst, angle_lst, grid_length, grid_width):
    """ create 2D checkerboadr contour + raster path
        contours/rasters = (xs, ys)
        checker = [contours1, contours2,..., rasters]
    """
    ncols = int(length // grid_length)
    nrows = int(width // grid_width)
    checker_lst = []
    for row in range(nrows):
        for col in range(ncols):
            idx = row*ncols + col
            raster_start_loc = raster_start_loc_lst[idx]
            angle = angle_lst[idx]
            contour_start_locs = contour_start_locs_lst[idx]
            curr_x = ox + grid_length * col
            curr_y = oy + grid_width * row
            checker = path2D(curr_x, curr_y, grid_length, grid_width,\
                    road_width, contour_air_gap, raster_air_gap, \
                    num_contours, contour_start_locs, raster_start_loc,\
                    angle)
            checker_lst.append(checker)
    return checker_lst


def checkerboard3D(ox, oy, oz, length, width, height, road_width,\
        layer_height, contour_air_gap_lst, raster_air_gap_lst, num_contours_lst,\
        contour_start_locs_lsts, raster_start_loc_lsts, angle_lsts, grid_length,\
        grid_width):
    """ create checkerboard3D checker_lst
    """
    num_layers = int(height // layer_height)
    hs = np.linspace(oz + 0.5*layer_height, oz + 0.5*layer_height + \
            (num_layers - 1)*layer_height, num_layers)
    points_lst = []
    for i in range(num_layers):
        z = hs[i]
        contour_start_locs_lst = contour_start_locs_lsts[i]
        raster_start_loc_lst = raster_start_loc_lsts[i]
        angle_lst = angle_lsts[i]
        contour_air_gap = contour_air_gap_lst[i]
        raster_air_gap = raster_air_gap_lst[i]
        num_contours = num_contours_lst[i]
        temp_checker_lst = checkerboard2D(ox, oy, length, width, \
                road_width, contour_air_gap, raster_air_gap, \
                num_contours, contour_start_locs_lst,\
        raster_start_loc_lst, angle_lst, grid_length, grid_width)
        temp_points_lst = insertZ(temp_checker_lst, z)
        points_lst.extend(temp_points_lst)
    return points_lst


def insertZ(checker_lst, z):
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
            points = (tempxs, tempys, [z]*len(tempxs))
            points_lst.append(points)

    # n = len(xs)
    # zs = [z]*n
    # return (xs, ys, zs)
    return points_lst


def path2D(ox, oy, length, width, road_width, contour_air_gap, \
        raster_air_gap, num_contours, contour_start_locs, \
        raster_start_loc, angle):
    """ create 2D contour + raster path
        points = (xs, ys)
        points_lst = [contours1, contours2,..., rasters]
    """
    lst = contour_path2D(ox, oy, length, width, road_width,\
            contour_air_gap, num_contours, contour_start_locs)
    gap = road_width + contour_air_gap
    # this update is questionable
    raster_ox = ox + 0.5*road_width + num_contours*gap
    raster_oy = oy + 0.5*road_width + num_contours*gap
    raster_length = length - road_width - num_contours*2*gap
    raster_width = width - road_width - num_contours*2*gap
    lst.extend(raster_path2D(raster_ox, raster_oy, raster_length,\
        raster_width, road_width, raster_air_gap, angle, raster_start_loc))
    return lst


def contour_path2D(ox, oy, length, width, road_width, air_gap, \
        num_contours, start_locs):
    """ create 2D contour path
        contour = (xs, ys)
        contour_lst = [contour1, contour2, ...]
    """
    contour_lst = []
    curr_x = ox + 0.5*road_width
    curr_y = oy + 0.5*road_width
    curr_length = length - road_width
    curr_width = width - road_width
    gap = road_width + air_gap
    for i in range(num_contours):
        start_loc = start_locs[i]
        contour = rectangle_border(curr_x, curr_y, curr_length, \
                curr_width, start_loc)
        contour_lst.append(contour)
        # update curr_x, curr_y, curr_length and curr_width
        curr_x += gap
        curr_y += gap
        assert curr_length >= 2*gap
        assert curr_width >= 2*gap
        curr_length -= 2*gap
        curr_width -= 2*gap
    return contour_lst


def raster_path2D(ox, oy, length, width, road_width, air_gap, angle,\
        start_loc = 'LL'):
    """create 2D path of 90 degree or 0 degree raster
       start_loc: LL, LR, UR, UL
    """
    x_start_func = min
    y_start_func = min
    if start_loc in ['LR', 'UR']:
        x_start_func = max
    if start_loc in ['UL', 'UR']:
        y_start_func = max

    xlims = [0.5*road_width, length - 0.5*road_width]
    ylims = [0.5*road_width, width - 0.5*road_width]

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
        for i in range(1, len(temp)*2):
            if ys[i] == ys[i-1]:
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
        for i in range(1, len(temp)*2):
            if xs[i] == xs[i-1]:
                ys.append(y1 + y2 - ys[-1])
            else:
                ys.append(ys[-1])
    # linear transform
    xs = [x + ox for x in xs]
    ys = [y + oy for y in ys]
    return [(xs, ys)]


def raster_path3D(ox, oy, oz, length, width, height, road_width, layer_height,\
        air_gap, angle, cross, start_loc="LL"):
    """ create 3D raster path ...
    """
    # hs = np.linspace(0.5*layer_height, height-0.5*layer_height,\
    #        height/layer_height)
    # hs = np.linspace(oz + 0.5*layer_height, oz + height-0.5*layer_height,\
    #         height/layer_height)
    # nlayers = len(hs)
    num_layers = int(height // layer_height)
    hs = np.linspace(oz + 0.5*layer_height, oz + 0.5*layer_height + \
            (num_layers)*layer_height, num_layers)
    if cross:
        angles = []
        for i in range(num_layers):
            if i % 2 == 0:
                angles.append(90 - angle)
            else:
                angles.append(angle)
    else:
        angles = [angle]*num_layers
    # xs = []
    # ys = []
    # zs = []
    points_lst = []
    for i in range(num_layers):
        # tempxs, tempys = raster_path2D(ox, oy, length, width, road_width,\
        # air_gap, angles[i], start_loc)
        temps = raster_path2D(ox, oy, length, width, road_width, air_gap,\
                angles[i], start_loc)
        tempxs, tempys = temps[0]
        # xs.extend(tempxs)
        # ys.extend(tempys)
        # zs.extend([hs[i]]*len(tempxs))
        points_lst.append((tempxs, tempys, [hs[i]]*len(tempxs)))
    # return (xs, ys, zs)
    return points_lst



# def path2D(ox, oy, length, width, road_width, contour_air_gap, \
# raster_air_gap, num_contours, contour_start_locs, raster_start_loc, angle):
def path3D(ox, oy, length, width, road_width, contour_air_gaps, \
        raster_air_gaps, num_contours_lst, contour_start_locs_lst,\
        raster_start_loc_lst, angles):
    """ create 3D contour + raster path
    """
    pass


def read_gcode(filename, max_layer=np.inf):
    '''read gcode from a file, store them into a road segments list,
       road format [x0, y0, x1, y1, z, layer_no, style] length, deltat,
       Area, isSupport, style
    '''
    print("Start Reading file " + filename)
    roads = [] # (x0, y0, x1, y1, z, layer_no, style)
    layer_no = 0
    gxyzef = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('G1'):
                last_gxyzef = gxyzef[:]
                items = line.split()
                for item in items:
                    index = None
                    if item[0] == 'G':
                        index = 0
                    elif item[0] == 'X':
                        index = 1
                    elif item[0] == 'Y':
                        index = 2
                    elif item[0] == 'Z':
                        index = 3
                    elif item[0] == 'E':
                        index = 4
                    elif item[0] == 'F':
                        index = 5
                    if not index is None:
                        gxyzef[index] = float(item[1:])
                if gxyzef[3] != last_gxyzef[3]:
                    layer_no += 1
                    if layer_no > max_layer:
                        return roads
                if gxyzef[0] == 1 and (gxyzef[1] != last_gxyzef[1] or \
                        gxyzef[2] != last_gxyzef[2]) and \
                        gxyzef[3] == last_gxyzef[3] and \
                        gxyzef[4] > last_gxyzef[4]:
                    roads.append((last_gxyzef[1], last_gxyzef[2], gxyzef[1],\
                            gxyzef[2], gxyzef[3], layer_no, 1))
    return roads


def plot_roads2D(roads):
    '''plot roads segments read from Gcode'''
    n = len(roads)
    print("Number of roads: " + str(n))
    road = (0, 0, 0, 0, 0, 0)
    n = len(roads)
    xs = []
    ys = []
    zs = []
    k = 0.1
    segs = -1
    tmp = []
    for i in range(n):
        if i > k*n:
            print(str(int(100*k)) + "%")
            k += 0.1
        old_road = road
        road = roads[i]
        if road[0] != old_road[2] or road[1] != old_road[3]\
                or road[4] != old_road[4]:
            segs += 1
            if len(xs) >= 1:
                xs.append(old_road[2])
                ys.append(old_road[3])
                zs.append(old_road[4])
                plt.plot(xs, ys)
                plt.scatter(xs[0], ys[0])
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
        plt.plot(xs, ys)
        plt.scatter(xs[0], ys[0])
        plt.draw()
    print("100%")
    # plt.show()


def plot_roads3D(roads):
    '''plot roads segments read from gecode'''
    n = len(roads)
    print("Number of roads: " + str(n))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    road = (0, 0, 0, 0, 0, 0)
    n = len(roads)
    xs = []
    ys = []
    zs = []
    k = 0.1
    segs = -1
    tmp = []
    for i in range(n):
        if i > k*n:
            print(str(int(100*k)) + "%")
            k += 0.1
        old_road = road
        road = roads[i]
        if road[0] != old_road[2] or road[1] != old_road[3] or\
                road[4] != old_road[4]:
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
    # plt.show()


def convert_to_gcode(points_lst, filename):
    """ points_lst = [subpath1, subpath2 ... ]
    """
    lines = ['G1 E0 Z0'] # 'G1 E0 Z0' is the first line
    distance = 0.0

    x, y, z = (float('Inf'), float('Inf'), float('Inf'))
    for points in points_lst:
        xs, ys, zs = points
        npoints = len(xs)
        for i in range(npoints):
            lst = ['G1']
            x = xs[i]
            y = ys[i]
            old_z = z
            z = zs[i]
            lst.append('X' + str(x))
            lst.append('Y' + str(y))
            if z != old_z:
                lst.append('Z' + str(z))
            if i != 0: # i == 0 means travel nozzle
                distance = distance + abs(x - x0) + abs(y - y0)
                lst.append('E' + str(distance))
            x0, y0 = (x, y)
            lines.append(' '.join(lst))

    with open(filename, 'w') as outfile:
        for line in lines:
            outfile.write(line + '\n')
    outfile.close()


def test_raster_path2D():
    ox = 1
    oy = 2
    length = 30
    width = 18
    road_width = 1
    angle = 0
    start_loc = 'LR'
    air_gap = 0 # 0.2*road_width
    points = raster_path2D(ox, oy, length, width, road_width, air_gap,\
            angle, start_loc)
    points_lst = points
    colors = ['b-']*len(points_lst)
    plot_path(points_lst, ox, oy, length, width, colors)
    plt.show()


def plot_checkerboard2D(checker_lst, ox, oy, length, width, grid_length,\
        grid_width, colors_lst, checker_plot_lst):
    """ plot checkboard given checker_lst

    """
    nrows = width // grid_width
    ncols = length // grid_length
    for row in range(nrows):
        for col in range(ncols):
            curr_x = ox + col*grid_length
            curr_y = oy + row*grid_width
            idx = col + row*ncols
            if not checker_plot_lst[idx]:
                continue
            checker = checker_lst[idx]
            colors = colors_lst[idx]
            plot_path(checker, curr_x, curr_y, grid_length,\
                    grid_width, colors)


def test_contour_path2D():
    ox = 1
    oy = 2
    length = 30
    width = 18
    road_width = 1
    air_gap = 0 # 0.2*road_width
    num_contours = 2
    start_locs = ['LL', 'UR']
    contour_lst = contour_path2D(ox, oy, length, width, road_width, \
            air_gap, num_contours, start_locs)
    colors = ['b-']*len(contour_lst)
    plot_path(contour_lst, ox, oy, length, width, colors)
    plt.show()


def test_path2D():
    """Test compute_path2D()"""
    ox = 1
    oy = 2
    length = 30
    width = 18
    road_width = 1
    angle = 0
    start_loc = 'LR'
    air_gap = 0 # 0.2*road_width
    contour_air_gap = 0
    raster_air_gap = 0
    num_contours = 3
    raster_start_loc = 'LR'
    contour_start_locs = ['LL']*num_contours
    points_lst = path2D(ox, oy, length, width, road_width,\
            contour_air_gap, raster_air_gap, num_contours, \
            contour_start_locs, raster_start_loc, angle)
    colors = ['r-']*num_contours + ['b-']
    plot_path(points_lst, ox, oy, length, width, colors)
    plt.show()


def create_checkerboard2D(ox, oy, grid_length, grid_width, nrows, ncols, \
        num_contours, road_width, contour_air_gap, raster_air_gap):
    num_checkers = nrows*ncols
    length = grid_length*ncols
    width = grid_width*nrows
    # # fixed value
    # raster_start_loc_lst = ['LR']*num_checkers
    # contour_start_locs = ['LL']*num_contours
    # angle_lst = [0]*num_checkers
    # # random pick
    raster_start_loc_lst = rand_start_locs(num_checkers)
    contour_start_locs_lst = []
    angle_lst = np.random.choice([0, 90], num_checkers, replace=True)
    for i in range(num_checkers):
        contour_start_locs_lst.append(rand_start_locs(num_contours))
    checker_lst = checkerboard2D(ox, oy, length, width, road_width,\
            contour_air_gap, raster_air_gap,\
            num_contours, contour_start_locs_lst, raster_start_loc_lst,\
            angle_lst, grid_length, grid_width)
    return checker_lst


def test_checkerboard2D():
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
    length = ncols*grid_length
    width = nrows*grid_width
    checker_lst = create_checkerboard2D(ox, oy, grid_length, grid_width,\
            nrows, ncols, num_contours, road_width, contour_air_gap,\
            raster_air_gap)

    colors = ['r-']*num_contours + ['b-']
    colors_lst = []
    for i in range(len(checker_lst)):
        colors_lst.append(colors)
    checker_plot_lst = [True]*len(checker_lst)
    # checker_plot_lst = np.random.choice([True, False], len(checker_lst))
    plot_checkerboard2D(checker_lst, ox, oy, length, width, grid_length,\
        grid_width, colors_lst, checker_plot_lst)
    # for test
    '''
    filename = 'test.gcode'
    points_lst = insertZ(checker_lst, 2)
    convert_to_gcode(points_lst, filename)
    roads = read_gcode(filename, 10)
    plot_roads3D(roads)
    '''
    plt.show()


def test_checkerboard3D():
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
    length = ncols*grid_length
    width = nrows*grid_width
    checker_lst = create_checkerboard2D(ox, oy, grid_length, grid_width,\
            nrows, ncols, num_contours, road_width, contour_air_gap,\
            raster_air_gap)

    filename = 'test.gcode'
    points_lst = insertZ(checker_lst, 2)
    convert_to_gcode(points_lst, filename)
    roads = read_gcode(filename, 10)
    plot_roads3D(roads)
    plt.show()


def test_checkerboard3D_multlayers():
    ox = 1
    oy = 2
    oz = 10
    grid_length = 30
    grid_width = 20
    nrows = 3
    ncols = 4
    road_width = 1
    layer_height = 2
    height = 4
    num_layers = int(height/layer_height)
    num_checkers = nrows*ncols
    num_contours = 3
    contour_air_gap_lst = [0]*num_layers
    raster_air_gap_lst = [0]*num_layers
    num_contours_lst = [3]*num_layers
    length = ncols*grid_length
    width = nrows*grid_width
    raster_start_loc_lsts = []
    for i in range(num_layers):
        raster_start_loc_lst = rand_start_locs(num_checkers)
        raster_start_loc_lsts.append(raster_start_loc_lst)
    contour_start_locs_lsts = []
    for i in range(num_layers):
        contour_start_locs_lst = []
        for i in range(num_checkers):
            contour_start_locs_lst.append(rand_start_locs(num_contours))
        contour_start_locs_lsts.append(contour_start_locs_lst)
    angle_lsts = []
    for i in range(num_layers):
        angle_lst = np.random.choice([0, 90], num_checkers, replace=True)
        angle_lsts.append(angle_lst)
    points_lst = checkerboard3D(ox, oy, oz, length, width, height, road_width,\
            layer_height, contour_air_gap_lst, raster_air_gap_lst, num_contours_lst,\
            contour_start_locs_lsts, raster_start_loc_lsts, angle_lsts, grid_length,\
            grid_width)
    points_lst = checkerboard3D(ox, oy, oz, length, width, height, road_width,\
        layer_height, contour_air_gap_lst, raster_air_gap_lst, num_contours_lst,\
        contour_start_locs_lsts, raster_start_loc_lsts, angle_lsts, grid_length,\
        grid_width)

    filename = 'test.gcode'
    convert_to_gcode(points_lst, filename)
    roads = read_gcode(filename, 10)
    plot_roads3D(roads)
    plt.show()


def rand_start_locs(n):
    """ generate start_locs randomly """
    return np.random.choice(['LL', 'LR', 'UL', 'UR'], n, replace=True)


def _compute_y(x0, y0, theta, x1):
    L = (x1 - x0)/np.cos(theta)
    return y0 + L*np.sin(theta)

def _compute_x(x0, y0, theta, y1):
    L = (y1 - y0)/np.sin(theta)
    return x0 + L*np.cos(theta)


def _line(x0, y0, theta, length, width, road_width, ox, oy):
    """ return the intersect points between line
        defined by (x0, y0) and theta and the inner rectangle
    """
    length -= road_width
    width -= road_width
    dx = ox + 0.5*road_width
    dy = oy + 0.5*road_width
    # left end
    left_x = 0.0
    left_y = _compute_y(x0, y0, theta, left_x)
    if left_y < 0.0 or left_y > width:
        left_y = 0
        left_x = _compute_x(x0, y0, theta, left_y)
    # right end
    right_x = length
    right_y = _compute_y(x0, y0, theta, right_x)
    if right_y < 0.0 or right_y > width:
        right_y = width
        right_x = _compute_x(x0, y0, theta, right_y)
    return ((left_x + dx, right_x + dx), (left_y + dy, right_y + dy))


def _connect_through_border(x0, y0, x1, y1):
    if x0 == x1 or y1 == y0:
        return ([x0, x1], [y0, y1])
    else:
        return([x0, y1, x1], [y0, x1, y1])


def test():
    ox = 1
    oy = 2
    length = 20
    width = 14
    road_width = 1
    xs, ys = rectangle_border(ox, oy, length, width)
    plt.plot(xs, ys, 'k-')
    xs, ys = rectangle_border(ox+0.5*road_width, oy+0.5*road_width, \
            length-road_width, width-road_width)
    plt.plot(xs, ys, 'r--')
    # calculate parallel lines
    # UL: (0, width)
    gap = 2
    theta = np.pi/4.0
    # theta = np.pi/6.0
    delt_x = gap/np.sin(theta)
    x = 0.5*road_width
    points_xs = []
    points_ys = []
    end = 0
    while x < length - road_width + (width-road_width)/np.tan(theta):
        lines = _line(x, width-road_width, theta, length, width, road_width, ox, oy)
        (x0, x1), (y0, y1) = lines
        if end == 0:
            points_xs.extend([x0, x1])
            points_ys.extend([y0, y1])
        else:
            points_xs.extend([x1, x0])
            points_ys.extend([y1, y0])
        end = 1 - end
        # plt.plot(lines[0], lines[1])
        # plt.scatter(lines[0], lines[1])
        x += delt_x
    plt.plot(points_xs, points_ys)
    print(points_xs)
    print(points_ys)

    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: >> python slice.py <test_type>")
        exit(1)
    test_type = int(sys.argv[1])
    if test_type == 1:
        test_raster_path2D()
    elif test_type == 2:
        test_contour_path2D()
    elif test_type == 3:
        test_path2D()
    elif test_type == 4:
        test_checkerboard2D()
    elif test_type == 5:
        test_checkerboard3D()
    elif test_type == 6:
        test_checkerboard3D_multlayers()
    elif test_type == 7:
        test()
    else:
        print("unknown test type!")
