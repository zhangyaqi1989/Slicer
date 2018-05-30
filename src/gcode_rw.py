#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
"""
This module contains functions for Gcode reader
and writer
"""

# standard library
import sys

# third party library
import numpy as np
# import matplotlib.pyplot as plt


def read_gcode(filename, max_layer=np.inf):
    """
    read a gcode file, store tool path into a road segments list,
    road format [x0, y0, x1, y1, z, layer_no, style]
    only keeps [1, max_layer] layers

    Args:
        filename: Gcode file
        max_layer: specify the maximum layer readed

    Returns:
        roads: road segments list
    """
    print("Start Reading file " + filename)
    roads = []  # (x0, y0, x1, y1, z, layer_no, style)
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
                if gxyzef[0] == 1 and (gxyzef[1] != last_gxyzef[1] or
                                       gxyzef[2] != last_gxyzef[2]) and \
                        gxyzef[3] == last_gxyzef[3] and \
                        gxyzef[4] > last_gxyzef[4]:
                    roads.append((last_gxyzef[1], last_gxyzef[2], gxyzef[1],
                                  gxyzef[2], gxyzef[3], layer_no, 1))
    return roads


def write_gcode(path_lst, filename):
    """
    write path to a Gcode file, the format is regular FDM

    Args:
        path_lst: [subpath1, subpath2 ... ]
                  subpath = (xs, ys, zs)
        filename: Gcode filename

    Returns:
        None
    """
    lines = ['G1 E0 Z0']  # 'G1 E0 Z0' is the first line
    distance = 0.0

    x, y, z = (float('Inf'), float('Inf'), float('Inf'))
    for path in path_lst:
        xs, ys, zs = path
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
            if i != 0:  # i == 0 means travel nozzle
                distance = distance + abs(x - x0) + abs(y - y0)
                lst.append('E' + str(distance))
            x0, y0 = (x, y)
            lines.append(' '.join(lst))

    with open(filename, 'w') as outfile:
        for line in lines:
            outfile.write(line + '\n')
    outfile.close()


if __name__ == "__main__":
    print("Hello World")
