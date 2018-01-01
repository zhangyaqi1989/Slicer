"""
Written by Yaqi Zhang

"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math


def plot_path(points_lst, ox, oy, length, width, colors, plot_border=True):
    ''' points = [xs, ys]
        points_lst = [points1, points2, ...]
        canvas is (ox, oy) --> (ox + length, oy + width)
    '''
    if plot_border:
        xs = np.asarray([0, length, length, 0, 0]) + ox
        ys = np.asarray([0, 0, width, width, 0]) + oy
        plt.plot(xs, ys, 'k-')
    for points, color in zip(points_lst, colors):
        plt.plot(points[0], points[1])
    plt.show()

'''
def raster_path2D(ox, oy, length, width, road_width, angle):
    """create 2D path of 90 degree or 0 degree raster """
    if angle == 0:
        x1 = 0.5*road_width
        x2 = length - 0.5*road_width
        ystart = 0.5*road_width
        temp = np.arange(ystart, width, road_width)
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
        y1 = 0.5*road_width
        y2 = width - 0.5*road_width
        xstart = 0.5*road_width
        temp = np.arange(xstart, length, road_width)
        xs = []
        for v in temp:
            xs.extend([v, v])
        ys = [y1]
        for i in range(1, len(temp)*2):
            if xs[i] == xs[i-1]:
                ys.append(y1 + y2 - ys[-1])
            else:
                ys.append(ys[-1])
    return (xs, ys)
'''

def raster_path2D(ox, oy, length, width, road_width, angle, start_loc = 'LL'):
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
            step = road_width
        else:
            step = -road_width
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
            step = road_width
        else:
            step = -road_width
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
    return (xs, ys)


def raster_path3D(ox, oy, length, width, height, road_width, layer_height, angle, cross, start_loc="LL"):
    hs = np.linspace(0.5*layer_height, height-0.5*layer_height, height/layer_height)
    nlayers = len(hs)
    if cross:
        angles = []
        for i in range(nlayers):
            if i % 2 == 0:
                angles.append(90 - angle)
            else:
                angles.append(angle)
    else:
        angles = [angle]*nlayers
    xs = []
    ys = []
    zs = []
    for i in range(nlayers):
        tempxs, tempys = raster_path2D(ox, oy, length, width, road_width, angles[i], start_loc)
        xs.extend(tempxs)
        ys.extend(tempys)
        zs.extend([hs[i]]*len(tempxs))
    return (xs, ys, zs)

def read_gcode(filename, max_layer=np.inf):
    '''read gcode from a file, store them into a road segments list,
    road format [x0, y0, x1, y1, z, layer_no, style] length, deltat, Area, isSupport, style'''
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
                if gxyzef[0] == 1 and (gxyzef[1] != last_gxyzef[1] or gxyzef[2] != last_gxyzef[2]) and gxyzef[3] == last_gxyzef[3] and gxyzef[4] > last_gxyzef[4]:
                    roads.append((last_gxyzef[1], last_gxyzef[2], gxyzef[1], gxyzef[2], gxyzef[3], layer_no, 1))
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
        if road[0] != old_road[2] or road[1] != old_road[3] or road[4] != old_road[4]:
            segs += 1
            if len(xs) >= 1:
                xs.append(old_road[2])
                ys.append(old_road[3])
                zs.append(old_road[4])
                plot.plot(xs, ys)
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
        plt.draw()
    print("100%")
    plt.show()

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
        if road[0] != old_road[2] or road[1] != old_road[3] or road[4] != old_road[4]:
            segs += 1
            if len(xs) >= 1:
                xs.append(old_road[2])
                ys.append(old_road[3])
                zs.append(old_road[4])
                ax.plot(xs, ys, zs)
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
        plt.draw()
    print("100%")
    plt.show()

def convert_to_gcode(points, filename):
    lines = ['G1 E0 Z0']
    npoints = len(points[0])
    x0, y0, z = (0.0, 0.0, 0.0)
    xs, ys, zs = points
    distance = 0.0
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
        if i != 0 or z == old_z:
            distance = distance + abs(x - x0) + abs(y - y0)
            lst.append('E' + str(distance))
        x0, y0 = (x, y)
        lines.append(' '.join(lst))
    with open(filename, 'w') as outfile:
        for line in lines:
            outfile.write(line + '\n')
    outfile.close()

if __name__ == '__main__':
    print("Hello World")
    ox = 1
    oy = 2
    length = 12
    width = 8
    road_width = 2
    angle = 90
    start_loc = 'LR'
    points = raster_path2D(ox, oy, length, width, road_width, angle, start_loc)
    points_lst = [points]
    colors = ['b-']*len(points_lst)
    plot_path(points_lst, ox, oy, length, width, colors)
