"""
Written by Yaqi Zhang

"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math


def plot_path(points_lst, ox, oy, length, width, colors, plot_border=True):
    ''' points = (xs, ys)
        points_lst = [points1, points2, ...]
        canvas is (ox, oy) --> (ox + length, oy + width)
    '''
    if plot_border:
        xs, ys = rectangle_border(ox, oy, length, width)
        plt.plot(xs, ys, 'k-')
    for points, color in zip(points_lst, colors):
        plt.scatter(points[0][0], points[1][0])
        plt.plot(points[0], points[1], color)


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



def checkerboard2D(ox, oy, length, width, road_width, contour_air_gap, raster_air_gap,\
        num_contours, contour_start_locs_lst, raster_start_loc_lst, angle_lst, \
        grid_length, grid_width):
    """ create 2D checkerboadr contour + raster path
        points = (xs, ys)
        points_lst = [contours1, contours2,..., rasters]
    """
    ncols = length // grid_length
    nrows = width // grid_width
    checker_lst = []
    for row in range(nrows):
        for col in range(ncols):
            idx = row*ncols + col
            raster_start_loc = raster_start_loc_lst[idx]
            angle = angle_lst[idx]
            contour_start_locs = contour_start_locs_lst[idx]
            curr_x = ox + grid_length * col
            curr_y = oy + grid_width * row
            checker = path2D(curr_x, curr_y, grid_length, grid_width, road_width,\
                    contour_air_gap, raster_air_gap, num_contours, contour_start_locs,\
                    raster_start_loc, angle)
            checker_lst.append(checker)
    return checker_lst


def path2D(ox, oy, length, width, road_width, contour_air_gap, raster_air_gap,\
        num_contours, contour_start_locs, raster_start_loc, angle):
    """ create 2D contour + raster path
        points = (xs, ys)
        points_lst = [contours1, contours2,..., rasters]
    """
    lst = contour_path2D(ox, oy, length, width, road_width, contour_air_gap, \
            num_contours, contour_start_locs)
    gap = road_width + contour_air_gap
    # this update is questionable
    raster_ox = ox + 0.5*road_width + num_contours*gap
    raster_oy = oy + 0.5*road_width + num_contours*gap
    raster_length = length - road_width - num_contours*2*gap
    raster_width = width - road_width - num_contours*2*gap
    lst.extend(raster_path2D(raster_ox, raster_oy, raster_length, raster_width,\
            road_width, raster_air_gap, angle, raster_start_loc))
    return lst


def contour_path2D(ox, oy, length, width, road_width, air_gap, num_contours, start_locs):
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
        contour = rectangle_border(curr_x, curr_y, curr_length, curr_width, start_loc)
        contour_lst.append(contour)
        # update curr_x, curr_y, curr_length and curr_width
        curr_x += gap
        curr_y += gap
        assert curr_length >= 2*gap
        assert curr_width >= 2*gap
        curr_length -= 2*gap
        curr_width -= 2*gap
    return contour_lst


def raster_path2D(ox, oy, length, width, road_width, air_gap, angle, start_loc = 'LL'):
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


def raster_path3D(ox, oy, length, width, height, road_width, layer_height, air_gap, angle, cross, start_loc="LL"):
    """ create 3D raster path ...
    """
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
        # tempxs, tempys = raster_path2D(ox, oy, length, width, road_width, air_gap, angles[i], start_loc)
        temps = raster_path2D(ox, oy, length, width, road_width, air_gap, angles[i], start_loc)
        tempxs, tempys = temps[0]
        xs.extend(tempxs)
        ys.extend(tempys)
        zs.extend([hs[i]]*len(tempxs))
    return (xs, ys, zs)



# def path2D(ox, oy, length, width, road_width, contour_air_gap, raster_air_gap, num_contours, contour_start_locs, raster_start_loc, angle):
def path3D(ox, oy, length, width, road_width, contour_air_gaps, raster_air_gaps,\
        num_contours_lst, contour_start_locs_lst, raster_start_loc_lst, angles):
    """ create 3D contour + raster path
    """
    pass


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


def test_raster_path2D():
    ox = 1
    oy = 2
    length = 30
    width = 18
    road_width = 1
    angle = 0
    start_loc = 'LR'
    air_gap = 0 # 0.2*road_width
    points = raster_path2D(ox, oy, length, width, road_width, air_gap, angle, start_loc)
    points_lst = points
    colors = ['b-']*len(points_lst)
    plot_path(points_lst, ox, oy, length, width, colors)
    plt.show()


def test_contour_path2D():
    ox = 1
    oy = 2
    length = 30
    width = 18
    road_width = 1
    air_gap = 0 # 0.2*road_width
    num_contours = 2
    start_locs = ['LL', 'UR']
    contour_lst = contour_path2D(ox, oy, length, width, road_width, air_gap, num_contours, start_locs)
    colors = ['b-']*len(contour_lst)
    plot_path(contour_lst, ox, oy, length, width, colors)
    plt.show()


def test_path2D():
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
    points_lst = path2D(ox, oy, length, width, road_width, contour_air_gap, raster_air_gap, num_contours, contour_start_locs, raster_start_loc, angle)
    colors = ['r-']*num_contours + ['b-']
    plot_path(points_lst, ox, oy, length, width, colors)
    plt.show()

def test_checkerboard2D():
    ox = 1
    oy = 2
    grid_length = 30
    grid_width = 18
    nrows = 3
    ncols = 4
    num_checkers = nrows*ncols
    length = grid_length*ncols
    width = grid_width*nrows
    road_width = 1
    angle_lst = [0]*num_checkers
    contour_air_gap = 0
    raster_air_gap = 0
    num_contours = 3
    raster_start_loc_lst = ['LR']*num_checkers
    contour_start_locs = ['LL']*num_contours
    contour_start_locs_lst = []
    for i in range(num_checkers):
        contour_start_locs_lst.append(contour_start_locs)
    checker_lst = checkerboard2D(ox, oy, length, width, road_width, contour_air_gap, raster_air_gap,\
        num_contours, contour_start_locs_lst, raster_start_loc_lst, angle_lst, \
        grid_length, grid_width)
    for row in range(nrows):
        for col in range(ncols):
            curr_x = ox + col*grid_length
            curr_y = oy + row*grid_width
            idx = col + row*ncols
            checker = checker_lst[idx]
            colors = ['r-']*num_contours + ['b-']
            plot_path(checker, curr_x, curr_y, grid_length, grid_width, colors)
    plt.show()


if __name__ == '__main__':
    print("Hello World")
    # test_raster_path2D()
    # test_contour_path2D()
    # test_path2D()
    test_checkerboard2D()
