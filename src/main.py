"""
Written by Yaqi Zhang

"""

import sys
import matplotlib.pyplot as plt
import slice


def test_single_layer():
    filename = 'test.gcode'
    length = 10
    width = 10
    road_width = 0.5
    layer_height = 0.5
    air_gap = 0.2*road_width
    angle = 0
    x0 = 0
    y0 = 0
    cross = True # 0/90 angle
    start_loc = "UR"

    # single layer 1D
    cross = False
    points_lst = slice.raster_path3D(x0, y0, length, width, layer_height,\
            road_width, layer_height, air_gap, angle, cross, start_loc)
    # print(points_lst)
    slice.convert_to_gcode(points_lst, filename)
    roads = slice.read_gcode(filename, 10)
    slice.plot_roads2D(roads)
    plt.show()


def test_multiple_layers():
    filename = 'test.gcode'
    length = 10
    width = 10
    road_width = 0.5
    layer_height = 0.5
    air_gap = 0.2*road_width
    angle = 0
    x0 = 0
    y0 = 0
    cross = True # 0/90 angle
    start_loc = "UR"

    # multiple layers 3D
    num_layers = 2
    height = num_layers * layer_height
    points_lst = slice.raster_path3D(x0, y0, length, width, height, \
            road_width, layer_height, air_gap, angle, cross, start_loc)
    # print(points_lst)
    slice.convert_to_gcode(points_lst, filename)
    roads = slice.read_gcode(filename, 10)
    slice.plot_roads3D(roads)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: >> python main.py <test_type>")
        exit(1)
    test_type = int(sys.argv[1])
    if test_type == 1:
        test_single_layer()
    elif test_type == 2:
        test_multiple_layers()

