from slice import *

if __name__ == "__main__":
    filename = 'test.gcode'
    length = 10
    width = 10
    road_width = 0.5
    layer_height = 0.5
    angle = 0
    x0 = 0
    y0 = 0
    cross = True # 0/90 angle

    # single layer 1D
    '''
    points = raster_path3D(x0, y0, length, width, layer_height, road_width, layer_height, angle, False)
    convert_to_gcode(points, filename)
    roads = read_gcode(filename, 10)
    plot_roads2D(roads)
    '''

    # multiple layers 3D
    num_layers = 2
    height = num_layers * layer_height
    start_loc = "UR"
    points = raster_path3D(x0, y0, length, width, height, road_width, layer_height, angle, cross, start_loc)
    convert_to_gcode(points, filename)
    roads = read_gcode(filename, 10)
    plot_roads3D(roads)
