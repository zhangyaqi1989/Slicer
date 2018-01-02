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
    points = slice.raster_path3D(x0, y0, length, width, layer_height,\
            road_width, layer_height, air_gap, angle, cross, start_loc)
    slice.convert_to_gcode(points, filename)
    roads = slice.read_gcode(filename, 10)
    slice.plot_roads2D(roads)


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
    points = slice.raster_path3D(x0, y0, length, width, height, \
            road_width, layer_height, air_gap, angle, cross, start_loc)
    slice.convert_to_gcode(points, filename)
    roads = slice.read_gcode(filename, 10)
    slice.plot_roads3D(roads)


if __name__ == "__main__":
    # test_single_layer()
    test_multiple_layers()

