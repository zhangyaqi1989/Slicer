import sys
import matplotlib.pyplot as plt
import slice

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: >> python gcode_reader.py <filename> <max_layer_plot>")
        exit(1)
    filename = sys.argv[1]
    max_layer_num = int(sys.argv[2])
    roads = slice.read_gcode(filename, max_layer_num)
    slice.plot_roads3D(roads)
    plt.axis('equal')
    plt.show()
