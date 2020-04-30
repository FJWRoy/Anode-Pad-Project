import numpy as np
import time
import math
import sys
from tqdm import tqdm
from shapely.geometry import LineString, MultiPolygon, MultiPoint, box, Polygon,Point


def contained_points(point_list, b):
    zeros = np.array([])
    for j in range(len(point_list)):
       zeros = np.append(zeros, b.contains(Point(point_list[j])))
    return zeros

class sim_anode:
    def __init__(self):
        self.coord_x = None
        self.coord_y = None
        self.lst_coord = list()
        self.amplitude = list()
        self.middle_point = None
        self.center_pads = list()

    def get_coord_grid(self, n_lasers, start, end):
        self.coord_x = np.linspace(start[0], end[0], n_lasers)#2 dimentional sampling points form a grid
        coeff_x = (end[1] - start[1]) / (end[0] - start[0])
        coeff_y = (end[1] - end[0] * coeff_x)#These coefficients are for non-axial sampling. Not currently used.
        self.coord_y = np.linspace(start[0], end[0], n_lasers)#2 dimentional sampling points form a grid
        self.middle_point = [self.coord_x[math.ceil(len(self.coord_x)/2)],self.coord_y[math.ceil(len(self.coord_y)/2)]]#The center of the grid.

    def update_end(self,pad):
        """
        lst_pad = [pad.box_array[11],pad.box_array[12],pad.box_array[13]]
        
        z = [contained_points(self.lst_coord, b) for b in lst_pad]
        left_start = -6
        center_start = self.coord_x[next(x for x, val in enumerate(z[1]) if val > 0)]
        right_start = self.coord_x[next(x for x, val in enumerate(z[2]) if val > 0)]
        right_end = self.coord_x[len(self.coord_x) - next(x for x, val in enumerate(z[2][::-1]) if val > 0)]
        self.center_pads = [left_start, center_start, right_start, right_end]#We save the coordinates of the pads of interest.
        """

    def run_sim(self, myPadArray, radius):
        self.amplitude = self.run_sim_table(myPadArray, radius)
    # saves lookup table of amplitudes.
    def run_sim_table(self, myPadArray, radius):
        radius = float(radius)
        s = myPadArray.side
        center = [myPadArray.center_x, myPadArray.center_y]#The center of the pad array.
        self.lst_coord = list((x,y) for x in self.coord_x for y in self.coord_y)#Define a grid from coord_x and coord_y
        lst_spot = [Point(x,y).buffer(radius) for (x,y) in self.lst_coord]
        lst_amp = [[x.intersection(b).area for x in lst_spot] for b in myPadArray.box_array]
        return lst_amp
        

if __name__ == "__main__":
    print("error:SimAnode is running as main")
