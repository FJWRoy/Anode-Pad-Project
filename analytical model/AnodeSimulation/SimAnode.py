import numpy as np
import time
import math
import sys
from tqdm import tqdm
from shapely.geometry import LineString, MultiPolygon, MultiPoint, box, Polygon,Point


def is_contain(point_list, b):
    zeros = np.array([])
    for j in range(len(point_list)):
       zeros = np.append(zeros, b.contains(Point(point_list[j])))
    return zeros

class sim_anode:
    def __init__(self):
        self.coord_x = None
        self.coord_y = None
        self.res = list()
        self.amplitude = list()
        self.reconstructed = list()
        self.middle_point = None
        self.center_pads = list()

    def get_coord_grid(self, n_lasers, start, end):
        self.coord_x = np.linspace(start[0], end[0], n_lasers)
        coeff_x = (end[1] - start[1]) / (end[0] - start[0])
        coeff_y = (end[1] - end[0] * coeff_x)
        self.coord_y = [x*coeff_x + coeff_y for x in self.coord_x]
        self.middle_point = [self.coord_x[math.ceil(len(self.coord_x)/2)],self.coord_y[math.ceil(len(self.coord_y)/2)]]

    def update_end(self,pad):
        lst_pad = [pad.box_array[11],pad.box_array[12],pad.box_array[13]]
        lst_coord = list(zip(self.coord_x, self.coord_y))
        z = [is_contain(lst_coord, b) for b in lst_pad]
        left_start = self.coord_x[next(x for x, val in enumerate(z[0]) if val > 0)]
        center_start = self.coord_x[next(x for x, val in enumerate(z[1]) if val > 0)]
        right_start = self.coord_x[next(x for x, val in enumerate(z[2]) if val > 0)]
        right_end = self.coord_x[len(self.coord_x) - next(x for x, val in enumerate(z[2][::-1]) if val > 0)]
        self.center_pads = [left_start, center_start, right_start, right_end]

    def run_sim(self, myPadArray, radius):
        radius = float(radius)
        s = myPadArray.side
        center = [myPadArray.center_x, myPadArray.center_y]
        lst_coord = list(zip(self.coord_x, self.coord_y))
        lst_spot = [Point(x,y).buffer(radius) for (x,y) in lst_coord]
        lst_amp = [[x.intersection(b).area for x in lst_spot] for b in myPadArray.box_array]
        for g in tqdm(range(len(lst_coord)), leave=False, desc='calculating resolution'):
            pad_shape = np.array(lst_amp)[:,g].reshape(5,5)/lst_spot[0].area
            coord = lst_coord[g]
            x_formula = np.array([center[0] - 2*s, center[0] - s, center[0], center[0] + s, center[0] + 2*s])
            y_formula = np.array([center[1] - 2*s, center[1] - s, center[1], center[1] + s, center[1] + 2*s])
            cons_x = np.multiply(pad_shape, x_formula).sum()
            cons_y = np.multiply(pad_shape.T, y_formula).sum()
            delta = ((cons_x - coord[0])**2 + (cons_y - coord[1])**2)**0.5
            self.reconstructed.append((cons_x, cons_y))
            self.res.append(delta)
            self.amplitude.append((pad_shape[2,1],pad_shape[2,2],pad_shape[2,3]))

if __name__ == "__main__":
    print("error:SimAnode is running as main")
