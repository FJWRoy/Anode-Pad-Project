import numpy as np
import time
import math
import sys
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
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

    def get_coord_grid(self, n_lasers, pad_size):
        start = -2.5*pad_size
        end = 2.5*pad_size
        self.coord_x = np.linspace(start, end, n_lasers+1)#2 dimentional sampling points form a grid. For sake of symmetry, 1 is added.
        self.coord_x =  self.coord_x[:-1]
        self.coord_y = np.linspace(start, end, n_lasers+1)#2 dimentional sampling points form a grid
        self.coord_y =  self.coord_y[:-1]
        self.middle_point = [self.coord_x[math.ceil(len(self.coord_x)/2)],self.coord_y[math.ceil(len(self.coord_y)/2)]]#The center of the grid.

    def get_coord_grid_multilayer(self, n_lasers, pad_size, layer):
        start = 0
        end = 4*layer*pad_size
        self.coord_x = np.linspace(start, end, n_lasers+1)#2 dimentional sampling points form a grid. For sake of symmetry, 1 is added.
        self.coord_x =  self.coord_x[:-1]
        self.coord_y = np.linspace(start, end, n_lasers+1)#2 dimentional sampling points form a grid
        self.coord_y =  self.coord_y[:-1]
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
        self.lst_coord = list((x,y) for y in self.coord_y for x in self.coord_x)#Define a grid from coord_x and coord_y
        #lst_spot = [Point(x,y).buffer(radius) for (x,y) in self.lst_coord]
        #lst_amp = [[x.intersection(b).area for x in lst_spot] for b in tqdm(myPadArray.box_array, leave=False, desc = 'simulation')]
        lst_amp = [self.sim_job_gaussian(30, radius *3 , radius/2, b) for b in tqdm(myPadArray.box_array, leave=False, desc = 'simulation')]
        #More memory efficient to not use lst_spot. Speed implications?
        return np.array(lst_amp)
    def run_sim_multithread(self, myPadArray, radius, num_thread):
        self.amplitude = self.run_sim_table_multithread(myPadArray, radius, num_thread)
    def run_sim_multilayer_multithread(self, myPadArray, radius, num_thread, array_size):
        table = self.run_sim_table_multithread(myPadArray, radius, num_thread)
        n = 4*array_size
        for c in range(8):
            if(c%4==0):
                for i in range(2*array_size):
                    sum = 0
                    for j in range(array_size):
                        sum += table[c//4+(c%4)*n+j*n*4 + 2*i + 2*n*(i%2)]
                    self.amplitude.append(sum)
            if(c%4==1):
                for i in range(2*array_size):
                    sum = 0
                    for j in range(array_size):
                        sum += table[c//4+(c%4)*n+j*4 + 2*(i%2) + 2*n*i]
                    self.amplitude.append(sum)
            if(c%4==2):
                for i in range(2*array_size):
                    sum = 0
                    for j in range(array_size):
                        sum += table[c//4+(c%4)*n+j*n*4 + 2*i - 2*n*(i%2)]
                    self.amplitude.append(sum)
            if(c%4==3):
                for i in range(2*array_size):
                    sum = 0
                    for j in range(array_size):
                        sum += table[c//4+(c%4-2)*n+2+j*4 - 2*(i%2) + 2*n*i]
                    self.amplitude.append(sum)
        self.amplitude = np.array(self.amplitude)
    # saves lookup table of amplitudes.
    def run_sim_table_multithread(self, myPadArray, radius, num_thread):
        self.lst_coord = list((x,y) for y in self.coord_y for x in self.coord_x)#Define a grid from coord_x and coord_y
        lst_amp = Parallel(n_jobs = num_thread, verbose = 10)(delayed(self.sim_job_gaussian)(6, radius *3 , radius/2, b) for b in myPadArray.box_array)
        return np.array(lst_amp)
    def sim_job(self, radius, b):
        return np.array([Point(x,y).buffer(radius).intersection(b).area for (x,y) in self.lst_coord])

    def sim_job_gaussian(self, partition, cutoff, std, b):
        return np.array([sum([Point(x,y).buffer(cutoff / partition * (i+1)).intersection(b).area * 1/(2*math.pi*( std ** 2))*(math.exp(-1/2*((cutoff / partition * (i+0.5) / std) ** 2)) - math.exp(-1/2*((cutoff / partition * (i+1.5) / std) ** 2)))  for i in range(partition)]) for (x,y) in self.lst_coord])
    """
    def gaussian_integral(self, partition, cutoff, std, radius, b):
        
        sum = np.array()
        for i in range(partition):
            current_point = cutoff / partition * i / std
            next_point = cutoff / partition * (i+1) / std
            disk_thickness = 1/(2* math.pi *( std ** 2))*(math.exp(-1/2*current_point ** 2) - math.exp(-1/2*next_point ** 2))
            sum += np.array([Point(x,y).buffer(next_point * std).intersection(b).area for (x,y) in self.lst_coord])
        return sum
    """



    def read_sim(self, filename):
        self.amplitude = np.load(filename)


if __name__ == "__main__":
    print("error:SimAnode is running as main")
