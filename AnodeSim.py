import numpy as np
import random
import pandas as pd
import math
from matplotlib import pyplot as plt
from shapely.geometry import LineString, MultiPolygon, MultiPoint, box, Polygon,Point
from shapely.ops import split,  unary_union
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit
import seaborn as sns
sns.set()

class myPadArray:
    def __init__(self, side):
        self.side = side
        self.box = None
        self.box_array = None
    # return a box on Second Quadrant. dont change
    def get_one_square_box(self):
        """Create a square box

        :param side: length of a side of the box. coord starts from negative x to 0, positive y to 0
        :return: a box on second quadrant  #
        """
        b = box(-self.side, 0.0, 0.0, self.side)
        self.box = b
    # return a list 9 polygon (defined in shapely)
    def get_pad_nine(self):
        """

        :param s: length of the side of the box
        :return: a list of polygon. center to right to SE, S to NE
        """
        s = self.side
        b = self.box
        lists = np.array(list(b.exterior.coords))
        lists_after_parallel_right = lists + np.array([s, 0])
        lists_after_parallel_left = lists + np.array([-s, 0])

        lists_after_parallel_upright = lists + np.array([s, s])
        lists_after_parallel_upcenter = lists + np.array([0, s])
        lists_after_parallel_upleft = lists + np.array([-s, s])

        lists_after_parallel_lowright = lists + np.array([s, -s])
        lists_after_parallel_lowcenter = lists + np.array([0, -s])
        lists_after_parallel_lowleft = lists + np.array([-s, -s])

        b_after_parallel_right = Polygon(lists_after_parallel_right.tolist())
        b_after_parallel_left = Polygon(lists_after_parallel_left.tolist())

        b_after_parallel_upright = Polygon(lists_after_parallel_upright.tolist())
        b_after_parallel_upcenter = Polygon(lists_after_parallel_upcenter.tolist())
        b_after_parallel_upleft = Polygon(lists_after_parallel_upleft.tolist())

        b_after_parallel_lowright = Polygon(lists_after_parallel_lowright.tolist())
        b_after_parallel_lowcenter = Polygon(lists_after_parallel_lowcenter.tolist())
        b_after_parallel_lowleft = Polygon(lists_after_parallel_lowleft.tolist())


        array = list([b, b_after_parallel_left, b_after_parallel_upleft, b_after_parallel_upcenter,
        b_after_parallel_upright, b_after_parallel_right, b_after_parallel_lowright,
        b_after_parallel_lowcenter, b_after_parallel_lowleft])

        self.box_array = array
# given a coord, return an average pos resolution delta r

class sim_anode:
    def __init__(self, padArray, r, n, noi):
        self.box_array = padArray.box_array
        self.side = padArray.side
        self.radius = r
        self.coord_x = None
        self.coord_y = None
        self.num_points = n
        self.noise = noi
        self.amp = np.array([])

    def get_coord_grid(self, num):
        x = np.linspace(-2 * self.side, self.side, num=num)
        y = np.linspace(-self.side, self.side * 2, num=num)
        self.coord_x, self.coord_y = np.meshgrid(x, y)

    def sim_n_times(self, k, j, n):
        """

        :param: k is the index of coord in list
        :param: n is how many times of simulation
        """
        def sim_one_laserpos(self, k, j):

            def is_contain(point_list, b):
                zeros = np.array([])
                for j in range(len(point_list)):
                    zeros = np.append(zeros, b.contains(Point(point_list[j])))
                return zeros

            l = self.box_array
            r = self.radius
            coord = list([np.asarray(self.coord_x)[k][j], np.asarray(self.coord_y)[k][j]])
            n = self.num_points
            noi = self.noise
            s = self.side
            random_noi = np.random.uniform(-noi+1, 1+noi, 1)
            n = math.ceil(random_noi*n)

            random_points_x = np.random.uniform(coord[0] - r, coord[0] + r, n)
            random_points_y = np.random.uniform(coord[1] - r, coord[1] + r, n)
            random_points = list(zip(random_points_x, random_points_y))

            charge_sum_left = np.count_nonzero(is_contain(random_points, l[1]))/n
            charge_sum_center = np.count_nonzero(is_contain(random_points, l[0]))/n
            charge_sum_right = np.count_nonzero(is_contain(random_points, l[5]))/n

            charge_sum_upleft = np.count_nonzero(is_contain(random_points, l[2]))/n
            charge_sum_upcenter = np.count_nonzero(is_contain(random_points, l[3]))/n
            charge_sum_upright = np.count_nonzero(is_contain(random_points, l[4]))/n

            charge_sum_lowleft = np.count_nonzero(is_contain(random_points, l[8]))/n
            charge_sum_lowcenter = np.count_nonzero(is_contain(random_points, l[7]))/n
            charge_sum_lowright = np.count_nonzero(is_contain(random_points, l[6]))/n

            left_a_x = charge_sum_upleft + charge_sum_left + charge_sum_lowleft
            center_a_x = charge_sum_upcenter + charge_sum_center + charge_sum_lowcenter
            right_a_x = charge_sum_upright + charge_sum_right + charge_sum_lowright

            up_a_y = charge_sum_upleft + charge_sum_upcenter + charge_sum_upright
            center_a_y = charge_sum_left + charge_sum_center + charge_sum_right
            low_a_y = charge_sum_lowleft + charge_sum_lowcenter + charge_sum_lowright

            total = up_a_y + center_a_y + low_a_y
            if total == 0:
                r_cons_x = 100
                r_cons_y = 100
            else:
                r_cons_x = (left_a_x * (-1.5*s/2) + center_a_x* (-s/2) + right_a_x * (s/2))/total
                r_cons_y = (up_a_y * (1.5*s) + center_a_y * (s/2) + low_a_y * (-s/2))/total

            delta = ((r_cons_x - coord[0])**2 + (r_cons_y - coord[1])**2 )**0.5
            return delta

        temp = 0
        for i in range(n):
            temp += sim_one_laserpos(self, k, j)
            coord = list([np.asarray(self.coord_x)[k][j], np.asarray(self.coord_y)[k][j]])
            print(coord)
            print(temp)
        self.amp = np.append(self.amp, temp/n)

    def sim_n_coord(self):
        amp_k = np.array([])
            for i in range(j):
                amp_k = sim_n_times(self, )

# test cases
side = 4
radius_uni = 1 # radius of random point around laser pos
n = 1000 # number of points around one laser pos
noi = 0.7 # noise level between 1 and 0
num = 20 # num of laser positions

newPad = myPadArray(side)
newPad.get_one_square_box()
newPad.get_pad_nine()
array5b = newPad.box_array
newSim = sim_anode(newPad, radius_uni, n, noi)
newSim.get_coord_grid(num)
poly5a = array5b[0]
poly5b = array5b[1]
poly5c = array5b[2]
poly5d = array5b[3]
poly5e = array5b[4]
poly5f = array5b[5]
poly5g = array5b[6]
poly5h = array5b[7]
poly5i = array5b[8]
x5a, y5a = poly5a.exterior.xy
x5b, y5b = poly5b.exterior.xy
x5c, y5c = poly5c.exterior.xy
x5d, y5d = poly5d.exterior.xy
x5e, y5e = poly5e.exterior.xy
x5f, y5f = poly5f.exterior.xy
x5g, y5g = poly5g.exterior.xy
x5h, y5h = poly5h.exterior.xy
x5i, y5i = poly5i.exterior.xy
newSim.sim_n_times(0, 0, 10)
print(newSim.amp)
