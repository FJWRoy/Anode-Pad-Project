import numpy as np
import random
import pandas as pd
import math
from matplotlib import pyplot as plt
from matplotlib import cm
from shapely.geometry import LineString, MultiPolygon, MultiPoint, box, Polygon,Point
from shapely.ops import split, unary_union
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit
import seaborn as sns
import time

class sim_anode:
    def __init__(self):
        self.box_array = None
        self.side = None
        self.radius = None
        self.start_pos = None
        self.end_pos = None
        self.coord_x = None
        self.coord_y = None
        self.num_points = None  # how many points around one laser pos
        self.noise_mean = None
        self.noise_variance = None
        self.res = None
        self.one_pad_center = None
        self.level = 0
        self.amplitude = None
        self.c = None

    def get_parameters(self, padArray, r, n, noise_mean, noise_variance, start_pos, end_pos):
        self.box_array = padArray.box_array
        self.side = padArray.side
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance
        self.radius = r
        self.num_points = n  # how many points around one laser pos
        self.one_pad_center = [padArray.center_x, padArray.center_y]

    def get_coord_grid(self, num):
        #x = np.linspace(-2 * self.side, self.side, num=num)
        #y = np.linspace(-self.side, self.side * 2, num=num)
        self.coord_x = np.linspace(self.start_pos[0], self.end_pos[0], num)
        coeff_x = (self.end_pos[1] - self.start_pos[1]) / (self.end_pos[0] - self.start_pos[0])
        coeff_y = (self.end_pos[1] - self.end_pos[0] * coeff_x)
        self.coord_y = [x*coeff_x + coeff_y for x in self.coord_x]
        #self.coord_x, self.coord_y = np.meshgrid(x, y)

    def sim_n_times(self, k, n):
        """

        :param: k is the index of coord in list
        :param: n is how many times of simulation
        """
        def sim_one_laserpos(self, k):
            def is_contain(point_list, b):
                zeros = np.array([])
                for j in range(len(point_list)):
                    zeros = np.append(zeros, b.contains(Point(point_list[j])))
                return zeros
            start = time.time()
            l = self.box_array
            r = self.radius
            coord = np.array([self.coord_x[k], self.coord_y[k]])
            n = self.num_points
            s = self.side
            center = self.one_pad_center

            n = math.ceil(n*np.random.uniform(-self.level+1, 1+self.level, 1))

            random_points_off = np.random.uniform(- r, r, (n, 2))
            random_points = np.add(coord, random_points_off)

            list_charges = [is_contain(random_points, x).sum() for x in l]
            list_noise = np.random.normal(n*self.noise_mean,n*self.noise_variance,(1, 9))
            final_list = np.add(list_noise, list_charges).reshape((3,3))

            total = final_list.sum()

            if total == 0:
                r_cons_x = self.side
                r_cons_y = self.side
            else:
                r_x_formula = np.array([(center[0] - s), center[0], center[0] + s])/total
                r_y_formula = np.array([(center[1] + s), center[1], center[1]-s])/total

                r_cons_x = np.multiply(final_list, r_x_formula).sum()
                r_cons_y = np.multiply(final_list.T, r_y_formula).sum()

            delta = ((r_cons_x - coord[0])**2 + (r_cons_y - coord[1])**2)**0.5
            end = time.time()
            print(end-start)
            s = np.array([final_list[1,0], final_list[1,1], final_list[1,2]])
            c = np.array([r_cons_x, r_cons_y])
            return delta, s, c

        temp = 0
        temp_b = np.array([[0, 0, 0]])
        temp_c = np.array([[0, 0]])
        for i in range(n):
            a,b,c = sim_one_laserpos(self, k)
            temp += a
            temp_b = np.append(temp_b, [b], axis=0)
            temp_c = np.append(temp_c, [c], axis=0)
        amp = np.sum(temp_b.tolist(), axis=0)/n
        temp_c = np.sum(temp_c.tolist(), axis=0)/n
        return temp/n, amp, temp_c


    def sim_n_coord(self, n):
        start = time.time()
        length = len(self.coord_x)
        self.res = np.array([])
        self.amplitude = np.array([[0, 0, 0]])
        self.c = np.array([[0,0]])
        for g in range(length):
            print("iteration: %s Total: %s" %(g+1, length))
            res, amplitude, c = self.sim_n_times(g, n)
            self.res = np.append(self.res, res)
            self.amplitude = np.append(self.amplitude, [amplitude], axis=0)
            self.c = np.append(self.c, [c], axis=0)
        self.amplitude = np.delete(self.amplitude, 0, 0)
        self.c = np.delete(self.c, 0, 0)
        print(self.c)
        end = time.time()
        print(end-start)


    def output_csv(self,input):
        file_name = input.file_name
        array_x = self.coord_x
        array_y = self.coord_y
        array_amp = self.res
        l = len(self.coord_x)
        d = [array_x, array_y, array_amp]
        df = pd.DataFrame(d, index=["x_coord", "y_coord", "amp"]).T

        d2 = [('shape_of_one_pad',input.shape),
        ('length_of_one_pad',input.side),
        ('nose_start',input.nose_start),
        ('nose_height_ratio',input.nose_height_ratio),
        ('sin_height_ratio',input.sin_height_ratio),
        ('radius_of_one_laser_spot',input.radius_uni),
        ('number_of_charges_of_laser',input.n_times),
        ('noise_mean',input.noise_mean),
        ('noise_variance',input.noise_variance),
        ('average_num',input.average_num),
        ('number_of_laser_pos',input.num),
        ('start_pos',input.start_pos),
        ('end_pos',input.end_pos)]
        l1 = [i[0] for i in d2]
        l2 = [i[1] for i in d2]
        list = [l1, l2]
        df2 = pd.DataFrame(data=list, index=['input', 'value'])
        data = df2.append([df])
        data.to_csv(file_name + ".csv")



if __name__ == "__main__":
    print("error:SimAnode is running as main")
