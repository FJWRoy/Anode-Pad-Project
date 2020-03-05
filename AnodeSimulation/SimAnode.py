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


class sim_anode:
    def __init__(self):
        self.box_array = None
        self.side = None
        self.radius = None
        self.coord_x = None
        self.coord_y = None
        self.num_points = None  # how many points around one laser pos
        self.noise = None
        self.amp = None
        self.one_pad_center = None

    def get_parameters(self, padArray, r, n, noi):
        self.box_array = padArray.box_array
        self.side = padArray.side
        self.noise = noi
        self.radius = r
        self.num_points = n  # how many points around one laser pos
        self.one_pad_center = [padArray.center_x, padArray.center_y]

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
            center = self.one_pad_center

            random_noi = np.random.uniform(-noi+1, 1+noi, 1)
            n = math.ceil(random_noi*n)

            random_points_x = np.random.uniform(coord[0] - r, coord[0] + r, n)
            random_points_y = np.random.uniform(coord[1] - r, coord[1] + r, n)
            random_points = list(zip(random_points_x, random_points_y))

            charge_sum_left = np.count_nonzero(is_contain(random_points, l[1]))/n + int(np.random.normal(n/12,n/120))
            charge_sum_center = np.count_nonzero(is_contain(random_points, l[0]))/n+ int(np.random.normal(n/12,n/120))
            charge_sum_right = np.count_nonzero(is_contain(random_points, l[5]))/n+ int(np.random.normal(n/12,n/120))

            charge_sum_upleft = np.count_nonzero(is_contain(random_points, l[2]))/n+ int(np.random.normal(n/12,n/120))
            charge_sum_upcenter = np.count_nonzero(is_contain(random_points, l[3]))/n+ int(np.random.normal(n/12,n/120))
            charge_sum_upright = np.count_nonzero(is_contain(random_points, l[4]))/n+ int(np.random.normal(n/12,n/120))

            charge_sum_lowleft = np.count_nonzero(is_contain(random_points, l[8]))/n+ int(np.random.normal(n/12,n/120))
            charge_sum_lowcenter = np.count_nonzero(is_contain(random_points, l[7]))/n+ int(np.random.normal(n/12,n/120))
            charge_sum_lowright = np.count_nonzero(is_contain(random_points, l[6]))/n+ int(np.random.normal(n/12,n/120))

            left_a_x = charge_sum_upleft + charge_sum_left + charge_sum_lowleft
            center_a_x = charge_sum_upcenter + charge_sum_center + charge_sum_lowcenter
            right_a_x = charge_sum_upright + charge_sum_right + charge_sum_lowright

            up_a_y = charge_sum_upleft + charge_sum_upcenter + charge_sum_upright
            center_a_y = charge_sum_left + charge_sum_center + charge_sum_right
            low_a_y = charge_sum_lowleft + charge_sum_lowcenter + charge_sum_lowright

            total = up_a_y + center_a_y + low_a_y
            if total == 0:
                r_cons_x = self.side
                r_cons_y = self.side
            else:
                r_cons_x = (left_a_x * (center[0] - s) + center_a_x* center[0] + right_a_x * (center[0] + s))/total
                r_cons_y = (up_a_y * (s + center[1]) + center_a_y * center[1] + low_a_y * (center[1] - s))/total

            delta = ((r_cons_x - coord[0])**2 + (r_cons_y - coord[1])**2 )**0.5
            return delta

        temp = 0
        for i in range(n):
            temp += sim_one_laserpos(self, k, j)
            coord = list([np.asarray(self.coord_x)[k][j], np.asarray(self.coord_y)[k][j]])
        return temp/n

    def sim_n_coord(self, n):
        length = len(self.coord_x)
        self.amp = np.empty([length, length])
        for g in range(length):
            print("iteration: %s Total: %s" %(g+1, length))
            amp_k = np.array([])
            for i in range(length):
                print("subiteration: %s/%s" %(i+1, length))
                amp_k = np.append(amp_k, [self.sim_n_times(g, i, n)])
            self.amp[g] = amp_k

    def load_csv(self, string):
        s = self.side
        self.coord_amp = np.empty([self.side, self.side])
        df = pd.read_csv(string, index_col = [0,1])
        df_amp = df.loc[['amp']]
        df_list = df_amp.values.tolist()
        arr = np.array(df_list)
        self.amp = np.where(arr > s, s, arr)

    def output_csv(self, string):
        array_x = self.coord_x
        array_y = self.coord_y
        array_amp = self.amp
        arrays = np.append(np.append(array_x, array_y, axis=0), array_amp, axis=0)
        col = np.arange(len(self.coord_x))
        index_array = [['x_coord', 'y_coord', 'amp'], np.arange(len(self.coord_y)).tolist()]
        ind = pd.MultiIndex.from_product(index_array, names=['coord', 'index'])
        df = pd.DataFrame(arrays.tolist(), index=ind, columns = col)
        df.to_csv(string)


#
#
# # test cases
# side = 6
# radius_uni = 1 # radius of random point around laser pos
# n = 500 # number of points around one laser pos 500
# noi = 0.2 # noise level between 1 and 0 0.2
# num = 60 # num of laser positions 30
# average_num = 5 #how many simulations at one laser pos 5
#
#
# newPad = myPadArray(side)
# newPad.get_one_square_box()
# #newPad.modify_one_o_box(0.25, newPad.side/5) #start at 0.25 end at 0.75, height is 1/5 of the side
# #newPad.modify_one_sin_box(0.25, 0.01, newPad.side/5) #start at 0.25, 0.01 is step, amplitude is side/5
# newPad.calculate_center()
# newPad.get_pad_nine()
#
# newSim = sim_anode()
# newSim.get_parameters(newPad, radius_uni, n, noi)
# newSim.get_coord_grid(num)
# #run simulation
# newSim.sim_n_coord(average_num)
#
# #export data
# newSim.output_csv(r'/Users/roywu/Desktop/git_repo/Anode-Pad-Project/AnodeSimtest_o_123.csv')
# #newSim.output_csv(r'/home/fjwu/cs/henry_sim/Anode-Pad-Project/AnodeSimtest.csv')
# #read data
#
#
#
# #newSim.load_csv('AnodeSimtest_rbox.csv')
# #newSim.load_csv('Anode_o.csv')
#
# print(newSim.coord_x[14])
# print(newSim.coord_y[14])
# print(newSim.amp[14])
#
#
# #draw figures
# fig = plt.figure(figsize = (16, 8))
# #draw pad
# ax = fig.add_subplot(221)
# array5b = newSim.box_array
# poly5a = array5b[0]
# poly5b = array5b[1]
# poly5c = array5b[2]
# poly5d = array5b[3]
# poly5e = array5b[4]
# poly5f = array5b[5]
# poly5g = array5b[6]
# poly5h = array5b[7]
# poly5i = array5b[8]
# x5a, y5a = poly5a.exterior.xy
# x5b, y5b = poly5b.exterior.xy
# x5c, y5c = poly5c.exterior.xy
# x5d, y5d = poly5d.exterior.xy
# x5e, y5e = poly5e.exterior.xy
# x5f, y5f = poly5f.exterior.xy
# x5g, y5g = poly5g.exterior.xy
# x5h, y5h = poly5h.exterior.xy
# x5i, y5i = poly5i.exterior.xy
# plt.plot(x5a, y5a, 'g', x5b, y5b, 'g', x5c, y5c, 'g', x5d, y5d, 'g',x5e, y5e, 'g', x5f, y5f, 'g', x5g, y5g, 'g',x5h, y5h, 'g', x5i, y5i, 'g')
# plt.plot(newSim.coord_x, newSim.coord_y, 'ro', markersize=3)
# plt.title('pad shape with coordinates in mm, red dots as laser positions')
# ax.set_ylabel('Y axis')
# #draw surface graph
# # ax2 = fig.add_subplot(222, projection='3d',sharex=ax)
# # ax2.set_title('Surface plot')
# # s = ax2.plot_surface(newSim.coord_x, newSim.coord_y, newSim.amp, cmap=cm.coolwarm,
# #                                linewidth=0, antialiased=False)
# # plt.colorbar(s, shrink=0.5, aspect=5)
# # ax2.set_xlabel('X axis')
# # ax2.set_ylabel('Y axis')
# # ax2.set_zlabel('Amplitude axis')
# #draw heatmap
# ax3 = fig.add_subplot(223)
# df = pd.DataFrame({'x': np.around(newSim.coord_x.flatten().tolist(), decimals=0), 'amp': newSim.amp.flatten(), 'y': np.around(newSim.coord_y.flatten().tolist(), decimals=0)})
# data_pivoted = df.pivot_table(index='y', columns='x', values='amp')
# ax3 = sns.heatmap(data_pivoted, cmap='Greens')
# ax3.invert_yaxis()
# plt.title("distance between reconstructed laser position and actual laser position in mm")
# plt.xlabel("laser position, x coordinate in mm")
# plt.ylabel("laser position y coordinate in mm")
#
# ax4 = fig.add_subplot(224)
# h = ax4.plot(newSim.coord_x[10], newSim.amp[10])
# ax4.set_xlabel('Laser position x/mm')
# ax4.set_ylabel('position resolution /mm')
# ax4.grid()
# plt.show()
