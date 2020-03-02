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


class myPadArray:
    def __init__(self, side):
        self.side = side
        self.box = None
        self.box_array = None
        self.center_x = None
        self.center_y = None

    # return a box on Second Quadrant. dont change
    def get_one_square_box(self):
        """
        Create a square box

        :param side: length of a side of the box. coord starts from negative x to 0, positive y to 0
        :return: a box on second quadrant
        """

        b = box(-self.side, 0.0, 0.0, self.side)
        self.box = b

    def modify_one_o_box(self, start, amp):
        """
        Create a o box

        """
        s = self.side
        b = self.box
        end = 1 - start
        list_turning_point_x = np.array([start, start, end, end]) * s
        list_turning_point_y = np.array([0, amp, amp, 0])
        box_right = list(zip(-list_turning_point_y, list_turning_point_x))
        box_left = list(zip(-list_turning_point_y - s, list_turning_point_x))
        box_down = list(zip(-list_turning_point_x, list_turning_point_y))
        box_up = list(zip(-list_turning_point_x, list_turning_point_y + s))
        # right down is for subtraction from the box, up left for union
        line_right = LineString(box_right)
        line_down = LineString(box_down)
        poly_left = Polygon(box_left)
        poly_up = Polygon(box_up)
        # cut/combine
        box_after_r = split(b, line_right)  # r is right
        box_after_r_d = split(box_after_r[0], line_down)  # d is down, the first item is desired polygon
        polygons = MultiPolygon([box_after_r_d[0], poly_up])  # intermediate between box_final and poly_left, up
        box_after_up = unary_union(polygons)
        polygons = MultiPolygon([box_after_up, poly_left])
        box_final = unary_union(polygons)
        self.box = box_final

    def modify_one_sin_box(self, start, step, amp):
        b = self.box
        s = self.side
        end= 1-start
        x_sin_range_temp = np.arange(s * start, end*s, step)

        x_sin_range = np.append(x_sin_range_temp, np.array(end * s))
        y_range = amp * np.append(np.sin((x_sin_range_temp - start*s) * 0.5 *  np.pi / (start * s)), np.array(0))
        sin_right = np.vstack([np.array([0, 0]),
                              np.vstack([np.array(list(zip(-1 * y_range, x_sin_range))),
                                         np.array([0, s])])])
        sin_left = np.array(list(zip(-1 * y_range - s, x_sin_range)))
        sin_down = np.vstack([np.array([0, 0]),
                             np.vstack([np.array(list(zip(-1 * x_sin_range, y_range))),
                                       np.array([-s, 0])])])
        sin_up = np.array(list(zip(-1 * x_sin_range, y_range + s)))
        # right down is for subtraction from the box, up left for union
        line_right = LineString(sin_right.tolist())
        line_down = LineString(sin_down.tolist())
        poly_left = Polygon(sin_left.tolist())
        poly_up = Polygon(sin_up.tolist())
        # cut/combine
        box_after_r = split(b, line_right)  # r is right
        box_after_r_d = split(box_after_r[0], line_down)  # d is down, the first item is desired polygon
        polygons = MultiPolygon([box_after_r_d[0], poly_up])  # intermediate between box_final and poly_left, up
        box_after_up = unary_union(polygons)
        polygons = MultiPolygon([box_after_up, poly_left])
        box_final = unary_union(polygons)
        self.box = box_final

    def calculate_center(self): #calculate the position of the center of the box
        point = self.box.centroid
        self.center_x = point.x
        self.center_y = point.y

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


# test cases
side = 6
radius_uni = 1 # radius of random point around laser pos
n = 500 # number of points around one laser pos
noi = 0.2 # noise level between 1 and 0
num = 30 # num of laser positions
average_num = 10 #how many simulations at one laser pos


newPad = myPadArray(side)
newPad.get_one_square_box()
#newPad.modify_one_o_box(0.25, newPad.side/5) #start at 0.25 end at 0.75, height is 1/4 of the side
newPad.modify_one_sin_box(0.25, 0.01, newPad.side/5) #start at 0.25, 0.01 is step, amplitude is side/5
newPad.calculate_center()
newPad.get_pad_nine()

newSim = sim_anode()
newSim.get_parameters(newPad, radius_uni, n, noi)
newSim.get_coord_grid(num)
#run simulation
newSim.sim_n_coord(average_num)


#export data
newSim.output_csv(r'/Users/roywu/Desktop/git_repo/Anode-Pad-Project/AnodeSimtest_sbox.csv')
#newSim.output_csv(r'/home/fjwu/cs/henry_sim/Anode-Pad-Project/AnodeSimtest.csv')
#read data

#newSim.load_csv('AnodeSimtest_obox.csv')

#draw figures
fig = plt.figure(figsize = (16, 9))
#draw pad
ax = fig.add_subplot(221)
array5b = newSim.box_array
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
plt.plot(x5a, y5a, 'g', x5b, y5b, 'g', x5c, y5c, 'g', x5d, y5d, 'g',x5e, y5e, 'g', x5f, y5f, 'g', x5g, y5g, 'g',x5h, y5h, 'g', x5i, y5i, 'g')
plt.plot(newSim.coord_x, newSim.coord_y, 'ro', markersize=3)
plt.title('pad shape with coord in mm')
#draw surface graph
ax2 = fig.add_subplot(222, projection='3d',sharex=ax,sharey=ax)
ax2.set_title('Surface plot')
s = ax2.plot_surface(newSim.coord_x, newSim.coord_y, newSim.amp, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
plt.colorbar(s, shrink=0.5, aspect=5)
#draw heatmap
ax3 = fig.add_subplot(223)
df = pd.DataFrame({'x': np.around(newSim.coord_x.flatten().tolist(), decimals=0), 'amp': newSim.amp.flatten(), 'y': np.around(newSim.coord_y.flatten().tolist(), decimals=0)})
data_pivoted = df.pivot_table(index='y', columns='x', values='amp')
ax3 = sns.heatmap(data_pivoted, cmap='Greens')
plt.title("Resolution in mm")
plt.xlabel("coord_x in mm")
plt.ylabel("coord_y in mm")

plt.show()
