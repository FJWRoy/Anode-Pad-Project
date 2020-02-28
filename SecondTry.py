import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import LineString, MultiPolygon, MultiPoint, box, Polygon,Point
from shapely.ops import split,  unary_union
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit
import seaborn as sns
sns.set()


def get_one_square_box(side: int):
    """Create a square box

    :param side: length of a side of the box. coord starts from negative x to 0, positive y to 0
    :return: a box on second quadrant
    """
    b = box(-side, 0.0, 0.0, side)
    return b, side

def modify_random_shape_of_box(b, s):
    """Add testing shape on a box to get a polygon

    :param b: box on second quadrant
    :param s: length of each side of the box
    :return: polygon on second quadrant
    """
    random_tuples_right = np.array([[0, 0], [0, 1]])
    random_tuples_left = np.array([-s, 1])
    random_tuples_up = np.array([0, s])
    random_tuples_down = np.array([[0, 0], [-1, 0]])
    for i in range(s):  # random shape generated
        if i == 0 or i == 1:
            continue
        random_num = np.random.randint(-i+1, 0)
        random_tuples_right = np.vstack([random_tuples_right, np.array([random_num, i])])
        random_tuples_down = np.vstack([random_tuples_down, np.array([-i, -random_num])])
        random_tuples_left = np.vstack([random_tuples_left, np.array([random_num - s, i])])
        random_tuples_up = np.vstack([random_tuples_up, np.array([-i, -random_num + s])])
    # right down is for subtraction from the box, up left for union
    line_right = LineString(np.vstack([random_tuples_right, np.array([0, s])]).tolist())
    line_down = LineString(np.vstack([random_tuples_down, np.array([-s, 0])]).tolist())
    poly_left = Polygon(np.vstack([random_tuples_left, np.array([-s, s])]).tolist())
    poly_up = Polygon(np.vstack([random_tuples_up, np.array([-s, s])]).tolist())
    # cut/combine
    box_after_r = split(b, line_right)  # r is right
    box_after_r_d = split(box_after_r[0], line_down)  # d is down, the first item is desired polygon
    polygons = MultiPolygon([box_after_r_d[0], poly_up])  # intermediate between box_final and poly_left, up
    box_after_up = unary_union(polygons)
    polygons = MultiPolygon([box_after_up, poly_left])
    box_final = unary_union(polygons)
    return box_final

def modify_specific_shape_of_box(b, s, string, amp):
    """

    :param b: box on second quadrant
    :param s: length of each side of the box
    :param string: specific function on side of the box
    :param amp: amplitude of the function
    :return: polygon on second quadrant
    """
    # test case sine wave
    if string == 'sin':
        # input
        step = 0.01
        start = 0.25
        end= 0.75
        x_sin_range_temp = np.arange(s * start, end*s, step)
        #
        x_sin_range = np.append(x_sin_range_temp, np.array(end * s))
        y_range = amp * np.append(np.sin((x_sin_range_temp - start*s) * np.pi / (start * s)), np.array(0))
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
        return box_final
    elif string == 'box':
        # input
        start = 0.25
        end = 0.75
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
        return box_final



def get_pad_array(b, s):
    """

    :param b: an modified box on second quadrant
    :param s: length of the side of the box
    :return: a list of polygon. center to right to SE, S to NE
    """
    lists = np.array(list(b.exterior.coords))
    lists_after_parallel_r = lists + np.array([s, 0])
    b_after_parallel_r = Polygon(lists_after_parallel_r.tolist())
    array = list([b, b_after_parallel_r])
    return array

def get_pad_nine(b, s):
    """

    :param b: an modified box on second quadrant
    :param s: length of the side of the box
    :return: a list of polygon. center to right to SE, S to NE
    """
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

    return array


def simulate_amplitude(l, s, n, r):
    """

    :param l: a list of adjacent pads/polygons
    :param s: length of side of one box
    :param n: the number of random points
    :param r: standard d of the laser
    :return: graph position resolution vs laser position
    """
    path = 2*s+1
    list_a = np.array([])
    charges_sum_right = np.array([])
    charges_sum_center = np.array([])
    center_of_pads = np.array([-s / 2, s / 2, s / 2, s / 2])
    y_axis = np.array([])
    y2_axis = np.array([])

    def is_contain(point_list, b):
        zeros = np.array([])
        for j in range(len(point_list)):
            zeros = np.append(zeros, b.contains(Point(point_list[j])))
        return zeros

    for i in range(path):
        coord = [i - s, int(s/2)]
        charges_center = np.array([])
        charges_right = np.array([])
        for j in range(20):
            random_points_x = np.random.uniform(coord[0] - r/2, coord[0] + r/2, n)
            #noise_x = np.random.uniform(coord[0]-noi/2, coord[0]+noi/2, (n, 1))
            #random_points_x = np.add(noise_x, random_points_x)
            random_points_y = np.random.uniform(coord[1] - r/2, coord[1] + r/2, n)
            #noise_y = np.random.uniform(coord[1]-noi/2, coord[1]+noi/2, (n, 1))
            #random_points_y = np.add(noise_y, random_points_y)

            random_points = list(zip(random_points_x, random_points_y))
            charges_center = np.append(charges_center, np.count_nonzero(is_contain(random_points, l[0])))
            charges_right = np.append(charges_right, np.count_nonzero(is_contain(random_points, l[1])))
        charge_sum_center = np.sum(charges_center) / 20 #/ (np.sum(charges_right) + np.sum(charges_center))
        charge_sum_right = np.sum(charges_right) / 20 #/ (np.sum(charges_right) + np.sum(charges_center))
        charge_sum_center = charge_sum_center  #(charge_sum_center + charge_sum_right)
        charge_sum_right = charge_sum_right  #(charge_sum_center + charge_sum_right)
        #print(i)
        #print(charge_sum_right)
        #print(charges_sum_center)
        y_axis = np.append(charge_sum_center, y_axis)
        y2_axis = np.append(charge_sum_right, y2_axis)
    x_plot = np.arange(-s, s+1)

    plt.plot(x_plot, y_axis, 'ro', x_plot, y2_axis, 'blue')

def simulate_amplitude2(l, s, n, r):
    """

    :param l: a list of adjacent pads/polygons
    :param s: length of side of one box
    :param n: the number of random points
    :param r: standard d of the laser
    :return: graph position resolution vs laser position      twinx
    """
    path = 2*s+1
    list_a = np.array([])
    charges_sum_right = np.array([])
    charges_sum_center = np.array([])
    center_of_pads = np.array([-s / 2, s / 2, s / 2, s / 2])
    y_axis = np.array([])
    y2_axis = np.array([])

    def is_contain(point_list, b):
        zeros = np.array([])
        for j in range(len(point_list)):
            zeros = np.append(zeros, b.contains(Point(point_list[j])))
        return zeros

    for i in range(path):
        coord = [i - s, int(s/2)]
        charges_center = np.array([])
        charges_right = np.array([])
        for j in range(20):
            random_points_x = np.random.uniform(coord[0] - r/2, coord[0] + r/2, n)
            #noise_x = np.random.uniform(coord[0]-noi/2, coord[0]+noi/2, (n, 1))
            #random_points_x = np.add(noise_x, random_points_x)
            random_points_y = np.random.uniform(coord[1] - r/2, coord[1] + r/2, n)
            #noise_y = np.random.uniform(coord[1]-noi/2, coord[1]+noi/2, (n, 1))
            #random_points_y = np.add(noise_y, random_points_y)

            random_points = list(zip(random_points_x, random_points_y))
            charges_center = np.append(charges_center, np.count_nonzero(is_contain(random_points, l[0])))
            charges_right = np.append(charges_right, np.count_nonzero(is_contain(random_points, l[1])))
        charge_sum_center = np.sum(charges_center) / 20 #/ (np.sum(charges_right) + np.sum(charges_center))
        charge_sum_right = np.sum(charges_right) / 20 #/ (np.sum(charges_right) + np.sum(charges_center))
        charge_weight = abs((charge_sum_right**1*6 + charge_sum_center**1 * (-6))/(charge_sum_center**1 + charge_sum_right**1) - coord[0])
        #print(i)
        #print(charge_sum_right)
        #print(charges_sum_center)
        y_axis = np.append(y_axis, charge_weight)
    x_plot = np.arange(-s, s+1)

    plt.plot(x_plot, y_axis, 'ro')

def simulate_amplitude3(l, s, n, r, laser):
    """

    :param l: a list of adjacent pads/polygons
    :param s: length of side of one box
    :param n: the number of random points
    :param r: standard d of the laser
    :return: graph position resolution vs laser position
    """
    y_axis = np.array([])
    k = 20

    def is_contain(point_list, b):
        zeros = np.array([])
        for j in range(len(point_list)):
            zeros = np.append(zeros, b.contains(Point(point_list[j])))
        return zeros

    random_coord_x = np.random.uniform(-2*s-s/4, s+s/4, laser)
    random_coord_y = np.random.uniform(-s-s/4, 2*s+s/4, laser)
    list_coord = list(zip(random_coord_x, random_coord_y))
    divi = k*n

    for z in range(len(list_coord)):
        print(z)
        coord = list_coord[z]
        charges_left = np.array([])
        charges_center = np.array([])
        charges_right = np.array([])

        charges_upleft = np.array([])
        charges_upcenter = np.array([])
        charges_upright = np.array([])

        charges_lowleft = np.array([])
        charges_lowcenter = np.array([])
        charges_lowright = np.array([])

        for j in range(k):
            random_points_x = np.random.uniform(coord[0] - r/2, coord[0] + r/2, n)
                        #noise_x = np.random.uniform(coord[0]-noi/2, coord[0]+noi/2, (n, 1))
                        #random_points_x = np.add(noise_x, random_points_x)
            random_points_y = np.random.uniform(coord[1] - r/2, coord[1] + r/2, n)
                        #noise_y = np.random.uniform(coord[1]-noi/2, coord[1]+noi/2, (n, 1))
                        #random_points_y = np.add(noise_y, random_points_y)

            random_points = list(zip(random_points_x, random_points_y))
            charges_left = np.append(charges_left, np.count_nonzero(is_contain(random_points, l[1])))
            charges_center = np.append(charges_center, np.count_nonzero(is_contain(random_points, l[0])))
            charges_right = np.append(charges_right, np.count_nonzero(is_contain(random_points, l[5])))

            charges_upleft = np.append(charges_upleft, np.count_nonzero(is_contain(random_points, l[2])))
            charges_upcenter = np.append(charges_upcenter, np.count_nonzero(is_contain(random_points, l[3])))
            charges_upright = np.append(charges_upright, np.count_nonzero(is_contain(random_points, l[4])))

            charges_lowleft = np.append(charges_lowcenter, np.count_nonzero(is_contain(random_points, l[8])))
            charges_lowcenter = np.append(charges_lowcenter, np.count_nonzero(is_contain(random_points, l[7])))
            charges_lowright = np.append(charges_lowright, np.count_nonzero(is_contain(random_points, l[6])))

        charge_sum_left = np.sum(charges_left) / divi#/ (np.sum(charges_right) + np.sum(charges_center)
        charge_sum_center = np.sum(charges_center) / divi #/ (np.sum(charges_right) + np.sum(charges_center))
        charge_sum_right = np.sum(charges_right) / divi #/ (np.sum(charges_right) + np.sum(charges_center))

        charge_sum_upleft = np.sum(charges_upleft) / divi #/ (np.sum(charges_right) + np.sum(charges_center)
        charge_sum_upcenter = np.sum(charges_upcenter) / divi #/ (np.sum(charges_right) + np.sum(charges_center))
        charge_sum_upright = np.sum(charges_upright) / divi #/ (np.sum(charges_right) + np.sum(charges_center))

        charge_sum_lowleft = np.sum(charges_lowleft) / divi #/ (np.sum(charges_right) + np.sum(charges_center)
        charge_sum_lowcenter = np.sum(charges_lowcenter) / divi #/ (np.sum(charges_right) + np.sum(charges_center))
        charge_sum_lowright = np.sum(charges_lowright) / divi #/ (np.sum(charges_right) + np.sum(charges_center))

        total = charge_sum_lowright + charge_sum_lowleft + charge_sum_lowcenter + charge_sum_right + charge_sum_center + charge_sum_left + charge_sum_upright + charge_sum_upcenter + charge_sum_upleft

        left_a_x = charge_sum_upleft + charge_sum_left + charge_sum_lowleft
        center_a_x = charge_sum_upcenter + charge_sum_center + charge_sum_lowcenter
        right_a_x = charge_sum_upright + charge_sum_right + charge_sum_lowright

        up_a_y = charge_sum_upleft + charge_sum_upcenter + charge_sum_upright
        center_a_y = charge_sum_left + charge_sum_center + charge_sum_right
        low_a_y = charge_sum_lowleft + charge_sum_lowcenter + charge_sum_lowright


        if total == 0:
            r_cons_x = np.nan
            r_cons_y = np.nan
        else:
            r_cons_x = (left_a_x * (-1.5*s/2) + center_a_x* (-s/2) + right_a_x * (s/2))/total
            r_cons_y = (up_a_y * (1.5*s) + center_a_y * (s/2) + low_a_y * (-s/2))/total

        delta = ((r_cons_x - list_coord[z][0])**2 + (r_cons_y-list_coord[z][1])**2 )**0.5
        y_axis = np.append(delta, y_axis)

    return random_coord_x, random_coord_y, y_axis

def simulate_amplitude4(l, s, n, r, laser):
    """

    :param l: a list of adjacent pads/polygons
    :param s: length of side of one box
    :param n: the number of random points
    :param r: standard d of the laser
    :return: graph position resolution vs laser position
    """
    path = 3*s+7
    list_a = np.array([])
    charges_sum_right = np.array([])
    charges_sum_center = np.array([])
    center_of_pads = np.array([-s / 2, s / 2, s / 2, s / 2])
    y_axis = np.array([])
    k = 20

    def is_contain(point_list, b):
        zeros = np.array([])
        for j in range(len(point_list)):
            zeros = np.append(zeros, b.contains(Point(point_list[j])))
        return zeros

    random_coord_x = np.random.uniform(-2*s-s/4, s+s/4, laser)
    random_coord_y = np.random.uniform(-s-s/4, 2*s+s/4, laser)
    list_coord = list(zip(random_coord_x, random_coord_y))
    divi = k*n

    for z in range(len(list_coord)):
        print(z)
        coord = list_coord[z]
        charges_left = np.array([])
        charges_center = np.array([])
        charges_right = np.array([])

        charges_upleft = np.array([])
        charges_upcenter = np.array([])
        charges_upright = np.array([])

        charges_lowleft = np.array([])
        charges_lowcenter = np.array([])
        charges_lowright = np.array([])

        for j in range(k):
            random_points_x = np.random.uniform(coord[0] - r/2, coord[0] + r/2, n)
                        #noise_x = np.random.uniform(coord[0]-noi/2, coord[0]+noi/2, (n, 1))
                        #random_points_x = np.add(noise_x, random_points_x)
            random_points_y = np.random.uniform(coord[1] - r/2, coord[1] + r/2, n)
                        #noise_y = np.random.uniform(coord[1]-noi/2, coord[1]+noi/2, (n, 1))
                        #random_points_y = np.add(noise_y, random_points_y)

            random_points = list(zip(random_points_x, random_points_y))
            charges_left = np.append(charges_left, np.count_nonzero(is_contain(random_points, l[1])))
            charges_center = np.append(charges_center, np.count_nonzero(is_contain(random_points, l[0])))
            charges_right = np.append(charges_right, np.count_nonzero(is_contain(random_points, l[5])))

            charges_upleft = np.append(charges_upleft, np.count_nonzero(is_contain(random_points, l[2])))
            charges_upcenter = np.append(charges_upcenter, np.count_nonzero(is_contain(random_points, l[3])))
            charges_upright = np.append(charges_upright, np.count_nonzero(is_contain(random_points, l[4])))

            charges_lowleft = np.append(charges_lowcenter, np.count_nonzero(is_contain(random_points, l[8])))
            charges_lowcenter = np.append(charges_lowcenter, np.count_nonzero(is_contain(random_points, l[7])))
            charges_lowright = np.append(charges_lowright, np.count_nonzero(is_contain(random_points, l[6])))

        charge_sum_left = np.sum(charges_left) / divi#/ (np.sum(charges_right) + np.sum(charges_center)
        charge_sum_center = np.sum(charges_center) / divi #/ (np.sum(charges_right) + np.sum(charges_center))
        charge_sum_right = np.sum(charges_right) / divi #/ (np.sum(charges_right) + np.sum(charges_center))

        charge_sum_upleft = np.sum(charges_upleft) / divi #/ (np.sum(charges_right) + np.sum(charges_center)
        charge_sum_upcenter = np.sum(charges_upcenter) / divi #/ (np.sum(charges_right) + np.sum(charges_center))
        charge_sum_upright = np.sum(charges_upright) / divi #/ (np.sum(charges_right) + np.sum(charges_center))

        charge_sum_lowleft = np.sum(charges_lowleft) / divi #/ (np.sum(charges_right) + np.sum(charges_center)
        charge_sum_lowcenter = np.sum(charges_lowcenter) / divi #/ (np.sum(charges_right) + np.sum(charges_center))
        charge_sum_lowright = np.sum(charges_lowright) / divi #/ (np.sum(charges_right) + np.sum(charges_center))

        total = charge_sum_lowright + charge_sum_lowleft + charge_sum_lowcenter + charge_sum_right + charge_sum_center + charge_sum_left + charge_sum_upright + charge_sum_upcenter + charge_sum_upleft

        left_a_x = charge_sum_upleft + charge_sum_left + charge_sum_lowleft
        center_a_x = charge_sum_upcenter + charge_sum_center + charge_sum_lowcenter
        right_a_x = charge_sum_upright + charge_sum_right + charge_sum_lowright

        up_a_y = charge_sum_upleft + charge_sum_upcenter + charge_sum_upright
        center_a_y = charge_sum_left + charge_sum_center + charge_sum_right
        low_a_y = charge_sum_lowleft + charge_sum_lowcenter + charge_sum_lowright


        if total == 0:
            r_cons_x = np.nan
            r_cons_y = np.nan
        else:
            r_cons_x = (left_a_x * (-1.5*s/2) + center_a_x* (-s/2) + right_a_x * (s/2))/total
            r_cons_y = (up_a_y * (1.5*s) + center_a_y * (s/2) + low_a_y * (-s/2))/total

        delta = ((r_cons_x - list_coord[z][0])**2 + (r_cons_y-list_coord[z][1])**2 )**0.5
        y_axis = np.append(delta, y_axis)

    return random_coord_x, random_coord_y, y_axis


# """
# # test closed polygon
# p1 = [(0, 0), (1, 1), (1, 0)]
# poly1 = Polygon(p1)
# print('This is poly1, area, expected 0.5:', poly1.area)
# p2 = [(0, 0), (1, 1), (2, 2), (2, 1), (2, 0), (1, 0)]
# poly2 = Polygon(p2)
# print('This is poly2, area, expected 2:', poly2.area)
# p3 = [(0, 0), (0, 1), (1, 1), (1, 0)]
# poly3 = Polygon(p3)
# print('This is poly3, area, expected 1:', poly3.area)
# p4 = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
# poly4 = Polygon(p4)
# print('This is poly4, area, expected 4:', poly4.area)
# p5 = [(0, 0), (0, -1), (0, -2), (-1, -2), (-2, -2), (-2, -1), (-2, 0)]
# poly5 = Polygon(p5)
# print('This is poly5, area, expected 4:', poly5.area)
# """
# """
# # test box function
# box, length = get_one_square_box(12)
# x, y = box.exterior.xy
# plt.plot(x, y)
# """
# """
# # test modify_random function
# box1, length1 = get_one_square_box(10)
# b_after1 = modify_random_shape_of_box(box1, length)
# x1, y1 = b_after1.exterior.xy
# plt.plot(x1, y1)
# """
# """
# # test get_pad_array
# box2, length2 = get_one_square_box(12)
# b_after2 = modify_random_shape_of_box(box2, length2)
# array2 = get_pad_array(b_after2, length2)
# poly2a = array2[0]
# poly2b = array2[1]
# x2a, y2a = poly2a.exterior.xy
# x2b, y2b = poly2b.exterior.xy
# plt.plot(x2a, y2a, 'r', x2b, y2b, 'g')
# """
# """
# # test simulate_amplitude with random shape
# box3, length3 = get_one_square_box(12)
# b_after3 = modify_random_shape_of_box(box3, length3)
# array3 = get_pad_array(b_after3, length3)
# poly3a = array3[0]
# poly3b = array3[1]
# x3a, y3a = poly3a.exterior.xy
# x3b, y3b = poly3b.exterior.xy
# plt.subplot(1, 2, 1)
# plt.plot(x3a, y3a, 'r', x3b, y3b, 'g')
# plt.title('pad shape with coord in mm')
# plt.subplot(1, 2, 2)
# #simulate_amplitude(array3, length3, 100, length3/2)
# simulate_amplitude2(array3, length3, 1000, length3/2)
# plt.title('f vs laser position, \n 1000 points per position \n normal distribution sd=%s' % length3)
# plt.ylabel('f = A1/(A1+A2)')
# plt.xlabel('laser position in mm')
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.2)
# plt.show()
# simulate_amplitude(array3, length3, 1000, length3/2)
# plt.title('f vs laser position, \n 1000 points per position \n normal distribution sd=%s' % length3)
# plt.ylabel('f = A1/(A1+A2)')
# plt.xlabel('laser position in mm')
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.2)
# """

# test simulate_amplitude with regular box
box5, length5 = get_one_square_box(4)
array5 = get_pad_array(box5, length5)
array5b = get_pad_nine(box5, length5)

# poly5a = array5[0]
# poly5b = array5[1]
# x5a, y5a = poly5a.exterior.xy
# x5b, y5b = poly5b.exterior.xy
# plt.subplot(1, 2, 1)
# plt.plot(x5a, y5a, 'r', x5b, y5b, 'g')
# plt.title('pad shape with coord in mm')
# plt.subplot(1, 2, 2)
# n = 1
# #simulate_amplitude(array5, length5, 100, length5/2)
# simulate_amplitude2(array5, length5, 100, n)
# plt.title('side length = 12, interval = %s' %n)
# plt.ylabel('f = A1*(-5)+A2*5/(A1+A2)')
# plt.xlabel('laser position in mm')
# #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.2)
# plt.show()
#
# simulate_amplitude(array5, length5, 100, n)
# plt.title('side length = 12, interval = %s' %n)
# plt.ylabel('number of charges in each pad')
# plt.xlabel('laser position in mm')
# #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.2)
# plt.show()

n=0.2
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

x, y, z = simulate_amplitude3(array5b, length5, 500, n, 100)
formatted_x = [round(elem) for elem in x ]
formatted_y = [round(elem) for elem in y ]
row_data = {'x': formatted_x, 'y': formatted_y, 'Amp': z, 'row_x': x, 'row_y': y}
df = pd.DataFrame(row_data, columns = ['x','y','Amp','row_x','row_y'])
index_drop = df[df['Amp'].isnull()].index
df.drop(index_drop, inplace=True)
df.to_csv(r'/Users/fjwu/Desktop/git\repo/SecondTry5.csv', index = None, header=True)

# fig = plt.figure(figsize=(6,4))
# ax = fig.add_subplot(212)
#
# data = pd.read_csv("SecondTry5.csv")

# data_pivoted = data.pivot_table(index='y', columns='x', values='Amp')
# sns.heatmap(data_pivoted, cmap='Greens')
# ax.invert_yaxis()
# plt.title("Resolution in mm")
# plt.xlabel("coord_x in mm")
# plt.ylabel("coord_y in mm")
#
# X = data['row_x'].to_numpy()
# Y = data['row_y'].to_numpy()
# Z = data['Amp'].to_numpy()
# col = np.arange(len(Z))
# # ax = plt.axes(projection='3d')
# # ax.plot_trisurf(X, Y, Z, cmap='binary')
# # ax.scatter(X, Y, Z, c=col, depthshade=True)
# # ax.set_xlabel('x')
# # ax.set_ylabel('y')
# # ax.set_zlabel('z');
# ax.scatter(Y,Z,c='b', alpha=0.5)
#
# # Add second axes object
# ax = fig.add_subplot(211)
# plt.plot(x5a, y5a, 'g', x5b, y5b, 'g', x5c, y5c, 'g',
#  x5d, y5d, 'g',x5e, y5e, 'g', x5f, y5f, 'g',
#  x5g, y5g, 'g',x5h, y5h, 'g', x5i, y5i, 'g')
# plt.plot(data['row_x'], data['row_y'], 'ro')
# plt.title('pad shape with coord in mm')
#
# # Make sure the elements of the plot are arranged properly
# plt.tight_layout()
# plt.show()



# n = 7
# #simulate_amplitude(array5, length5, 100, length5/2)
# simulate_amplitude2(array5, length5, 100, n)
# plt.title('side length = 12, interval = %s' %n)
# plt.ylabel('f = A1*(-5)+A2*5/(A1+A2)')
# plt.xlabel('laser position in mm')
# #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.2)
# plt.show()
# simulate_amplitude(array5, length5, 100, n)
# plt.title('side length = 12, interval = %s' %n)
# plt.ylabel('number of charges in each pad')
# plt.xlabel('laser position in mm')
# #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.2)
# plt.show()
