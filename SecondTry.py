import numpy as np
import math
import random
from matplotlib import pyplot as plt
from shapely.geometry import LineString, MultiPolygon, MultiPoint, box, Polygon,Point
from shapely.ops import split,  unary_union


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


def simulate_amplitude(l, s, n, r):
    """

    :param l: a list of adjacent pads/polygons
    :param s: length of side of one box
    :param n: the number of random points
    :param r: standard d of the laser
    :return: graph position resolution vs laser position
    """
    path = 2*s+1
    charges_center = np.array([])
    charges_right = np.array([])
    del_r_center = np.array([])
    del_r_right = np.array([])

    def is_contain(point_list, b):
        zeros = np.array([])
        for j in range(len(point_list)):
            zeros = np.append(zeros, b.contains(Point(point_list[j])))
        return zeros

    for i in range(path):
        coord = [i - s, int(s/2)]
        random_points_x = np.random.normal(coord[0], r, (n, 1))
        random_points_y = np.random.normal(coord[1], r, (n, 1))
        random_points = list(zip(random_points_x, random_points_y))
        charges_center = np.append(charges_center, np.count_nonzero(is_contain(random_points, l[0])))
        charges_right = np.append(charges_right, np.count_nonzero(is_contain(random_points, l[1])))
        # track each point
        if i < s:
            del_r_center = np.append(del_r_center, np.sqrt(
                np.square(np.average(random_points_x + s/2)) + np.square(np.average(random_points_y - s/2))))
        elif i == 0:
            del_r_center = np.append(del_r_center, np.sqrt(
                np.square(np.average(random_points_x)) + np.square(np.average(random_points_y -s/2))))
        else:
            del_r_right = np.append(del_r_right, np.sqrt(
                np.square(np.average(random_points_x - s/2)) + np.square(np.average(random_points_y - s/2))))
    charges_numerator = np.append(np.split(charges_center, [int(path/2)])[0], np.split(charges_right, [int(path/2)])[1])
    charges_denominator = charges_center + charges_right
    y_plot = charges_numerator / charges_denominator
    x_plot = np.arange(-s, s+1)
    # calculate resolution
    del_r = np.array(list(np.append(del_r_center, del_r_right)))
    pos_res = np.multiply(del_r, y_plot)
    plt.plot(x_plot, y_plot, 'ro')
    # plt.plot(x_plot, pos_res, 'ro')
    # plt.show()


"""
# test closed polygon
p1 = [(0, 0), (1, 1), (1, 0)]
poly1 = Polygon(p1)
print('This is poly1, area, expected 0.5:', poly1.area)
p2 = [(0, 0), (1, 1), (2, 2), (2, 1), (2, 0), (1, 0)]
poly2 = Polygon(p2)
print('This is poly2, area, expected 2:', poly2.area)
p3 = [(0, 0), (0, 1), (1, 1), (1, 0)]
poly3 = Polygon(p3)
print('This is poly3, area, expected 1:', poly3.area)
p4 = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
poly4 = Polygon(p4)
print('This is poly4, area, expected 4:', poly4.area)
p5 = [(0, 0), (0, -1), (0, -2), (-1, -2), (-2, -2), (-2, -1), (-2, 0)]
poly5 = Polygon(p5)
print('This is poly5, area, expected 4:', poly5.area)
"""
"""
# test box function
box, length = get_one_square_box(12)
x, y = box.exterior.xy
plt.plot(x, y)
plt.show()
"""
"""
# test modify_random function
box1, length1 = get_one_square_box(10)
b_after1 = modify_random_shape_of_box(box1, length)
x1, y1 = b_after1.exterior.xy
plt.plot(x1, y1)
plt.show()
"""
"""
# test get_pad_array
box2, length2 = get_one_square_box(12)
b_after2 = modify_random_shape_of_box(box2, length2)
array2 = get_pad_array(b_after2, length2)
poly2a = array2[0]
poly2b = array2[1]
x2a, y2a = poly2a.exterior.xy
x2b, y2b = poly2b.exterior.xy
plt.plot(x2a, y2a, 'r', x2b, y2b, 'g')
plt.show()
"""
"""
# test simulate_amplitude with random shape
box3, length3 = get_one_square_box(12)
b_after3 = modify_random_shape_of_box(box3, length3)
array3 = get_pad_array(b_after3, length3)
poly3a = array3[0]
poly3b = array3[1]
x3a, y3a = poly3a.exterior.xy
x3b, y3b = poly3b.exterior.xy
plt.subplot(1, 2, 1)
plt.plot(x3a, y3a, 'r', x3b, y3b, 'g')
plt.title('pad shape with coord in mm')
plt.subplot(1, 2, 2)
simulate_amplitude(array3, length3, 1000, length3/2)
plt.title('f vs laser position, 1000 points per position, normal distribution sd=%s' % (length3/2))
plt.ylabel('f = A1/(A1+A2)')
plt.xlabel('laser position in mm')
plt.show()
"""
"""
# test simulate_amplitude with regular box
box5, length5 = get_one_square_box(12)
array5 = get_pad_array(box5, length5)
poly5a = array5[0]
poly5b = array5[1]
x5a, y5a = poly5a.exterior.xy
x5b, y5b = poly5b.exterior.xy
plt.subplot(1, 2, 1)
plt.plot(x5a, y5a, 'r', x5b, y5b, 'g')
plt.title('pad shape with coord in mm')
plt.subplot(1, 2, 2)
simulate_amplitude(array5, length5, 1000, length5/2)
plt.title('f vs laser position \n 1000 points per position \n normal distribution sd=%s' % (length5/2))
plt.ylabel('f = A1/(A1+A2)')
plt.xlabel('laser position in mm')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.2)
plt.show()
"""

# test simulate_amplitude with specific shape
box4, length4 = get_one_square_box(12)
# b_after4 = modify_specific_shape_of_box(box4, length4, 'sin', 3)
b_after4 = modify_specific_shape_of_box(box4, length4, 'box', 2)
array4 = get_pad_array(b_after4, length4)
poly4a = array4[0]
poly4b = array4[1]
x4a, y4a = poly4a.exterior.xy
x4b, y4b = poly4b.exterior.xy
plt.subplot(1, 2, 1)
plt.plot(x4a, y4a, 'r', x4b, y4b, 'g')
plt.title('pad shape with coord in mm')
plt.subplot(1, 2, 2)
simulate_amplitude(array4, length4, 1000, length4/2)
plt.title('f vs laser position \n 1000 points per position \n normal distribution sd=%s' % (length4/2))
plt.ylabel('f = A1/(A1+A2)')
plt.xlabel('laser position in mm')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.2)
plt.show()

