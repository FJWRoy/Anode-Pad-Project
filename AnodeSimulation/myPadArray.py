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
