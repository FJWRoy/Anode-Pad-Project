# we use shapely to define pad
from shapely.geometry import LineString, MultiPolygon, MultiPoint, box, Polygon,Point
from shapely.ops import split, unary_union
import numpy as np


class myPadArray:
    def __init__(self, side):

        b = box(-side/2, -side/2, side/2, side/2)
        point = b.centroid
        self.side = side
        self.box = b
        self.box_array = None
        self.center_x = point.x
        self.center_y = point.y
    def grid(self, div, x_ind, y_ind):
        div = float(div)
        side = self.side
        return box(side/div*x_ind-side/(2*div),side/div*y_ind-side/(2*div) ,side/div*x_ind+side/(2*div),side/div*y_ind+side/(2*div)).buffer(side/div/100000)
    def modify_one_n_box(self, start, end, amp):
        """

        modify a nose box
        param: takes
        return: update unit box, update center of pad

        """
        s = self.side
        b = self.box
        start = float(start)
        end = float(end)
        amp = s * float(amp)
        x,y = np.array([start, start, end, end])*s - s/2,np.array([0, amp, amp, 0])-s/2
        line_right = LineString(list(zip(-y,x)))
        line_down = LineString(list(zip(-x,y)))
        poly_left = Polygon(list(zip(-y - s,x)))
        poly_up = Polygon(list(zip(-x, y + s)))
        spl = split(split(b, line_right)[0], line_down)[0]
        poly = unary_union(MultiPolygon([spl, poly_up]))
        new_box = unary_union(MultiPolygon([poly, poly_left]))
        self.box = new_box
        point = new_box.centroid
        self.center_x = point.x
        self.center_y = point.y
    def modify_one_cross_box(self):
        """

        modify a nose box
        param: takes
        return: update unit box, update center of pad

        """
        center_sq = self.grid(5,0,0)
        rect_up = unary_union(MultiPolygon([self.grid(5,0,1),self.grid(5,0,2),self.grid(5,1,2),self.grid(5,2,2),self.grid(5,1,3),self.grid(5,1,4)]))
        rect_down = unary_union(MultiPolygon([self.grid(5,0,-1),self.grid(5,0,-2),self.grid(5,-1,-2),self.grid(5,-2,-2),self.grid(5,-1,-3),self.grid(5,-1,-4)]))
        rect_left = unary_union(MultiPolygon([self.grid(5,1,0),self.grid(5,2,0),self.grid(5,2,-1),self.grid(5,2,-2),self.grid(5,3,-1),self.grid(5,4,-1)]))
        rect_right = unary_union(MultiPolygon([self.grid(5,-1,0),self.grid(5,-2,0),self.grid(5,-2,1),self.grid(5,-2,2),self.grid(5,-3,1),self.grid(5,-4,1)]))
        new_box = unary_union(MultiPolygon([center_sq, rect_up, rect_down, rect_left, rect_right]))
        self.box = new_box
        point = new_box.centroid
        self.center_x = point.x
        self.center_y = point.y

    def modify_one_sin_box(self, step, amp):
        """

        modify a sin wave unit box
        param: takes
        return: update unit box, update center of pad

        """
        s = self.side
        x_range_left = np.arange(0, 0.5*s, step)
        y_range_left = np.sin((x_range_left * np.pi)/ (0.5*s)) * float(amp) *s
        x_range_right = np.arange(0.5*s, s, step)
        y_range_right = -1*y_range_left
        down_left_coords = np.array(list(zip(x_range_left-s, y_range_left)))
        down_right_coords = np.array(list(zip(x_range_right-s, y_range_right)))
        right_down_coords = np.array(list(zip(y_range_right, x_range_left)))
        right_up_coords = np.array(list(zip(y_range_left, x_range_right)))
        down_coords = np.array(list(down_left_coords) + list(down_right_coords) + [(0,0)]) + [s/2, -s/2]
        up_coords = down_coords + [0,s]
        right_coords = np.array(list(right_down_coords) + list(right_up_coords) + [(0,s)]) + [s/2, -s/2]
        left_coords = right_coords + [-s,0]
        b = Polygon(list(down_coords)+list(right_coords)+list(up_coords)[::-1]+list(left_coords)[::-1])
        self.box = b
        point = b.centroid
        self.center_x = point.x
        self.center_y = point.y

    def get_pad_nine(self):
        """

        purpose: return a list of 9 polygons
        """
        s = self.side
        b = self.box
        lists = np.array(list(b.exterior.coords)).tolist()
        off_set = np.array([[-s,s],[0,s],[s,s],[-s,0],[0,0],[s,0],[-s,-s],[0,-s],[s,-s]])
        l_ext = [(x+lists).tolist() for x in off_set]
        list_poly = list([Polygon(x) for x in l_ext])
        self.box_array = list_poly

    def get_pad_5x5(self):
        """

        purpose: return a list of 25 polygons
        """
        s = self.side
        b = self.box
        lists = np.array(list(b.exterior.coords)).tolist()
        off_set = np.array([[-2*s,2*s],[-s,2*s],[0,2*s],[s,2*s],[2*s,2*s],[-2*s,s],[-s,s],[0,s],[s,s],[2*s,s],[-2*s,0],[-s,0],[0,0],[s,0],[2*s,0],[-2*s,-s],[-s,-s],[0,-s],[s,-s],[2*s,-s],[-2*s,-2*s],[-s,-2*s],[0,-2*s],[s,-2*s],[2*s,-2*s]])
        l_ext = [(x+lists).tolist() for x in off_set]
        list_poly = list([Polygon(x) for x in l_ext])
        self.box_array = list_poly


if __name__ == "__main__":
    print("error: myPadArray is running as main")
