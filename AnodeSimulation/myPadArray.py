
# we use shapely to define pad
from shapely.geometry import LineString, MultiPolygon, MultiPoint, box, Polygon,Point
from shapely.ops import split, unary_union
import numpy as np



class myPadArray:
    def __init__(self, side):
        b = box(-side, 0.0, 0.0, side)
        point = b.centroid
        self.side = side
        self.box = b
        self.box_array = None
        self.center_x = point.x
        self.center_y = point.y

    def modify_one_n_box(self, start, amp):
        """
        Create a o box

        """
        s = self.side
        b = self.box
        end = 1 - start
        x,y = np.array([start, start, end, end])*s,np.array([0, amp, amp, 0])
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

    # def modify_one_sin_box(self, step, amp):
    #     b = self.box
    #     s = self.side
    #     step = 0.01
    #     start = 0.25
    #     end= 0.75
    #     x_sin_range_temp = np.arange(s * start, end*s, step)
    #     #
    #     x_sin_range = np.append(x_sin_range_temp, np.array(end * s))
    #     y_range = amp * np.append(np.sin((x_sin_range_temp - start*s) * np.pi / (start * s)), np.array(0))
    #     sin_right = np.vstack([np.array([0, 0]),
    #                           np.vstack([np.array(list(zip(-1 * y_range, x_sin_range))),
    #                                      np.array([0, s])])])
    #     sin_left = np.array(list(zip(-1 * y_range - s, x_sin_range)))
    #     sin_down = np.vstack([np.array([0, 0]),
    #                          np.vstack([np.array(list(zip(-1 * x_sin_range, y_range))),
    #                                    np.array([-s, 0])])])
    #     sin_up = np.array(list(zip(-1 * x_sin_range, y_range + s)))
    #     # right down is for subtraction from the box, up left for union
    #     line_right = LineString(sin_right.tolist())
    #     line_down = LineString(sin_down.tolist())
    #     poly_left = Polygon(sin_left.tolist())
    #     poly_up = Polygon(sin_up.tolist())
    #     # cut/combine
    #     box_after_r = split(b, line_right)  # r is right
    #     box_after_r_d = split(box_after_r[0], line_down)  # d is down, the first item is desired polygon
    #     polygons = MultiPolygon([box_after_r_d[0], poly_up])  # intermediate between box_final and poly_left, up
    #     box_after_up = unary_union(polygons)
    #     polygons = MultiPolygon([box_after_up, poly_left])
    #     box_final = unary_union(polygons)
    #     return box_final


    def get_pad_nine(self):
        """

        purpose: return a list of 9 polygons
        """
        s = self.side
        b = self.box
        lists = np.array(list(b.exterior.coords)).tolist()
        off_set = np.array([[-s,s],[0,s],[s,s],[-s,0],[0,0],[s,0],[s,-s],[0,-s],[-s,-s]])
        l_ext = [(x+lists).tolist() for x in off_set]
        list_poly = list([Polygon(x) for x in l_ext])
        self.box_array = list_poly


if __name__ == "__main__":
    print("error: myPadArray is running as main")
