import numpy as np
import time
import math
from shapely.geometry import LineString, MultiPolygon, MultiPoint, box, Polygon,Point

class sim_anode:
    def __init__(self):
        self.coord_x = None
        self.coord_y = None
        self.res = None
        self.amplitude = None
        self.center_amps = None
        self.random_points = None

    def get_coord_grid(self, n_lasers, start, end):
        self.coord_x = np.linspace(start[0], end[0], n_lasers)
        coeff_x = (end[1] - start[1]) / (end[0] - start[0])
        coeff_y = (end[1] - end[0] * coeff_x)
        self.coord_y = [x*coeff_x + coeff_y for x in self.coord_x]

    def run_sim(self, average, myPadArray, radius, charges, uncertainty, mean, variance):
        def sim_n_times(k,avergae):
            def sim_one_laserpos(k):

                def is_contain(point_list, b):
                    zeros = np.array([])
                    for j in range(len(point_list)):
                        zeros = np.append(zeros, b.contains(Point(point_list[j])))
                    return zeros

                start = time.time()
                coord = [self.coord_x[k], self.coord_y[k]]
                center = [myPadArray.center_x, myPadArray.center_y]
                cha = math.ceil(charges*np.random.uniform(-uncertainty+1, 1+uncertainty, 1))

                theta = 2 * np.pi * np.random.uniform(0, 1, cha)
                r = radius *  (np.random.uniform(0,1,cha))**0.5
                x = r*np.cos(theta)
                y = r*np.sin(theta)
                random_points = np.stack((x,y), axis=1)

                list_charges = [is_contain(random_points, x).sum() for x in myPadArray.box_array]
                list_noise = np.random.normal(cha*mean,cha*variance,(1, 9))
                final_list = np.add(list_noise, list_charges).reshape((3,3))
                total = final_list.sum()
                try:
                    r_x_formula = np.array([(center[0] - myPadArray.side), center[0], center[0] + myPadArray.side])/total
                    r_y_formula = np.array([(center[1] + myPadArray.side), center[1], center[1]-myPadArray.side])/total

                    r_cons_x = np.multiply(final_list, r_x_formula).sum()
                    r_cons_y = np.multiply(final_list.T, r_y_formula).sum()
                except ZeroDivisorError:
                    r_cons_x = self.side
                    r_cons_y = self.side
                    pass

                delta = ((r_cons_x - coord[0])**2 + (r_cons_y - coord[1])**2)**0.5
                center_amps = np.array([final_list[1,0], final_list[1,1], final_list[1,2]])
                res_r = np.array([r_cons_x, r_cons_y])
                if k == math.ceil(len(self.coord_x)/2):
                    self.random_points = random_points
                return delta, center_amps, res_r

            temp = 0
            temp_b = np.array([[0, 0, 0]])
            temp_c = np.array([[0, 0]])
            for i in range(average):
                delta,b,c = sim_one_laserpos(k)
                temp += delta
                temp_b = np.append(temp_b, [b], axis=0)
                temp_c = np.append(temp_c, [c], axis=0)
            amp = np.sum(temp_b.tolist(), axis=0)/average
            temp_c = np.sum(temp_c.tolist(), axis=0)/average
            return temp/average, amp, temp_c
        length = len(self.coord_x)
        self.res = np.array([])
        self.amplitude = np.array([[0, 0, 0]])
        self.center_amps = np.array([[0,0]])
        for g in range(length):
            print("iteration: %s Total: %s" %(g+1, length))
            res, amplitude, c = sim_n_times(g, average)
            self.res = np.append(self.res, res)
            self.amplitude = np.append(self.amplitude, [amplitude], axis=0)
            self.center_amps = np.append(self.center_amps, [c], axis=0)
        self.amplitude = np.delete(self.amplitude, 0, 0)
        print(time.time()-start)

if __name__ == "__main__":
    print("error:SimAnode is running as main")
