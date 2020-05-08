import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
class reconstruction:
    #This reconstruction is for 3x3 pads. 
    def reconstruction(self, event, lookup_table):
        event = np.array(event)
        lookup_table = np.array(lookup_table)
        # Find maximal amplitude pad from the input. 
        main_pad = np.argmax(event)#Note that the index starts from 0.

        # Move to the sampling point corresponding to the center of that pad. We assume that the pad has the largest amplitude at that sampling point.
        main_pad_center_point = np.argmax(lookup_table[main_pad])

        # variance = [square sum of amplitude difference over all pads]
        var = np.linalg.norm(event - lookup_table[:,main_pad_center_point])

        # To find "adjacent" points, we need to know the topology of the sampling points of the lookup table. Here we assume that its 100*100 grid.
        n = int(np.sqrt(np.size(lookup_table, 1)))
        center_point_x = main_pad_center_point % n
        center_point_y = main_pad_center_point // n

        #Repeat until we reach a local minimum.
        current = (center_point_x,center_point_y)
        search_range = n/2
        stride = int(n/20)
        
        #We are using stride points for optimization.
        stride_points = [(x,y) for x in range(center_point_x-search_range,center_point_x+search_range+1, stride) for y in range(center_point_y-search_range, center_point_y+search_range+1, stride)]
        for p in stride_points:
            # Check if out of bounds:
            if(p[0]<0 or p[0]>= n or p[1]<0 or p[1]>= n):
                continue
            tmp = np.linalg.norm(event - lookup_table[:,p[0]+n*p[1]])#Chi square minimization
            if(var>tmp):
                var = tmp
                current = p
        #We are happy with this result for now.
        
        adjacent_points = [(x,y) for x in range(current[0]-stride,current[0]+stride+1) for y in range(current[1]-stride, current[1]+stride+1)]
        
        for p in adjacent_points:
            # Check if out of bounds:
            if(p[0]<0 or p[0]>= n or p[1]<0 or p[1]>= n):
                continue
            tmp = np.linalg.norm(event - lookup_table[:,p[0]+n*p[1]])#Chi square minimization
            if(var>tmp):
                var = tmp
                current = p
        #We are happy with this result for now.
        
        return current
    #lookup_table = [pad_num] x [sampling points]
    def variance(self, lookup_table, point, radius, scale):
        #We want to calculate deviation of position, using partial derivative of ith pad signal dPi/dx and dPi/dy.
        #The variation of p, calculated from experimental noise data 
        var_p = (0.015*np.pi*(radius ** 2))**2
        #lookup_table = np.array(lookup_table)
        k = self.jacobian_inv(lookup_table, point, scale)
        var_x = var_p * np.linalg.norm(k[0])
        var_y = var_p * np.linalg.norm(k[1])
        return var_x, var_y
    #lookup_table = [pad_num] x [sampling points]
    def sd(self, lookup_table, point, radius, scale):
        vx, vy = self.variance(lookup_table, point, radius, scale)
        return np.sqrt(vx+vy)
    #lookup_table = [pad_num] x [sampling points]
    def jacobian(self, lookup_table, point, scale):
        #lookup_table = np.array(lookup_table)
        n = int(np.sqrt(np.size(lookup_table, 1)))
        #We take the difference of i+1, i-1 instead of i+1, i to prevent biases.
        dPdx = (lookup_table[:,point[0] + 1+n*point[1]] - lookup_table[:,point[0] - 1+n*point[1]])/(2.0*scale)
        dPdy = (lookup_table[:,point[0] +n*(point[1]+1)] - lookup_table[:,point[0]+n*(point[1]-1)])/(2.0*scale)
        return dPdx, dPdy
    def jacobian_inv(self, lookup_table, point, scale):
        dPdx , dPdy = self.jacobian(lookup_table, point, scale)
        j = np.column_stack((dPdx,dPdy))
        s = np.array([[np.linalg.norm(dPdy)**2, (-1)*np.dot(dPdx, dPdy)],[(-1)*np.dot(dPdx, dPdy),np.linalg.norm(dPdx)**2]])
        s = s / np.linalg.det(s)
        return np.dot(s,  np.transpose(j))


    #we can input a lookup table as events. When the input events table is identital to the reference lookup table, we can run degeneracy check on corresponing pad pattern.
    #recon_positipns is a 2d position vector by n * n.
    def degeneracy_check(self, recon_positions):
        log = ""
        #Set the rectangle of interest. We will only check degeneracies in this rectangle.
        n = np.sqrt(np.size(recon_positions, 0))
        xmin = int(n/5)
        xmax = int(n/5*4)#exclusive
        ymin = xmin
        ymax = xmax#exclusive
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                if(recon_positions[x+y*n]!= (x,y)):
                    log+="Degeneracy found at ("+str(x)+","+str(y)+"): reconstucted position is ("+str(recon_positions[x+y*n][0])+","+str(recon_positions[x+y*n][1])+").\n"
        return log


if __name__ == "__main__":
    events = list()
    lookup_table = list()
    # Input
    events_filename = input("Type input event file name:")#Input file format: pickle, list with 25 elements by arbitrary number of events.
    with open(events_filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        events = np.array(pickle.load(f))
    lookup_filename = input("Type lookup table file name:")#Lookup table file format: pickle, list with 25 elements by 10000 sampling points.
    with open(lookup_filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        lookup_table = np.array(pickle.load(f))
    rec = reconstruction()
    recon_positions = [rec.reconstruction(events[:,i], lookup_table) for i in tqdm(range(np.size(events, 1)), leave=False, desc='reconstruction')]
    
    with open(events_filename+".reconstruction.pickle", 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(recon_positions, f, pickle.HIGHEST_PROTOCOL)
    with open(events_filename+".reconstruction.csv", 'w') as f:
        for x in recon_positions:
            f.write(str(x[0])+","+str(x[1])+'\n')
    b = input("Run degeneracy check?")
    if(b=="y"):
        with open(events_filename+".reconstruction.log", 'w') as f:
            f.write(rec.degeneracy_check(recon_positions))

    
