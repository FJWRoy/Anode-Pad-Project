import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
class reconstruction:
    def __init__(self):
        self.log = ""
        np.seterr(all='raise')
    
    #This reconstruction is for n x n pads, with the sampling points (kn+1) x (kn+1).
    def reconstruction(self, event, lookup_table):
        event = np.array(event)
        lookup_table = np.array(lookup_table)
        # Find maximal amplitude pad from the input. 
        table_size = np.size(lookup_table[0])
        n = int(np.sqrt(table_size))
        ind = np.argpartition(event, -4)[-4:]
        #We start from the point where two largest-signal pads have lar
        main_pad_center_point = min(range(table_size), key=lambda i: ((event[ind[0]] - lookup_table[ind[0]][i]) ** 2 + (event[ind[1]] - lookup_table[ind[1]][i]) ** 2 + (event[ind[2]] - lookup_table[ind[2]][i]) ** 2 + (event[ind[3]] - lookup_table[ind[3]][i]) ** 2))
        return (main_pad_center_point % n, main_pad_center_point // n)

    #lookup_table = [pad_num] x [sampling points]
    def variance(self, lookup_table, point, radius, scale):
        #We want to calculate deviation of position, using partial derivative of ith pad signal dPi/dx and dPi/dy.
        #The variation of p, calculated from experimental noise data 
        var_p = (0.02)**2
        #lookup_table = np.array(lookup_table)
        k = self.jacobian_inv(lookup_table, point, scale)
        var_x = var_p * np.linalg.norm(k[0])
        var_y = var_p * np.linalg.norm(k[1])
        if(var_x <0.001 or np.isnan(var_x)):#If less then 1 micron
            var_x = float('inf')
        if(var_y <0.001 or np.isnan(var_y)):
            var_x = float('inf')
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
        n = int(np.sqrt(np.size(lookup_table, 1)))
        j = np.column_stack((dPdx,dPdy))
        s = np.array([[np.linalg.norm(dPdy)**2, (-1)*np.dot(dPdx, dPdy)],[(-1)*np.dot(dPdx, dPdy),np.linalg.norm(dPdx)**2]])
        try:
            s = s / np.linalg.det(s)
        except:
            if(np.size(dPdx)==25):
                self.log+= 'Degeneracy at ('+str((point[0]-0.5*n)*scale)+','+str((point[1]-0.5*n)*scale)+'):\ndPdx:\n'+np.array_str(np.reshape(dPdx, (5,5)))+'\ndPdy:\n'+np.array_str(np.reshape(dPdy, (5,5)))+'\n'
            else:
                self.log+= 'Degeneracy at ('+str((point[0])*scale)+','+str((point[1])*scale)+'):\ndPdx:\n'+np.array_str(dPdx)+'\ndPdy:\n'+np.array_str(dPdy)+'\n'
            
        return np.dot(s,  np.transpose(j))
    def clear_log(self):
        self.log = ""
    def print_log(self):
        return self.log


    #we can input a lookup table as events. When the input events table is identital to the reference lookup table, we can run degeneracy check on corresponing pad pattern.
    #recon_positipns is a 2d position vector by n * n.
    def degeneracy_check(self, recon_positions):
        log = ""
        #Set the rectangle of interest. We will only check degeneracies in this rectangle.
        n = int(np.sqrt(np.size(recon_positions, 0)))
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

    
