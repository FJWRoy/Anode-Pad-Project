from AnodeSimulation.myPadArray import myPadArray
from AnodeSimulation.SimAnode import sim_anode
from AnodeSimulation.parameter import dictInput, input_check, display
from Reconstruction.reconstruction import reconstruction
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from joblib import Parallel, delayed
import matplotlib.ticker as plticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from shapely.geometry.polygon import Polygon
from descartes import PolygonPatch
from tqdm import tqdm
import time
import pickle
import numpy as np
import sys

def make():
    pads = myPadArray(float(dictInput['length']))
    if dictInput['shape'] == 'sin':
        pads.modify_one_sin_box(0.01, dictInput['pattern_height'])
    elif dictInput['shape'] == 'nose':
        pads.modify_one_n_box(dictInput['nose_start'], dictInput['nose_end'], dictInput['pattern_height'])
    elif dictInput['shape'] == 'cross':
        pads.modify_one_cross_box()
    elif dictInput['shape'] == '45nose':
        pads.modify_one_45degree_n_box(dictInput['nose_start'], dictInput['nose_end'], dictInput['pattern_height'], dictInput['trapezoid_height'])
    elif dictInput['shape'] == '45wedge':
        pads.modify_one_wedge_n_box(dictInput['nose_start'], dictInput['nose_end'], dictInput['pattern_height'])
    elif dictInput['shape'] == 'square':
        pass
    else:
        print("wrong input pad shape")
        sys.exit(1)
    pads.get_pad_5x5()
    sim = sim_anode()
    sim.get_coord_grid(int(dictInput['laser_positions']), float(dictInput['length']))
    sim.update_end(pads)
    id = plotID()
    if dictInput['read']:
        sim.read_sim(id+"_sim.npy")
    else:
        sim.run_sim(pads, float(dictInput['radius']))
        np.save(id+"_sim.npy",sim.amplitude)
    
    return pads, sim
#Run simulations with differing pad size
def make_step():
    pad = myPadArray(float(dictInput['length']))
    if dictInput['shape'] == 'sin':
        pad.modify_one_sin_box(0.01, dictInput['pattern_height'])
    elif dictInput['shape'] == 'nose':
        pad.modify_one_n_box(dictInput['nose_start'], dictInput['nose_end'], dictInput['pattern_height'])
    elif dictInput['shape'] == 'cross':
        pad.modify_one_cross_box()
    elif dictInput['shape'] == '45nose':
        pad.modify_one_45degree_n_box(dictInput['nose_start'], dictInput['nose_end'], dictInput['pattern_height'], dictInput['trapezoid_height'])
    elif dictInput['shape'] == '45wedge':
        pad.modify_one_wedge_n_box(dictInput['nose_start'], dictInput['nose_end'], dictInput['pattern_height'])
    elif dictInput['shape'] == 'square':
        pass
    else:
        print("wrong input pad shape")
        sys.exit(1)
    pad.get_pad_5x5()
    sims = Parallel(n_jobs = int(dictInput['processes']), verbose = 10)(delayed(sim_job)(i, pad, int(dictInput['laser_positions']), float(dictInput['radius']), float(dictInput['length']), float(dictInput['length_incr'])) for i in range(0,int(dictInput['num_sim'])))
    return pad, sims

def sim_job(i, pads, n, r, l,dl):
    sim = sim_anode()
    sim.get_coord_grid(n,l)
    sim.update_end(pads)
    sim.run_sim(pads, r / (1+i * dl/l))
    return sim

def draw_pattern(a, ax):
    array = a.box_array
    l = list()
    [l.append(i.exterior.xy) for i in array]
    [ax.plot(j,k,'g') for (j,k) in list(l)]
    ax.set_axisbelow(True)
    ax.set_xlabel('x[mm]')
    ax.set_ylabel('y[mm]')
    ax.set_xlim([-1.5*a.side, 1.5*a.side])
    ax.set_ylim([-1.5*a.side, 1.5*a.side])
    ax.text(1, 0, plotDesc(), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color = 'black')
    ax.grid(which='both', axis='both')
    ax.minorticks_on()
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    loc = plticker.MultipleLocator(base = float(dictInput['length'])) # this locator puts ticks at square intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.1, linestyle='-')
    ax.grid(b=True, which='minor', axis='both', color='#000000', alpha=0.1, linestyle='-')
def draw_pattern_colored(a, ax):
    draw_pattern(a,ax)
    pad_patch = PolygonPatch(a.box_array[12])
    ax.add_patch(pad_patch)

def draw_pattern_embed(pad, ax, x1, y1, x2, y2):
    axins = ax.inset_axes([x1, y1, x2, y2])
    l = list()
    [l.append(i.exterior.xy) for i in pad.box_array]
    pad_patch = PolygonPatch(pad.box_array[12])
    axins.add_patch(pad_patch)
    [axins.plot(j,k,'g') for (j,k) in list(l)]
    axins.set_xlim(-1.5*pad.side, 1.5*pad.side)
    axins.set_ylim(-1.5*pad.side, 1.5*pad.side)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    axins.xaxis.set_ticks_position('none') 
    axins.yaxis.set_ticks_position('none') 

def draw_radius(SimAnode, pad, ax):
    draw_pattern(pad, ax)
    ax.add_artist(plt.Circle((SimAnode.middle_point[0], SimAnode.middle_point[1]), float(dictInput['radius']), alpha =0.8, color='crimson'))
    ax.add_artist(plt.Circle((SimAnode.middle_point[0]+pad.side*0.5, SimAnode.middle_point[1]), float(dictInput['radius']), alpha =0.8, color='crimson'))
    ax.add_artist(plt.Circle((SimAnode.middle_point[0]-pad.side*0.5, SimAnode.middle_point[1]), float(dictInput['radius']), alpha =0.8, color='crimson'))
    legend_lst = [Line2D([0], [0], marker='o', color='crimson', label='ring spot', markersize=10), Line2D([0], [0], marker='x', color='b', label='actual ring position', markersize=4)]
    ax.plot(SimAnode.middle_point[0], SimAnode.middle_point[1], c='b', marker='x')
    ax.plot(SimAnode.middle_point[0]+pad.side*0.5, SimAnode.middle_point[1], c='b', marker='x')
    ax.plot(SimAnode.middle_point[0]-pad.side*0.5, SimAnode.middle_point[1], c='b', marker='x')
    ax.legend(loc=1, framealpha=0.7, fontsize='x-small', handles=legend_lst)
    ax.set_xlabel('x[mm]')
    ax.set_ylabel('y[mm]')
    ax.set_xlim([-1.5*pad.side, 1.5*pad.side])
    ax.set_ylim([-1.5*pad.side, 1.5*pad.side])
    #ax.title.set_text('pad shape with coord in mm')
    ax.text(1, 0, plotDesc(), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color = 'black')
    ax.minorticks_on()
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    loc = plticker.MultipleLocator(base = float(dictInput['length'])) # this locator puts ticks at square intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.1, linestyle='-')
    ax.grid(b=True, which='minor', axis='both', color='#000000', alpha=0.1, linestyle='-')

def draw_reconstructed(sim, pad, ax):
    draw_pattern(pad, ax)
    recon_positions = list()
    p_x = list()
    p_y = list()
    id = plotID()
    if dictInput['read']:
        recon_positions = np.load(id+"_reconstruction.npy")
        p_x = [(recon_positions[i][0]/float(dictInput['laser_positions'])-0.5)*float(dictInput['length'])*5 for i in range(len(recon_positions))]
        p_y = [(recon_positions[i][1]/float(dictInput['laser_positions'])-0.5)*float(dictInput['length'])*5 for i in range(len(recon_positions))]
    else:
        rec = reconstruction()
        recon_positions = [rec.reconstruction(sim.amplitude[:, i],sim.amplitude) for i in tqdm(range(int(dictInput['laser_positions'])**2),leave=False, desc='reconstruction')]
        p_x = [(recon_positions[i][0]/float(dictInput['laser_positions'])-0.5)*float(dictInput['length'])*5 for i in range(len(recon_positions))]
        p_y = [(recon_positions[i][1]/float(dictInput['laser_positions'])-0.5)*float(dictInput['length'])*5 for i in range(len(recon_positions))]
        np.save(id+"_reconstruction.npy",recon_positions)
        with open(id+"_reconstruction.csv", 'w') as f:
            for i in range(len(recon_positions)):
                f.write(str(i)+','+str(p_x[i])+','+str(p_y[i])+'\n')
        log = rec.degeneracy_check(recon_positions)
        with open(id+"_reconstruction.log", 'w') as f:
            f.write(log)
    ax.scatter(p_x, p_y, s=10,c='crimson', label='reconstructed ring position')
    ax.scatter(np.array(sim.lst_coord)[:,0], np.array(sim.lst_coord)[:,1], c='blue', marker="_",label='actual ring position')
    ax.legend(loc=1, framealpha=0.7, fontsize='x-small')
    ax.set_xlabel('x[mm]')
    ax.set_ylabel('y[mm]')
    ax.set_xlim([-1.5*pad.side, 1.5*pad.side])
    ax.set_ylim([-1.5*pad.side, 1.5*pad.side])
    #ax.title.set_text('pad shape with coord in mm')
    ax.text(1, 0, plotDesc(), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color = 'black')
    ax.minorticks_on()
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    loc = plticker.MultipleLocator(base = float(dictInput['length'])) # this locator puts ticks at square intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.1, linestyle='-')
    ax.grid(b=True, which='minor', axis='both', color='#000000', alpha=0.1, linestyle='-')
    
def plotID():
    return dictInput['shape']+'_res_'+dictInput['laser_positions']+'x'+dictInput['laser_positions']+'_L_'+dictInput['length']+'_R_'+dictInput['radius']+'_H_'+dictInput['pattern_height']
def plotDesc():
    return "L = "+str(float(dictInput['length'])/float(dictInput['radius'])/2)[0:5]#dictInput['shape']+' '+dictInput['laser_positions']+'x'+dictInput['laser_positions']+'\nside: '+dictInput['length']+'mm radius: '+dictInput['radius']+'mm\npattern height:'+dictInput['pattern_height']+'mm'
#Draw the standard deviation plot from noise data
def draw_sd_colorplot(sim, pad, ax):
    draw_pattern(pad, ax)
    rx = [x for x in range(0, len(sim.coord_x)) if (-1.5*pad.side<=sim.coord_x[x] and sim.coord_x[x]<=1.5*pad.side)]#We are only plotting for the area of interest.
    ry = [y for y in range(0, len(sim.coord_y)) if (-1.5*pad.side<=sim.coord_y[y] and sim.coord_y[y]<=1.5*pad.side)]
    X, Y = np.meshgrid([sim.coord_x[i] for i in rx],[sim.coord_y[i] for i in ry])
    S = list()
    id = plotID()
    if dictInput['read']:
        S = np.load(id+"_sd_colorplot.npy")
    else:
        rec = reconstruction()
        S = [[1000*rec.sd(sim.amplitude, (i,j),float(dictInput['radius']), 5*pad.side/float(dictInput['laser_positions'])) for i in rx] for j in tqdm(ry,leave=False,desc = 'SD calculation y' )]
        np.save(id+"_sd_colorplot.npy",S)
        with open(id+"_sd_colorplot.csv", 'w') as f:
            for x in range(len(rx)):
                for y in range(len(ry)):
                    f.write(str(x+rx[0])+','+str(y+ry[0])+','+str(S[y][x])+'\n')
        with open(id+"_sd_colorplot.log", 'w') as f:
            f.write(rec.print_log())
    maxv = 1000#min(np.amax(S), 1000)
    pc = ax.pcolor(X,Y, S, vmax = maxv)
    cbar = plt.colorbar(pc, ax = ax)
    cbar.set_label('Position Resolution[μm]', rotation=90)
    ax.set_xlabel('x[mm]')
    ax.set_ylabel('y[mm]')
    ax.set_xlim([-1.5*pad.side, 1.5*pad.side])
    ax.set_ylim([-1.5*pad.side, 1.5*pad.side])
    #ax.title.set_text('Standard Deviation of Reconstruction[mm]')
    ax.text(1, 0, plotDesc(), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color = 'white')
    ax.minorticks_on()
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    loc = plticker.MultipleLocator(base = float(dictInput['length'])) # this locator puts ticks at square intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.1, linestyle='-')
    ax.grid(b=True, which='minor', axis='both', color='#000000', alpha=0.1, linestyle='-')
def draw_sd_colorplot_debug(sim, pad, ax):
    draw_pattern(pad, ax)
    rx = [x for x in range(0, len(sim.coord_x)) if (-1.5*pad.side<=sim.coord_x[x] and sim.coord_x[x]<=1.5*pad.side)]#We are only plotting for the area of interest.
    ry = [y for y in range(0, len(sim.coord_y)) if (-1.5*pad.side<=sim.coord_y[y] and sim.coord_y[y]<=1.5*pad.side)]
    mx = [x for x in range(0, len(sim.coord_x)) if (-0.5*pad.side<=sim.coord_x[x] and sim.coord_x[x]<=0.5*pad.side)]#We are only plotting for the area of interest.
    my = [y for y in range(0, len(sim.coord_y)) if (-0.5*pad.side<=sim.coord_y[y] and sim.coord_y[y]<=0.5*pad.side)]
    X, Y = np.meshgrid([sim.coord_x[i] for i in rx],[sim.coord_y[i] for i in ry])
    S = list()
    id = plotID()
    S = np.load(id+"_sd_colorplot.npy")
    outliers = list()
    rad = float(dictInput['radius'])
    radunit = rad/(5*float(dictInput['length']))* float(dictInput['laser_positions'])
    scale = 5* float(dictInput['length']) / float(dictInput['laser_positions'])
    for i in mx:
        for j in my:
            if((S[i][j]<=1 or S[i][j]>4000) and not any((p[0]-i)**2+(p[1]-j)**2<0.7*radunit**2 for p in outliers)):#filtering outliers 
                ax.add_artist(plt.Circle((sim.coord_x[i],sim.coord_y[j]), rad, alpha =0.4, color='crimson'))
                ax.plot(sim.coord_x[i], sim.coord_y[j], c='b', marker='x')
                ax.text(sim.coord_x[i], sim.coord_y[j], str((i - 0.5*len(sim.coord_x))* scale)+','+str((j - 0.5*len(sim.coord_y))* scale))
                outliers.append((i,j))
    maxv = 1000
    pc = ax.pcolor(X,Y, S, vmax = maxv)
    cbar = plt.colorbar(pc, ax = ax)
    cbar.set_label('Position Resolution[μm]', rotation=90)
    ax.set_xlabel('x[mm]')
    ax.set_ylabel('y[mm]')
    ax.set_xlim([-1.5*pad.side, 1.5*pad.side])
    ax.set_ylim([-1.5*pad.side, 1.5*pad.side])
    #ax.title.set_text('Standard Deviation of Reconstruction[mm]')
    ax.text(1, 0, plotDesc(), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color = 'white')
    ax.minorticks_on()
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    loc = plticker.MultipleLocator(base = float(dictInput['length'])) # this locator puts ticks at square intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.1, linestyle='-')
    ax.grid(b=True, which='minor', axis='both', color='#000000', alpha=0.1, linestyle='-')    
def draw_sd_pos(sim, pad, y_offset, ax):
    n = int(dictInput['laser_positions'])
    #ax.title.set_text('SD of reconstruction vs ring positions'+' y='+str(y_offset*5*pad.side/n))
    ax.set_xlabel('x[mm]')
    ax.set_xlim([-1.5*pad.side, 1.5*pad.side])
    ax.set_ylabel('Position Resolution[μm]')
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    loc = plticker.MultipleLocator(base = float(dictInput['length'])) # this locator puts ticks at square intervals
    ax.xaxis.set_major_locator(loc)
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.2, linestyle='-')
    initpoint = n*int(n/2)
    rx = [x for x in range(0, len(sim.coord_x)) if (-1.5*pad.side<sim.coord_x[x] and sim.coord_x[x]<1.5*pad.side)]#We are only plotting for the area of interest.
    rec = reconstruction()
    S = list()
    id = plotID()
    if dictInput['read']:
        S = np.load(id+"_sd_xaxis.npy")
    else:
        S = [1000*rec.sd(sim.amplitude, (i,y_offset + int(n/2)),float(dictInput['radius']), 5*pad.side/float(dictInput['laser_positions'])) for i in range(n)] 
        np.save(id+"_sd_xaxis.npy", S)
    ax.plot(sim.coord_x[rx[0]:rx[len(rx)-1]], S[rx[0]:rx[len(rx)-1]],label='Position Resolution')
    ax.set_ylim(bottom=0)
    draw_pattern_embed(pad, ax, 0, 0, 0.2, 0.2)
    ax.text(1, 0, plotDesc(), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color = 'black')
    with open(id+"_sd_xaxis.csv", 'w') as f:
        for i in range(n):
            f.write(str(sim.coord_x[i])+','+str(S[i])+'\n')
            """
    with open("draw_sd_pos_lastrun.csv", 'a') as f:
        for i in range(n):
            f.write(str(sim.coord_x[i])+','+str(S[i])+'\n')
            """
def save_sd(sims, pad, ax, filename):
    file_object = open(filename+'.csv', 'w')
    L_list = list()
    median_res_list = list()
    min_res_list = list()
    max_res_list = list()
    q3_res_list = list()
    u10_res_list = list()
    l10_res_list = list()
    for i in range(0,int(dictInput['num_sim'])):
        sim = sims[i]
        rx = [x for x in range(0, len(sim.coord_x)) if (-1.5*pad.side<sim.coord_x[x] and sim.coord_x[x]<1.5*pad.side)]#We are only plotting for the area of interest.
        ry = [y for y in range(0, len(sim.coord_y)) if (-1.5*pad.side<sim.coord_y[y] and sim.coord_y[y]<1.5*pad.side)]
        rec = reconstruction()
        meaningful_res = [1000*rec.sd(sim.amplitude, (i,j),float(dictInput['radius']), 5*pad.side/float(dictInput['laser_positions'])) for i in rx for j in ry]
        median_res = np.median(meaningful_res)
        median_res_list.append(median_res)
        """
        min_res = np.min(meaningful_res)
        min_res_list.append(min_res)
        max_res = np.max(meaningful_res)
        max_res_list.append(max_res)
        
        try:
            q3_res = np.percentile(meaningful_res,75)
        except:
            q3_res = float('inf')
        """
        try:
            u10_res = np.percentile(meaningful_res,90)
        except:
            u10_res = float('inf')
        try:
            l10_res = np.percentile(meaningful_res,10)
        except:
            l10_res = float('inf')
        u10_res_list.append(u10_res)
        l10_res_list.append(l10_res)
        #q3_res_list.append(q3_res)
        side = float(dictInput['length'])+i * float(dictInput['length_incr'])
        side0 = float(dictInput['length'])
        L_list.append(side/float(dictInput['radius'])/2)
        file_object.write(str(side/float(dictInput['radius'])))
        file_object.write(',')
        file_object.write(str(side/side0*l10_res))
        file_object.write(',')
        file_object.write(str(side/side0*median_res))
        file_object.write(',')
        file_object.write(str(side/side0*u10_res))
        file_object.write('\n')
    
    ax.set_xlabel('L')
    ax.set_ylabel('Resolution[μm]')
    ax.plot(L_list, l10_res_list, '-v',markersize = 4, label='10th percentile spot')
    ax.plot(L_list, median_res_list,'-o',markersize = 4, label='median')
    ax.plot(L_list, u10_res_list,'-+',markersize = 4, label='90th percentile')
    #ax.plot(L_list, max_res_list,'-o',markersize = 4, label='maximal spot')
    ax.fill_between(L_list, l10_res_list, u10_res_list, color = 'gray')

    ax.legend(loc=1, framealpha=0.5, fontsize='medium')
    maxv = 1000#min(np.amax(u10_res_list), 1000)
    ax.set_ylim(top = maxv, bottom=0)
    ax.set_xlim(left=0)
    draw_pattern_embed(pad, ax, 0, 0.8, 0.2, 0.2)
    ax.text(1, 0, dictInput['length_incr']+'mm '+dictInput['num_sim']+'increments', verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color = 'black')

def construct_table(filename):
    pads = myPadArray(float(dictInput['length']))
    if dictInput['shape'] == 'sin':
        pads.modify_one_sin_box(0.01, dictInput['pattern_height'])
    elif dictInput['shape'] == 'nose':
        pads.modify_one_n_box(dictInput['nose_start'], dictInput['nose_end'], dictInput['pattern_height'])
    elif dictInput['shape'] == 'cross':
        pads.modify_one_cross_box()
    elif dictInput['shape'] == '45nose':
        pads.modify_one_45degree_n_box(dictInput['nose_start'], dictInput['nose_end'], dictInput['pattern_height'], dictInput['trapezoid_height'])
    elif dictInput['shape'] == '45wedge':
        pads.modify_one_wedge_n_box(dictInput['nose_start'], dictInput['nose_end'], dictInput['pattern_height'])
    elif dictInput['shape'] == 'square':
        pass
    else:
        print("wrong input pad shape")
        sys.exit(1)
    pads.get_pad_5x5()
    for i in range(0,int(dictInput['num_sim'])):
        sim = sim_anode()
        sim.get_coord_grid(int(dictInput['laser_positions']),float(dictInput['length']))
        sim.update_end(pads)
        table = sim.run_sim_table(pads, float(dictInput['radius']) / (1+i * float(dictInput['length_incr'])/float(dictInput['length'])))
        with open(filename + '_'+dictInput['shape']+"_ratio_"+str((float(dictInput['length'])+i * float(dictInput['length_incr']))/float(dictInput['radius']))[0:5]+".pickle", 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(table, f, pickle.HIGHEST_PROTOCOL)
 
def draw_amp_pos(SimAnode, pad, y_offset, ax):
    noise_level = 0.011*np.pi*float(dictInput['radius'])**2
    #[ax.axvline(x, linestyle='-', color='darkblue') for x in SimAnode.center_pads]
    n = int(dictInput['laser_positions'])
    #ax.title.set_text('amplitude vs ring positions'+' y='+str(y_offset*5*pad.side/n))
    ax.set_xlabel('x[mm]')
    ax.set_xlim([-1.5*pad.side, 1.5*pad.side])
    ax.set_ylabel('charges on the pad / total charges')
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    loc = plticker.MultipleLocator(base = float(dictInput['length'])) # this locator puts ticks at square intervals
    ax.xaxis.set_major_locator(loc)
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.2, linestyle='-')
    initpoint = n*int(n/2)
    amp_indexed = SimAnode.amplitude[:,(initpoint+y_offset*n):(initpoint+(y_offset+1)*n)]
    ax.plot(SimAnode.coord_x, (amp_indexed[11]+noise_level)/(np.pi*float(dictInput['radius'])**2),label='center-left pad')
    ax.plot(SimAnode.coord_x, (amp_indexed[12]+noise_level)/(np.pi*float(dictInput['radius'])**2),label='center pad')
    ax.plot(SimAnode.coord_x, (amp_indexed[13]+noise_level)/(np.pi*float(dictInput['radius'])**2),label='center-right pad')

    ax.legend(loc=1, framealpha=0.5, fontsize='medium')
    ax.set_ylim(bottom=0)
    draw_pattern_embed(pad, ax, 0, 0, 0.2, 0.2)
    ax.text(1, 0, plotDesc(), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color = 'black')

def draw_amp_noise_ratio_pos(SimAnode, pad, y_offset, ax):
    noise_level = 0.011*np.pi*float(dictInput['radius'])**2
    np.seterr(all='print')
    ax.set_yscale('log')
    #[ax.axvline(x, linestyle='-', color='darkblue') for x in SimAnode.center_pads]
    #ax.title.set_text('amplitude+noise ratio vs ring positions'+' y='+str(y_offset*5*pad.side/float(dictInput['laser_positions'])))
    ax.set_xlabel('x[mm]')
    ax.set_xlim([-1.5*pad.side, 1.5*pad.side])
    ax.set_ylabel('ratio of charge on pads including noise')
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    loc = plticker.MultipleLocator(base = float(dictInput['length'])) # this locator puts ticks at square intervals
    ax.xaxis.set_major_locator(loc)
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.2, linestyle='-')
    n = int(dictInput['laser_positions'])
    initpoint = n*int(n/2)
    amp_indexed = SimAnode.amplitude[:,(initpoint+y_offset*n):(initpoint+(y_offset+1)*n)]
    ax.plot(SimAnode.coord_x, (amp_indexed[11]+noise_level)/(amp_indexed[12]+noise_level),label='center-right pad / center pad')
    ax.plot(SimAnode.coord_x, (amp_indexed[13]+noise_level)/(amp_indexed[12]+noise_level),label='center-left pad / center pad')
    ax.plot(SimAnode.coord_x, (amp_indexed[12]+noise_level)/(amp_indexed[11]+noise_level),label='center pad / center-right pad')
    ax.plot(SimAnode.coord_x, (amp_indexed[12]+noise_level)/(amp_indexed[13]+noise_level),label='center pad / center-left pad')
    ax.legend(loc=1, framealpha=0.5, fontsize='x-small')
    draw_pattern_embed(pad, ax, 0, 0, 0.2, 0.2)
    ax.text(1, 0, plotDesc(), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color = 'black')
#Not used
def draw_res_pos(SimAnode, pad, ax):
    ax.set_xlabel('ring position x[mm]')
    ax.set_ylabel('position resolution [mm]')
    #ax.title.set_text('resolution vs ring positions')
    ax.text(1, 0, plotDesc(), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color = 'black')
    ax.minorticks_on()
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    loc = plticker.MultipleLocator(base = float(dictInput['length'])) # this locator puts ticks at square intervals
    ax.xaxis.set_major_locator(loc)
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.2, linestyle='-')
    ax.grid(which='minor', alpha=0.2, linestyle=':', linewidth='0.5', color='black')
    [ax.axvline(x, linestyle='-', color='darkblue') for x in SimAnode.center_pads]
    ax.plot(SimAnode.coord_x, SimAnode.res)
    ax.set_ylim(bottom=0)
#Not used
def draw_res_central_pos(SimAnode, pad, ax):
    ax.plot(SimAnode.coord_x, SimAnode.res)
    ax.set_xlim(SimAnode.center_pads[0]-0.5,SimAnode.center_pads[3]+0.5)
    l = list(zip(SimAnode.coord_x,SimAnode.res))
    y_max = max([y for (x,y) in l if x >= SimAnode.center_pads[0] and x <= SimAnode.center_pads[3]])
    ax.set_ylim(0,y_max*1.2)
    ax.minorticks_on()
    ax.set_xlabel('ring position x[mm]')
    ax.set_ylabel('position resolution [mm]')
    #ax.title.set_text('resolution vs ring positions for central pad')
    ax.text(1, 0, plotDesc(), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,color = 'black')
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    loc = plticker.MultipleLocator(base = float(dictInput['length'])) # this locator puts ticks at square intervals
    ax.xaxis.set_major_locator(loc)
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.2, linestyle='-')
    ax.grid(which='minor', alpha=0.2, linestyle=':', linewidth='0.5', color='black')
    [ax.axvline(x, linestyle='-', color='darkblue') for x in SimAnode.center_pads]

def draw():
    pad, sim = make()
    
    #fig, axes = plt.subplots(3,2,figsize=(7.5,10))
    #plt.setp(axes.flat, adjustable='box')
    plt.rcParams.update({'font.size': 12})
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    draw_pattern_colored(pad, ax1)
    id = plotID()
    save_plot(fig1, id+"_pattern")
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    draw_radius(sim, pad, ax2)
    save_plot(fig2, id+"_shower")
    #draw_reconstructed(sim,pad,axes[1,0])
    fig3, ax3 = plt.subplots()
    draw_sd_colorplot(sim, pad, ax3)
    save_plot(fig3, id+"_sd_colorplot")
    fig4, ax4 = plt.subplots()
    draw_sd_pos(sim, pad, 0, ax4)
    save_plot(fig4, id+"_sd_xaxis")
    fig5, ax5 = plt.subplots()
    draw_amp_pos(sim, pad, 0, ax5)
    save_plot(fig5, id+"_amp_xaxis")
    fig6, ax6 = plt.subplots()
    draw_amp_noise_ratio_pos(sim, pad, 0, ax6)
    save_plot(fig6, id+"_amp_ratio_xaxis")
    fig7, ax7 = plt.subplots()
    draw_sd_colorplot_debug(sim, pad, ax7)
    save_plot(fig7, id+"_sd_colorplot_debug")
    #draw_amp_noise_ratio_pos(sim, pad, 2, axes[2,0])
    #draw_amp_noise_ratio_pos(sim, pad, 6, axes[2,1])
    #draw_res_pos(sim, pad,axes[2,0])
    #draw_res_central_pos(sim, pad, axes[2,1])
    #save_sd(sim, pad, 0,"lastrun")
def draw_step():
    pad, sims = make_step()
    plt.rcParams.update({'font.size': 12})
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    id = plotID()+'_'+dictInput['length_incr']+'mm_'+dictInput['num_sim']+'steps'
    save_sd(sims, pad, ax1, id)
    
    save_plot(fig1, id+"_sd_dist")
   
    
def save_plot(fig, name):
    fig.savefig(name+'.jpg', bbox_inches='tight', dpi=200)


if __name__ == "__main__":
    # check if inputs are correct
    display()
    try:
        input_check()
    except Exception:
        print("Error: Please enter either y or n")
        sys.exit(1)

    if dictInput['compare']=='yes':
        #draw_multiple(float(dictInput['step']))
        plt.show()
    elif dictInput['save_sims']=='yes':
        draw_step()
    elif dictInput['lookup_table']:
        construct_table(dictInput['lookup_table'])
    else:
        draw()
    plt.show()
