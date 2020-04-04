from AnodeSimulation.myPadArray import myPadArray
from AnodeSimulation.SimAnode import sim_anode
from AnodeSimulation.parameter import dictInput, input_check, display
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from tqdm import tqdm
import time
import numpy as np
import sys

def make():
    pads = myPadArray(float(dictInput['length']))
    if dictInput['shape'] == 'sin':
        pads.modify_one_sin_box(0.01, dictInput['sin_height'])
    elif dictInput['shape'] == 'nose':
        pads.modify_one_n_box(dictInput['nose_start'], dictInput['nose_end'], dictInput['nose_height'])
    elif dictInput['shape'] == 'regular':
        pass
    else:
        print("wrong input pad shape")
        sys.exit(1)
    pads.get_pad_5x5()
    sim = sim_anode()
    sim.get_coord_grid(int(dictInput['laser_positions']),list(eval(dictInput['start'])), list(eval(dictInput['end'])))
    sim.update_end(pads)
    sim.run_sim(pads, dictInput['radius'])
    return pads, sim

def draw_pattern(a, ax):
    array = a.box_array
    l = list()
    [l.append(i.exterior.xy) for i in array]
    [ax.plot(j,k,'g') for (j,k) in list(l)]
    ax.set_axisbelow(True)
    ax.set_xlabel('x/mm')
    ax.set_ylabel('y/mm')
    ax.title.set_text('pad shape with coord in mm')
    # ax.get_xaxis().set_tick_params(labelsize=15, length=14, width=2, which='major')
    # ax.get_xaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
    # ax.get_yaxis().set_tick_params(labelsize=15,length=14, width=2, which='major')
    # ax.get_yaxis().set_tick_params(labelsize=15,length=7, width=2, which='minor')
    # majorLocatorY = MultipleLocator(5)
    # minorLocatorY = MultipleLocator(1)
    # majorLocatorX = MultipleLocator(.100)
    # minorLocatorX = MultipleLocator(.010)
    # ax.get_xaxis().set_major_locator(majorLocatorX)
    # ax.get_xaxis().set_minor_locator(minorLocatorX)
    # ax.get_yaxis().set_major_locator(majorLocatorY)
    # ax.get_yaxis().set_minor_locator(minorLocatorY)




    ax.grid(which='both', axis='both')
    ax.minorticks_on()
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.1, linestyle='-')
    ax.grid(b=True, which='minor', axis='both', color='#000000', alpha=0.1, linestyle='-')

def draw_radius(SimAnode, pad, ax):
    draw_pattern(pad, ax)
    ax.add_artist(plt.Circle((SimAnode.middle_point[0], SimAnode.middle_point[1]), float(dictInput['radius']), alpha =0.8, color='crimson'))
    legend_lst = [Line2D([0], [0], marker='o', color='crimson', label='laser spot', markersize=10), Line2D([0], [0], marker='x', color='b', label='actual laser position', markersize=4)]
    ax.plot(SimAnode.middle_point[0], SimAnode.middle_point[1], c='b', marker='x')
    ax.legend(loc=1, framealpha=0.7, fontsize='x-small', handles=legend_lst)
    ax.set_xlabel('x/mm')
    ax.set_ylabel('y/mm')
    ax.title.set_text('pad shape with coord in mm')
    ax.minorticks_on()
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.1, linestyle='-')
    ax.grid(b=True, which='minor', axis='both', color='#000000', alpha=0.1, linestyle='-')

def draw_reconstructed(sim, pad, ax):
    draw_pattern(pad, ax)
    ax.scatter(np.array(sim.reconstructed)[:,0], np.array(sim.reconstructed)[:,1], s=10,c='crimson', label='reconstructed laser position')
    ax.scatter(sim.coord_x, sim.coord_y, c='blue', marker="_",label='actual laser position')
    ax.legend(loc=1, framealpha=0.7, fontsize='x-small')
    ax.set_xlabel('x/mm')
    ax.set_ylabel('y/mm')
    ax.title.set_text('pad shape with coord in mm')
    ax.minorticks_on()
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.1, linestyle='-')
    ax.grid(b=True, which='minor', axis='both', color='#000000', alpha=0.1, linestyle='-')

def draw_amp_pos(SimAnode, pad, ax):
    [ax.axvline(x, linestyle='-', color='darkblue') for x in SimAnode.center_pads]
    ax.title.set_text('amplitude vs laser positions')
    ax.set_xlabel('x/mm')
    ax.set_ylabel('charges on the pad / total charges')
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.2, linestyle='-')
    ax.plot(SimAnode.coord_x, np.array(SimAnode.amplitude)[:,0],label='center-left pad')
    ax.plot(SimAnode.coord_x, np.array(SimAnode.amplitude)[:,1],label='center pad')
    ax.plot(SimAnode.coord_x, np.array(SimAnode.amplitude)[:,2],label='center-right pad')
    ax.legend(loc=1, framealpha=0.5, fontsize='x-small')

def draw_res_pos(SimAnode, pad, ax):
    ax.set_xlabel('Laser position x/mm')
    ax.set_ylabel('position resolution /mm')
    ax.title.set_text('resolution vs laser positions')
    ax.minorticks_on()
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.2, linestyle='-')
    ax.grid(which='minor', alpha=0.2, linestyle=':', linewidth='0.5', color='black')
    [ax.axvline(x, linestyle='-', color='darkblue') for x in SimAnode.center_pads]
    ax.plot(SimAnode.coord_x, SimAnode.res)

def draw_res_central_pos(SimAnode, pad, ax):
    ax.plot(SimAnode.coord_x, SimAnode.res)
    ax.set_xlim(SimAnode.center_pads[0]-0.5,SimAnode.center_pads[3]+0.5)
    l = list(zip(SimAnode.coord_x,SimAnode.res))
    y_max = max([y for (x,y) in l if x >= SimAnode.center_pads[0] and x <= SimAnode.center_pads[3]])
    ax.set_ylim(0,y_max*1.2)
    ax.minorticks_on()
    ax.set_xlabel('Laser position x/mm')
    ax.set_ylabel('position resolution /mm')
    ax.title.set_text('resolution vs laser positions for central pad')
    ax.tick_params(which='both', width=3)
    ax.tick_params(which='major', length=5, color='b')
    ax.grid(b=True, which='major', axis='both', color='#000000', alpha=0.2, linestyle='-')
    ax.grid(which='minor', alpha=0.2, linestyle=':', linewidth='0.5', color='black')
    [ax.axvline(x, linestyle='-', color='darkblue') for x in SimAnode.center_pads]

def draw():
    pad, sim = make()
    fig, axes = plt.subplots(3,2,figsize=(7.5,10))
    plt.setp(axes.flat, adjustable='box')
    draw_pattern(pad, axes[0,0])
    draw_radius(sim, pad, axes[0,1])
    draw_reconstructed(sim,pad,axes[1,0])
    draw_amp_pos(sim, pad, axes[1,1])
    draw_res_pos(sim, pad,axes[2,0])
    draw_res_central_pos(sim, pad, axes[2,1])
    fig.tight_layout()

def draw_multiple(step):
    x_label = np.array([])
    y_list = list()
    x_list = np.array([])
    plt.figure(figsize=(8,8))
    if dictInput['shape']=='sin':
        plt.title('resolution vs laser positions for sin shape')
        range1 = np.arange(0, 0.5, step)
        for i in tqdm(range(len(range1)), desc='compare multiple R/L ratio'):
            dictInput['sin_height']=float(range1[i])
            pad, sim = make()
            y_list.append(list(sim.res))
            x_label = np.append(x_label, [float(range1[i])])
            x_list = sim.coord_x
    elif dictInput['shape']=='nose':
        range1 = np.arange(0, 0.4, step)
        plt.title('resolution vs laser positions for nose shape')
        for i in tqdm(range(len(range1)), desc='compare multiple R/L ratio'):
            dictInput['nose_height']=float(range1[i])
            pad, sim = make()
            y_list.append(list(sim.res))
            x_label = np.append(x_label, float(range1[i]))
            x_list = sim.coord_x
    else:
        print("Error: regular shape has no parameter to compare")
        sys.exit(1)
    [plt.plot(x_list, i, label="R/L ratio: "+str(round(j,2))) for i,j in list(zip(y_list, x_label))]
    plt.grid(which='both', axis='both')
    plt.minorticks_on()
    plt.tick_params(which='both', width=3)
    plt.tick_params(which='major', length=5, color='b')
    plt.grid(b=True, which='major', axis='both', color='#000000', alpha=0.1, linestyle='-')
    plt.grid(b=True, which='minor', axis='both', color='#000000', alpha=0.1, linestyle='-')
    plt.legend(loc="best", fontsize='x-small')


if __name__ == "__main__":
    # check if inputs are correct
    display()
    try:
        input_check()
    except Exception:
        print("Error: Please enter either y or n")
        sys.exit(1)

    if dictInput['compare']=='yes':
        draw_multiple(float(dictInput['step']))
        if dictInput['save']:
            plt.savefig(dictInput['save'])
        plt.show()
    else:
        draw()
        if dictInput['save']:
            plt.savefig(dictInput['save'])
        plt.show()
