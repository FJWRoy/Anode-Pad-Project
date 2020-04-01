from AnodeSimulation.myPadArray import myPadArray
from AnodeSimulation.SimAnode import sim_anode
from AnodeSimulation.parameter import dictInput, input_check, display
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import numpy as np
import sys

def make():
    pads = myPadArray(float(dictInput['length']))
    if dictInput['shape'] == 'sin':
        pads.modify_one_sin_box(0.01, dictInput['sin_height'])
    elif dictInput['shape'] == 'nose':
        pads.modify_one_n_box(dictInput['nose_start'], dictInput['nose_height'])
    elif dictInput['shape'] == 'regular':
        pass
    else:
        print("wrong input pad shape")
        sys.exit(1)
    pads.get_pad_nine()
    sim = sim_anode()
    sim.get_coord_grid(int(dictInput['laser_positions']),list(eval(dictInput['start'])), list(eval(dictInput['end'])))
    sim.run_sim(dictInput['average'], pads, dictInput['radius'], dictInput['charges'], dictInput['uncertainty'], dictInput['noise_mean'], dictInput['noise_variance'])
    return pads, sim

def draw_pattern(a, ax):
    array5b = a.box_array
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
    ax.plot(x5a, y5a, 'g', x5b, y5b, 'g', x5c, y5c, 'g', x5d, y5d, 'g',x5e, y5e, 'g', x5f, y5f, 'g', x5g, y5g, 'g',x5h, y5h, 'g', x5i, y5i, 'g')
    ax.grid(which='both', axis='both')
    ax.set_axisbelow(True)
    ax.set_xlabel('x/mm')
    ax.set_ylabel('y/mm')
    ax.title.set_text('pad shape with coord in mm')

def draw_radius(SimAnode, pad, ax):
    draw_pattern(pad, ax)
    ax.scatter(SimAnode.random_points[:,0], SimAnode.random_points[:,1], s=1, c='crimson', label='charges')
    ax.scatter(SimAnode.middle_point[0], SimAnode.middle_point[1], s=5, c='blue', label='actual laser position')
    ax.legend(loc="best", fontsize='x-small')
    ax.set_xlabel('x/mm')
    ax.set_ylabel('y/mm')
    ax.title.set_text('pad shape with coord in mm')

def draw_reconstructed(sim, pad, ax):
    draw_pattern(pad, ax)
    ax.scatter(sim.reconstructed[:,0], sim.reconstructed[:,1], s=20, marker=1,c='black', label='reconstructed laser position')
    ax.scatter(sim.coord_x, sim.coord_y, c='blue', s=15,label='actual laser position')
    ax.legend(loc="best", fontsize='x-small')
    ax.set_xlabel('x/mm')
    ax.set_ylabel('y/mm')
    ax.title.set_text('pad shape with coord in mm')

def draw_amp_pos(SimAnode, ax):
    ax.plot(SimAnode.coord_x, SimAnode.amplitude[:,0],label='center-left pad')
    ax.plot(SimAnode.coord_x, SimAnode.amplitude[:,1],label='center pad')
    ax.plot(SimAnode.coord_x, SimAnode.amplitude[:,2],label='center-right pad')
    ax.legend(loc="best", fontsize='x-small')
    ax.title.set_text('amplitude vs laser positions')
    ax.set_xlabel('x/mm')
    ax.set_ylabel('number of charges')
    ax.grid()

def draw_res_pos(SimAnode, ax):
    ax.plot(SimAnode.coord_x, SimAnode.res)
    ax.set_xlabel('Laser position x/mm')
    ax.set_ylabel('position resolution /mm')
    ax.title.set_text('resolution vs laser positions')
    ax.grid()

def draw_multiple():
    x_label = np.array([])
    y_list = list()
    x_list = np.array([])
    plt.figure(figsize=(8,8))
    if dictInput['shape']=='sin':
        plt.title('resolution vs laser positions for sin shape')
        range1 = np.arange(0, 0.5, 0.15)
        for i in tqdm(range(len(range1)), desc='compare multiple R/L ratio'):
            dictInput['sin_height']=float(range1[i])
            pad, sim = make()
            y_list.append(list(sim.res))
            x_label = np.append(x_label, [float(range1[i])])
            x_list = sim.coord_x
    elif dictInput['shape']=='nose':
        range1 = np.arange(0, 0.25, 0.1)
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
    [plt.plot(x_list, i, label="R/L ratio "+str(round(j,2))) for i,j in list(zip(y_list, x_label))]
    plt.legend(loc="best", fontsize='x-small')

def draw():
    pad, sim = make()
    fig, axes = plt.subplots(3,2)
    plt.setp(axes.flat, adjustable='box')
    draw_pattern(pad, axes[0,0])
    draw_radius(sim, pad, axes[0,1])
    draw_reconstructed(sim,pad,axes[1,0])
    draw_amp_pos(sim, axes[2,0])
    draw_res_pos(sim, axes[2,1])
    fig.tight_layout()


if __name__ == "__main__":
    # check if inputs are correct
    display()
    try:
        input_check()
    except Exception:
        print("Error: Please enter either y or n")
        sys.exit(1)

    if dictInput['compare']=='yes':
        draw_multiple()
        plt.show()
    else:
        draw()
        plt.show()
