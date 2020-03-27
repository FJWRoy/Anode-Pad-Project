from AnodeSimulation.myPadArray import myPadArray
from AnodeSimulation.SimAnode import sim_anode
from AnodeSimulation.parameter import dictInput, input_check, display
from matplotlib import pyplot as plt
import sys

def make():
    pads = myPadArray(float(dictInput['length']))
    if dictInput['shape'] == 'sin':
        print("shape is modified to sin")
        print("sin need to be implemneted")
    elif dictInput['shape'] == 'nose':
        print("shape is modified to nose")
        pads.modify_one_n_box(float(dictInput['nose_start']), float(dictInput['length'])*float(dictInput['nose_height']))
    elif dictInput['shape'] == 'regular':
        pass
    else:
        print("wrong input pad shape")
        sys.exit(0)
    pads.get_pad_nine()
    sim = sim_anode()
    sim.get_coord_grid(int(dictInput['laser_positions']),list(eval(dictInput['start'])), list(eval(dictInput['end'])))
    sim.run_sim(int(dictInput['average']), pads, float(dictInput['radius']), int(dictInput['charges']), float(dictInput['uncertainty']), float(dictInput['noise_mean']), float(dictInput['noise_variance']))
    return pads, sim

def draw_pattern(a, t, n, z):
    ax = fig.add_subplot(t,n,z)
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
    plt.plot(x5a, y5a, 'g', x5b, y5b, 'g', x5c, y5c, 'g', x5d, y5d, 'g',x5e, y5e, 'g', x5f, y5f, 'g', x5g, y5g, 'g',x5h, y5h, 'g', x5i, y5i, 'g')
    # plt.plot(SimAnode.coord_x, SimAnode.coord_y, 'ro', markersize=1)
    # plt.scatter(SimAnode.c[:,0],SimAnode.c[:,1])
    # plt.title('pad shape with coordinates in mm, red dots as laser positions')
    # ax.plot(SimAnode.coord_x[-1], SimAnode.coord_y[-1],'ro', markersize=3)
    # ax.scatter(SimAnode.r[:,0], SimAnode.r[:,1])
    # ax.set_ylabel('Y axis')
    # ax.grid()

def draw_amp_pos(SimAnode, t, n, z, show):
    ax5 = fig.add_subplot(t, n, z)
    ax5.plot(SimAnode.coord_x, SimAnode.amplitude[:,0])
    ax5.plot(SimAnode.coord_x, SimAnode.amplitude[:,1])
    ax5.plot(SimAnode.coord_x, SimAnode.amplitude[:,2])
    if (show == 1):
        plt.show()

def draw_radius(SimAnode, pad, t, n, z):
    ax7 = fig.add_subplot(t, n, z)
    ax7.scatter(SimAnode.coord_x[-1], SimAnode.coord_y[-1])
    draw_pattern(pad,t,n,z)
    ax7.scatter(SimAnode.random_points[:,0], SimAnode.random_points[:,1])

def draw_res_pos(SimAnode, t, n, z, show):
    ax4 = fig.add_subplot(t, n, z)
    ax4.plot(SimAnode.coord_x, SimAnode.res)
    # ax4.plot(SimAnode2.coord_x, SimAnode2.res)
    # ax4.plot(SimAnode3.coord_x, SimAnode3.res)
    ax4.set_xlabel('Laser position x/mm')
    ax4.set_ylabel('position resolution /mm')
    ax4.grid()
    if (show == 1):
        plt.show()
#
#
# # test cases
# side = 6
# radius_uni = 1 # radius of random point around laser pos
# n = 500 # number of points around one laser pos 500
# noi = 0.2 # noise level between 1 and 0 0.2
# num = 60 # num of laser positions 30
# average_num = 5 #how many simulations at one laser pos 5
#
#
# newPad = myPadArray(side)
# newPad.get_one_square_box()
# #newPad.modify_one_o_box(0.25, newPad.side/5) #start at 0.25 end at 0.75, height is 1/5 of the side
# #newPad.modify_one_sin_box(0.25, 0.01, newPad.side/5) #start at 0.25, 0.01 is step, amplitude is side/5
# newPad.calculate_center()
# newPad.get_pad_nine()
#
# newSim = sim_anode()
# newSim.get_parameters(newPad, radius_uni, n, noi)
# newSim.get_coord_grid(num)
# #run simulation
# newSim.sim_n_coord(average_num)
#
# #export data
# newSim.output_csv(r'/Users/roywu/Desktop/git_repo/Anode-Pad-Project/AnodeSimtest_o_123.csv')
# #newSim.output_csv(r'/home/fjwu/cs/henry_sim/Anode-Pad-Project/AnodeSimtest.csv')
# #read data
#
#
#
# #newSim.load_csv('AnodeSimtest_rbox.csv')
# #newSim.load_csv('Anode_o.csv')
#
# print(newSim.coord_x[14])
# print(newSim.coord_y[14])
# print(newSim.amp[14])
#
# #draw surface graph
# # ax2 = fig.add_subplot(222, projection='3d',sharex=ax)
# # ax2.set_title('Surface plot')
# # s = ax2.plot_surface(newSim.coord_x, newSim.coord_y, newSim.amp, cmap=cm.coolwarm,
# #                                linewidth=0, antialiased=False)
# # plt.colorbar(s, shrink=0.5, aspect=5)
# # ax2.set_xlabel('X axis')
# # ax2.set_ylabel('Y axis')
# # ax2.set_zlabel('Amplitude axis')
# #draw heatmap
# ax3 = fig.add_subplot(223)
# df = pd.DataFrame({'x': np.around(newSim.coord_x.flatten().tolist(), decimals=0), 'amp': newSim.amp.flatten(), 'y': np.around(newSim.coord_y.flatten().tolist(), decimals=0)})
# data_pivoted = df.pivot_table(index='y', columns='x', values='amp')
# ax3 = sns.heatmap(data_pivoted, cmap='Greens')
# ax3.invert_yaxis()
# plt.title("distance between reconstructed laser position and actual laser position in mm")
# plt.xlabel("laser position, x coordinate in mm")
# plt.ylabel("laser position y coordinate in mm")


if __name__ == "__main__":
    # check if inputs are correct
    display()
    try:
        input_check()
    except Exception:
        print("Error: Please enter either y or n")
        sys.exit(0)
    finally:
        pad, sim = make()
        fig = plt.figure(figsize=(12, 8))
        draw_pattern(pad,2,1,1)
        draw_radius(sim,pad,2,1,2)
        plt.show()