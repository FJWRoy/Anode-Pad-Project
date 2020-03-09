from AnodeSimulation import myPadArray
from AnodeSimulation import SimAnode
from AnodeSimulation import parameter
from matplotlib import pyplot as plt


def input():
    input = parameter.parameter_from_txt()
    return input

def set_pad(input):
    padArray = myPadArray.myPadArray(input.side)
    padArray.get_one_square_box()
    if (input.shape == "nose"):
        padArray.modify_one_o_box(input.nose_start, input.side * input.nose_height_ratio)
        print("shape is modified to nose")
    elif (input.shape == 'sin'):
        padArray.modify_one_sin_box(0.01, input.side * input.sin_height_ratio)
        print("shape is modified to sin")
    padArray.get_pad_nine()
    padArray.calculate_center()
    return padArray


def set_sim(input, padArray):
    simAnode = SimAnode.sim_anode(padArray, input.radius_uni, input.n_times, input.noise_mean, input.noise_variance, input.start_pos, input.end_pos)
    simAnode2 = SimAnode.sim_anode(padArray, input.radius_uni, input.n_times, input.noise_mean, input.noise_variance/10, input.start_pos, input.end_pos)
    simAnode3 = SimAnode.sim_anode(padArray, input.radius_uni, input.n_times, input.noise_mean, input.noise_variance/100, input.start_pos, input.end_pos)
    simAnode.get_coord_grid(input.num)
    simAnode2.get_coord_grid(input.num)
    simAnode3.get_coord_grid(input.num)
    return simAnode

def run_sim(input, simAnode):
    simAnode.sim_n_coord(input.average_num)
    return simAnode

def draw_pattern(SimAnode, t, n, z, show):
    #draw figures
    #draw pad
    ax = fig.add_subplot(t,n,z)
    array5b = SimAnode.box_array
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
    plt.plot(SimAnode.coord_x, SimAnode.coord_y, 'ro', markersize=1)
    plt.scatter(SimAnode.c[:,0],SimAnode.c[:,1])
    plt.title('pad shape with coordinates in mm, red dots as laser positions')
    ax.scatter(SimAnode.coord_x[-1], SimAnode.coord_y[-1])
    ax.scatter(SimAnode.r[:,0], SimAnode.r[:,1])
    ax.set_ylabel('Y axis')
    ax.grid()
    if (show == 1):
        plt.show()

def draw_amp_pos(SimAnode, t, n, z, show):
    ax5 = fig.add_subplot(t, n, z)
    ax5.plot(SimAnode.coord_x, SimAnode.amplitude[:,0])
    ax5.plot(SimAnode.coord_x, SimAnode.amplitude[:,1])
    ax5.plot(SimAnode.coord_x, SimAnode.amplitude[:,2])
    if (show == 1):
        plt.show()

def draw_radius(SimAnode, t, n, z, show):
    ax7 = fig.add_subplot(t, n, z)
    ax7.scatter(SimAnode.coord_x[-1], SimAnode.coord_y[-1])
    ax7.scatter(SimAnode.r[:,0], SimAnode.r[:,1])
    if (show == 1):
        plt.show()

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
    input = input()
    # if (input != None):
    if (input != None):
        myPad = set_pad(input)
        set_sim = set_sim(input, myPad)
        run_sim(input, set_sim)
        #set_sim.output_csv(input)
        fig = plt.figure(figsize=(12,8))
        draw_pattern(set_sim, 2, 2, 1 ,0)
        draw_amp_pos(set_sim, 2, 2, 2, 0)
        #draw_radius(set_sim, 2, 2, 4, 0)
        draw_res_pos(set_sim, 2, 2, 3, 1)
    else:
        print("error input is None")
