from tabulate import tabulate

class parameter_from_txt():
    def __init__(self):
        # read input from the txt.
        f=open("input.txt", "r")
        contents =f.readlines()
        #define a pad
        self.shape = contents[1].split(":")[1][:-1]
        self.side = float(contents[2].split(":")[1])
        self.nose_start = float(contents[3].split(":")[1])
        self.nose_height_ratio = float(contents[4].split(":")[1])
        self.sin_height_ratio = float(contents[5].split(":")[1])

        #define a Laser
        self.radius_uni = float(contents[8].split(":")[1])
        self.n_times = int(float(contents[9].split(":")[1]))
        self.noise_mean = float(contents[10].split(":")[1])
        self.noise_variance = float(contents[11].split(":")[1])

        #define simulations
        self.average_num = int(float(contents[14].split(":")[1]))
        self.num = int(float(contents[15].split(":")[1]))
        self.start_pos = eval(contents[16].split(":")[1])
        self.end_pos = eval(contents[17].split(":")[1])

        #define output/import
        self.import_data = int(float(contents[20].split(":")[1]))
        self.outport_data = int(float(contents[21].split(":")[1]))
        self.file_name = contents[22].split(":")[1][:-1]

        #draw_graph
        print(tabulate([
        ["DEFINE PADS"],
        ["shape_of_one_pad", self.shape],
        ["length_of_one_pad", self.side],
        ["nose_start", self.nose_start],
        ["nose_height_ratio", self.nose_height_ratio],
        ["sin_height_ratio", self.sin_height_ratio],
        ["DEFINE LASER"],
        ["radius_of_one_laser_spot", self.radius_uni],
        ["number_of_charges_of_laser", self.n_times],
        ["noise_mean", self.noise_mean],
        ["noise_variance", self.noise_variance],
        ["DEFINE SIMULATION"],
        ["#simulation at one laser pos", self.average_num],
        ["number_of_laser_pos", self.num],
        ["start_position", self.start_pos],
        ["end_pos", self.end_pos],
        ["import_data?", self.import_data],
        ["output_data?", self.outport_data],
        ["file_name", self.file_name]],
        ["input", "value", "Notes"], tablefmt="grid"))

    # def check_input(self):
    #     if (self.import_data == self.outport_data):
    #         print("error: input data has error")
    #         return False
    #     else:
    #         return True

    # def load_csv(self):
    #     file_name = input.file_name
    #     df = pd.read_csv(file_name, index_col['amp', 'x_coord', 'y_coord'])
    #     print(df)


if __name__ == "__main__":
    print("error: run parameter as main")
