import numpy as np
import json
import Space as sp
#import SpaceOptim as sp
#import SpaceOptimProcessed as sp
import matplotlib
import time
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def facade(config):
    time0 = time.time()

    file = open(config, 'r')
    params = json.loads(file.read())
    file.close()



    n_start = params["n_start_seeds"]
    space = sp.Space(**params["space"])



    if params["load_start"]:
        file = open(params["load_file_name"], 'r')
        load = json.loads(file.read())
        file.close()
        space.load_space(np.array(load), n_start)
    else:
        space.seed_random_seeds(params["n_start_seeds"])

    if params["save_start"]:
        file = open(params["save_start_file_name"]+config, "w+")
        file.write(json.dumps(space.show_space().tolist()))
        file.close()

    if params["show"]:
        plt.imshow(space.show_space().astype(np.uint8))
        plt.colorbar()
        plt.show()
    counter = 0
    time1 = time.time()
    if params["space"]["optymalization_type"] == "CA":
        while space.nb_empty() != 0:
            if params["show"]:
                print(counter, space.nb_empty())
            counter += 1
            space.grain_grow()
    else:
        space.find_boundary()
        for i in range(params["steps"]):
            space.grain_grow()
    time2 = time.time()
    if params["show"]:
        plt.imshow(space.show_space().astype(np.uint8))
        plt.colorbar()
        plt.show()
    if params["save_results"]:

        if params["space"]["optymalization_type"] == "CA":
            file = open(params["save_results_file_name"]+config, "w+")
        else:
            file = open(params["save_results_file_name_2"]+config, "w+")
        file.write(json.dumps(space.show_space().tolist()))
        file.close()
    time3 = time.time()

    if params["save_time"]:
        file = open(params["save_time_file_name"]+config, "w+")
        file.write("initializing " + str(time1-time0) + "s\n" + "computing " + str(time2-time1) + "s\n" + "finishing " + str(time3-time2) + "s")
        file.close()

    print("initializing " + str(time1-time0) + "s")
    print("computing " + str(time2-time1) + "s")
    print("finishing " + str(time3-time2) + "s")






