import numpy as np
import json
#import Space as sp
import SpaceOptim as sp
import matplotlib
import time
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import multiprocessing


def calc_process(params, l_space, queue, queue2):
    space_l = sp.Space(**params["space"])
    l_s, n = l_space
    space_l.load_space(l_s, n)
    work = True
    while work:
        while not queue.empty():
            changes, acb = queue.get()
            space_l.update(changes)
            space_l.set_acboundary(acb)
            ch, b, _ = space_l.grain_grow()
            queue2.put([ch, b])
            #time.sleep(0.01)



def facade(config, nt):
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

    if params["space"]["optymalization_type"] == "CA":
        acb = space.get_acboundary()
    else:
        space.find_boundary()
        acb = space.get_boundary()
    ac = []
    l = len(acb)
    c = int(np.ceil(l / nt))
    for i in range(nt):
        ac.append(acb[i * c:c + i * c])
    queues = []
    queues2 = []
    processes = []
    running = False
    ch_t_s = []
    b = []
    dones = []
    for i in range(nt):
        ch_t_s += [[]]
        dones += [False]
        queues += [multiprocessing.Queue()]
        queues2 += [multiprocessing.Queue()]
        queues[i].put([[], ac[i]])
        processes.append(multiprocessing.Process(target=calc_process, args=(params, (space.show_space(), space.nb_empty()), queues[i], queues2[i])))
    for i in processes:
        i.start()

    time1 = time.time()



    while (np.array(space.show_space()).min() == 0 and params["space"]["optymalization_type"] == "CA") or (params["space"]["optymalization_type"] == "MC" and counter<params["steps"]):
        ch = []
        if params["show"]:
            print(counter, space.nb_empty())

        counter += 1
        b = []
        while not all(dones):
            for i in range(nt):
                if not queues2[i].empty():
                    ch0, b0 = queues2[i].get()
                    ch += [ch0]
                    b += [b0]
                    dones[i] = True
                    if len(ch0)>0:
                        space.update(ch0)
        for i in range(len(queues)):
            for j in range(len(ch)):
                if i != j:
                    ch_t_s[i] += ch[j]

        c = []
        for i in b:
            for j in i:
                c+=[j]
        acb = sorted(set(c))
        ac = []
        l = len(acb)
        c = int(np.ceil(l / nt))

        for i in range(nt):
            ac.append(acb[i * c:c + i * c])
            dones[i] = False
            #space.update(ch[i])
            #ch[i] = []
        for i in range(len(queues)):
            queues[i].put([ch_t_s[i], ac[i]])
        for i in range(nt):
            ch_t_s[i] = []

    time2 = time.time()
    for i in processes:
        i.terminate()
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