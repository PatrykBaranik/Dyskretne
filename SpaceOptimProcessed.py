import numpy as np
from typing import Dict
import collections
import random
from multiprocessing import Process
from multiprocessing import Queue
import multiprocessing

def most_frequent(lista):
    occurence_count = collections.Counter(lista)
    return occurence_count.most_common(1)[0][0]

def dump_queue(q):
    l = []
    while not q.empty():
        l += [(q.get())]
    return l

def monte_test(val, neighborhood):
    nval = random.sample(neighborhood, 1)[0]
    if nval == val:
        return val
    prev = 0
    newv = 0
    for i in neighborhood:
        if val == i:
            prev += 1
        if nval == i:
            newv += 1
    if newv > prev:
        return nval
    else:
        return val


def monte_check(val, neighborhood):
    if neighborhood == []:
        return False
    col = collections.Counter(neighborhood)
    if val in col.keys():
        val_am = col[val]
    else:
        return True
    valu = list(col.values())
    sorted = np.sort(valu)
    if len(sorted) >= 2 and sorted[1] >= val_am:
        return True
    return False


class Space:
    def __init__(self, sizex=1, sizey=1, sizez=1, neighborhood_type="M", cell_shape="r", boundary_type="a", optymalization_type="CA"):
        self.boundary_type = boundary_type
        self.neighborhood_type = neighborhood_type
        self.dimention = (sizex, sizey, sizez)
        self.cell_shape = cell_shape
        self.n_empty = sizex * sizey * sizez
        self.space = np.zeros(self.dimention, int)
        self.optymalization_type = optymalization_type
        self.n = 0
        self.acboundary = []
        self.seed_boundary = dict


    def seed_random_seeds(self, n):
        self.n = n
        if self.dimention[0]*self.dimention[1]*self.dimention[2] < n:
            return "Too many seeds"
        self.n_empty -= n
        val = random.sample(range(self.dimention[0] * self.dimention[1] * self.dimention[2]), n)
        val = np.sort(val).tolist()

        #print(val)
        for i in range(n):
            x, y, z = self.num2xyzT(val[i])
            self.space[int(x), int(y), int(z)] = i + 1
            self.acboundary += [(x,y,z)]

    def num2xyzT(self, v):
        z = int(v) / (self.dimention[0] * self.dimention[1]) % self.dimention[2]
        y = int(v) / self.dimention[0] % (self.dimention[1])
        x = int(v) % self.dimention[0]
        return int(x), int(y), int(z)


    def num2xyz(self, v):
        x, y, z = v
        return int(x), int(y), int(z)
        z = int(v) / (self.dimention[0] * self.dimention[1]) % self.dimention[2]
        y = int(v) / self.dimention[0] % (self.dimention[1])
        x = int(v) % self.dimention[0]
        return int(x), int(y), int(z)


    def xyz2num(self, x, y, z):
        return (x,y,z)
        return x + y*self.dimention[0] + z*self.dimention[0]*self.dimention[1]

    def load_space(self, space, n):
        self.space = np.reshape(space, self.dimention)
        self.n_empty -= n

    def show_space(self):
        return self.space[:,:,:]

    def nb_empty(self):
        return self.n_empty

    def rmca(self):
        res = self.space.copy()
        while len(self.acboundary) > 0:
            (x, y, z) = self.acboundary.pop()
            valc, neighborhood, _, boundary = self.neighborhood_3d_M(int(x), int(y), int(z), self.space)
            v = self.add_to_grain(valc, neighborhood)
            res[x, y, z] = v
        self.acboundary = sorted(set(boundary))
        self.space = res

    def cat(self, ac):
        r = []
        for i in range(len(ac)):
            (x, y, z) = (ac[i])
            valc, neighborhood, _, boundary = self.neighborhood_3d_M(int(x), int(y), int(z), self.space)
            v = self.add_to_grain(valc, neighborhood)
            r += [(x, y, z, v, boundary)]
        return r



    def rmcat(self, nt):
        res = self.space.copy()
        r = []
        boundary = []
        ac = np.copy(self.acboundary)
        self.n_empty -= 1*len(ac)

        with multiprocessing.Pool(nt) as p:
            r.append(p.map(self.cat, (ac,)))
        for k in r:
            for j in k:
                for i in j:
                    x,y,z,v,b = i
                    boundary += b
                    res[x, y, z] = v
        self.acboundary = sorted(set(boundary))
        self.space = res


    def rmmc(self):
        l = len(self.acboundary)
        val = random.sample(self.acboundary, l)
        # print(val)
        for i in val:
            x, y, z = self.num2xyz(i)
            valc, neighborhood, cordc = self.neighborhood_3d_M(int(x), int(y), int(z), self.space)
            res = monte_test(valc, neighborhood)
            if monte_check(res, neighborhood):
                self.boundary.put(i)
            if valc != res:
                self.space[x, y, z] = res
                j = [cordc[x] for x in range(len(neighborhood)) if res != neighborhood[x]]
                if j != []:
                    self.boundary.put(j)
        self.acboundary = list(set(dump_queue(self.boundary)))

    def grain_grow(self):
        if self.cell_shape == "r":
            if self.neighborhood_type == "M":
                if self.optymalization_type == "CA":
                    #self.rmca()
                    self.rmcat(1)
                if self.optymalization_type == "MC":
                    self.rmmc()


            if self.neighborhood_type == "N":
                three = 3


    def neighborhood_3d_M(self, x, y, z, space3d):
        valc, neighborhood, cordc, boundary = self.neighborhood_2d_M(x, y, z, space3d[:, :, z])
        behind_val = None
        under_val = None
        if z == 0:
            if self.boundary_type == "p":
                behind_val, behind_nieghborhood, behind_cord, behind_boundary = self.neighborhood_2d_M(x, y, self.dimention[2]-1, space3d[:, :, self.dimention[2]-1])
                cord = self.dimention[2]-1
                neighborhood += behind_nieghborhood
                boundary += behind_boundary
        else:
            behind_val, behind_nieghborhood, behind_cord, behind_boundary = self.neighborhood_2d_M(x, y, z-1, space3d[:, :, z-1])
            neighborhood += behind_nieghborhood
            boundary += behind_boundary
            cord = z-1
        if behind_val is not None:
            cordc += behind_cord
            if behind_val == 0:
                boundary += [(x, y, cord)]
            else:
                neighborhood += [behind_val]
                cordc += [ (x, y, cord)]

        if z == self.dimention[2] - 1:
            if self.boundary_type == "p":
                under_val, under_nieghborhood, under_cord, under_boundary = self.neighborhood_2d_M(x, y, 0, space3d[:, :, 0])
                cord = 0
                boundary += under_boundary
        else:
            under_val, under_nieghborhood, under_cord, under_boundary = self.neighborhood_2d_M(x, y, z+1, space3d[:, :, z+1])
            cord = z+1
            boundary += under_boundary

        if under_val is not None:
            cordc += under_cord
            if under_val == 0:
                boundary += [(x, y, cord)]
            else:
                neighborhood += [under_val]
                cordc += [ (x, y, cord)]

        return valc, neighborhood, cordc, boundary



    def neighborhood_2d_M(self, x, y, z, matrix):
        valc, neighborhood, cordc, boundary = self.neighborhood_1d(x, y, z, matrix[:, y])
        behind_val = None
        under_val = None
        if y == 0:
            if self.boundary_type == "p":
                behind_val, behind_nieghborhood, behind_cord, behind_boundary = self.neighborhood_1d(x, self.dimention[1]-1, z, matrix[:, self.dimention[1]-1])
                cord = self.dimention[1]-1
                neighborhood += behind_nieghborhood
                boundary += behind_boundary
        else:
            behind_val, behind_nieghborhood, behind_cord, behind_boundary = self.neighborhood_1d(x, y-1, z, matrix[:, y-1])
            neighborhood += behind_nieghborhood
            boundary += behind_boundary
            cord = y-1
        if behind_val is not None:
            cordc += behind_cord
            if behind_val == 0:
                boundary += [(x, cord, z)]
            else:
                neighborhood += [behind_val]
                cordc += [(x, cord, z)]

        if y == self.dimention[1] - 1:
            if self.boundary_type == "p":
                under_val, under_nieghborhood, under_cord, under_boundary = self.neighborhood_1d(x, 0, z, matrix[:, 0])
                neighborhood += under_nieghborhood
                boundary += under_boundary
                cord = 0
        else:
            under_val, under_nieghborhood, under_cord, under_boundary = self.neighborhood_1d(x, y+1, z, matrix[:, y+1])
            neighborhood += under_nieghborhood
            boundary += under_boundary

            cord = y+1

        if under_val is not None:
            cordc += under_cord
            if under_val == 0:
                boundary += [(x, cord, z)]
            else:
                neighborhood += [under_val]
                cordc += [(x, cord, z)]
        return valc, neighborhood, cordc, boundary


    def neighborhood_1d(self, x, y, z, line):
        neighborhood = []
        cordc = []
        boundary = []
        cord = None
        valc = line[x]
        if x == 0:
            if self.boundary_type == "p":
                cord = self.dimention[0]-1
        else:
            cord = x - 1
        if cord is not None:
            val = line[cord]
            if val == 0:
                boundary += [(cord, y, z)]
            else:
                neighborhood += [val]
                cordc += [ (cord, y, z)]

        cord = None
        if x == self.dimention[0] - 1:
            if self.boundary_type == "p":
                cord = 0
        else:
            cord = x + 1
        if cord is not None:
            val = line[cord]
            if val == 0:
                boundary += [(cord, y, z)]
            else:
                neighborhood += [val]
                cordc += [(cord, y, z)]

        return valc, neighborhood, cordc, boundary


    def add_to_grain(self, val, neighborhood):
        if val != 0:
            return val
        if neighborhood == []:
            return 0
        return most_frequent(neighborhood)

    def find_boundary(self):
        seed_boundary = [[] for i in range(int(self.space.max())+1)]
        boundary = []
        for x in range(self.dimention[0]):
            for y in range(self.dimention[1]):
                for z in range(self.dimention[2]):
                    v, n, _, boundary = self.neighborhood_3d_M(x, y, z, self.space)
                    n = [i for i in n if i != v]
                    if n != []:
                        i = [x, y, z]
                        seed_boundary[int(v)] += [i]
                        boundary+=[i]

        self.acboundary = boundary


