import numpy as np
from typing import Dict
import collections
import random
from multiprocessing import Process

def most_frequent(lista):
    occurence_count = collections.Counter(lista)
    return occurence_count.most_common(1)[0][0]


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
        self.boundary = []
        self.acboundary = []
        self.seed_boundary = dict

    def update(self, changes):
        for i in changes:
            x, y, z, v = i
            self.space[x, y, z] = v
        self.n_empty -= len(changes)

    def get_acboundary(self):
        return self.acboundary.copy()


    def set_acboundary(self, acb):
        self.acboundary = acb.copy()

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
        return np.copy(self.space[:,:,:])

    def nb_empty(self):
        return self.n_empty

    def set_nb_empty(self, n):
        self.n_empty = n

    def get_boundary(self):
        return self.boundary.copy()

    def rmca(self):
        res = self.space.copy()
        changes = []
        n_e = self.n_empty
        for i in self.acboundary:
            (x, y, z) = i
            valc, neighborhood, _ = self.neighborhood_3d_M(int(x), int(y), int(z), self.space)
            v = self.add_to_grain(valc, neighborhood)
            res[x, y, z] = v
            changes += [(x, y, z, v)]

        self.acboundary = sorted(set(self.boundary))
        self.boundary = []
        self.space = res
        n_e = n_e - self.n_empty
        return changes, self.acboundary, n_e

    def rmmc(self):
        l = len(self.acboundary)
        val = random.sample(self.acboundary, l)
        changes = []
        n_e = self.n_empty
        for i in val:
            x, y, z = self.num2xyz(i)
            valc, neighborhood, cordc = self.neighborhood_3d_M(int(x), int(y), int(z), self.space)
            res = monte_test(valc, neighborhood)
            if monte_check(res, neighborhood):
                self.boundary += [i]
            if valc != res:
                self.space[x, y, z] = res
                changes +=[(x,y,z,res)]
                j = [cordc[x] for x in range(len(neighborhood)) if res != neighborhood[x]]
                if j != []:
                    self.boundary += j
        self.acboundary = list(set(self.boundary))
        self.boundary = []
        n_e = n_e - self.n_empty
        return changes, self.acboundary, n_e

    def grain_grow(self):
        if self.cell_shape == "r":
            if self.neighborhood_type == "M":
                if self.optymalization_type == "CA":
                    return self.rmca()
                if self.optymalization_type == "MC":
                    return self.rmmc()

            if self.neighborhood_type == "N":
                three = 3


    def neighborhood_3d_M(self, x, y, z, space3d):
        valc, neighborhood, cordc = self.neighborhood_2d_M(x, y, z, space3d[:, :, z])
        behind_val = None
        under_val = None
        if z == 0:
            if self.boundary_type == "p":
                behind_val, behind_nieghborhood, behind_cord = self.neighborhood_2d_M(x, y, self.dimention[2]-1, space3d[:, :, self.dimention[2]-1])
                cord = self.dimention[2]-1
                neighborhood += behind_nieghborhood
        else:
            behind_val, behind_nieghborhood, behind_cord = self.neighborhood_2d_M(x, y, z-1, space3d[:, :, z-1])
            neighborhood += behind_nieghborhood
            cord = z-1
        if behind_val is not None:
            cordc += behind_cord
            if behind_val == 0:
                self.boundary += [ (x, y, cord)]
            else:
                neighborhood += [behind_val]
                cordc += [ (x, y, cord)]

        if z == self.dimention[2] - 1:
            if self.boundary_type == "p":
                under_val, under_nieghborhood, under_cord = self.neighborhood_2d_M(x, y, 0, space3d[:, :, 0])
                cord = 0
        else:
            under_val, under_nieghborhood, under_cord = self.neighborhood_2d_M(x, y, z+1, space3d[:, :, z+1])
            cord = z+1

        if under_val is not None:
            cordc += under_cord
            if under_val == 0:
                self.boundary += [ (x, y, cord)]
            else:
                neighborhood += [under_val]
                cordc += [ (x, y, cord)]

        return valc, neighborhood, cordc



    def neighborhood_2d_M(self, x, y, z, matrix):
        valc, neighborhood, cordc = self.neighborhood_1d(x, y, z, matrix[:, y])
        behind_val = None
        under_val = None
        if y == 0:
            if self.boundary_type == "p":
                behind_val, behind_nieghborhood, bechind_cord = self.neighborhood_1d(x, self.dimention[1]-1, z, matrix[:, self.dimention[1]-1])
                cord = self.dimention[1]-1
                neighborhood += behind_nieghborhood
        else:
            behind_val, behind_nieghborhood, bechind_cord = self.neighborhood_1d(x, y-1, z, matrix[:, y-1])
            neighborhood += behind_nieghborhood
            cord = y-1
        if behind_val is not None:
            cordc += bechind_cord
            if behind_val == 0:
                self.boundary += [ (x, cord, z)]
            else:
                neighborhood += [behind_val]
                cordc += [ (x, cord, z)]

        if y == self.dimention[1] - 1:
            if self.boundary_type == "p":
                under_val, under_nieghborhood, under_cord = self.neighborhood_1d(x, 0, z, matrix[:, 0])
                neighborhood += under_nieghborhood
                cord = 0
        else:
            under_val, under_nieghborhood, under_cord = self.neighborhood_1d(x, y+1, z, matrix[:, y+1])
            neighborhood += under_nieghborhood

            cord = y+1

        if under_val is not None:
            cordc += under_cord
            if under_val == 0:
                self.boundary += [ (x, cord, z)]
            else:
                neighborhood += [under_val]
                cordc += [ (x, cord, z)]
        return valc, neighborhood, cordc


    def neighborhood_1d(self, x, y, z, line):
        neighborhood = []
        cordc = []
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
                self.boundary += [ (cord, y, z)]
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
                self.boundary += [(cord, y, z)]
            else:
                neighborhood += [val]
                cordc += [(cord, y, z)]

        return valc, neighborhood, cordc


    def add_to_grain(self, val, neighborhood):
        if val != 0:
            return val
        if neighborhood == []:
            return 0
        self.n_empty -= 1
        return most_frequent(neighborhood)

    def find_boundary(self):
        seed_boundary = [[] for i in range(int(self.space.max())+1)]
        self.boundary = []
        for x in range(self.dimention[0]):
            for y in range(self.dimention[1]):
                for z in range(self.dimention[2]):
                    v, n, _ = self.neighborhood_3d_M(x, y, z, self.space)
                    n = [i for i in n if i != v]
                    if n != []:
                        i = (x, y, z)
                        seed_boundary[int(v)] += [i]
                        self.boundary += [i]


