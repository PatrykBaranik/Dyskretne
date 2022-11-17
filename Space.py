import numpy as np
from typing import Dict
from collections import Counter
import random


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

class Space:
    def __init__(self, sizex=1, sizey=1, sizez=1, neighborhood_type="M", cell_shape="r", boundary="a", optymalization_type="CA"):
        self.boundary = boundary
        self.neighborhood_type = neighborhood_type
        self.dimention = (sizex, sizey, sizez)
        self.cell_shape = cell_shape
        self.n_empty = sizex * sizey * sizez
        self.space = np.zeros(self.dimention, int)
        self.optymalization_type = optymalization_type
        self.n = 0

    def seed_random_seeds(self, n):
        self.n = n
        if self.dimention[0]*self.dimention[1]*self.dimention[2] < n:
            return "Too many seeds"
        self.n_empty -= n
        val = random.sample(range(self.dimention[0] * self.dimention[1] * self.dimention[2]), n)
        #print(val)
        for i in range(n):
            z = int(val[i]) % self.dimention[2]
            y = int(val[i])/self.dimention[2] % self.dimention[1]
            x = int(val[i])/self.dimention[2]/self.dimention[1] % self.dimention[0]
            self.space[int(x), int(y), int(z)] = i + 1

    def load_space(self, space, n):
        self.space = space
        self.n_empty -= n

    def show_space(self):
        return self.space

    def nb_empty(self):
        return self.n_empty

    def grain_grow(self):
        if self.cell_shape == "r":
            if self.neighborhood_type == "M":
                if self.optymalization_type == "CA":
                    res = np.empty(shape=(self.dimention[0], self.dimention[1], self.dimention[2]))
                    for x in range(self.dimention[0]):
                        for y in range(self.dimention[1]):
                            for z in range(self.dimention[2]):
                                res[x, y, z] = self.add_to_grain(self.neighborhood_3d_M(x, y, z, self.space))
                    self.space = res
                if self.optymalization_type == "MC":
                    val = random.sample(range(self.dimention[0] * self.dimention[1] * self.dimention[2]), self.dimention[0] * self.dimention[1] * self.dimention[2])
                # print(val)
                    for i in range(self.dimention[0] * self.dimention[1] * self.dimention[2]):
                        z = int(val[i]) % self.dimention[2]
                        y = int(val[i]) / self.dimention[2] % self.dimention[1]
                        x = int(val[i]) / self.dimention[2] / self.dimention[1] % self.dimention[0]
                        self.space[int(x), int(y), int(z)] = self.monte_test(self.neighborhood_3d_M(int(x), int(y), int(z), self.space))
            if self.neighborhood_type == "N":
                if self.optymalization_type == "CA":
                    res = np.empty(shape=(self.dimention[0], self.dimention[1], self.dimention[2]))
                    for x in range(self.dimention[0]):
                        for y in range(self.dimention[1]):
                            for z in range(self.dimention[2]):
                                res[x, y, z] = self.add_to_grain(self.neighborhood_3d_M(x, y, z, self.space))
                    self.space = res
                if self.optymalization_type == "MC":
                    val = random.sample(range(self.dimention[0] * self.dimention[1] * self.dimention[2]), self.dimention[0] * self.dimention[1] * self.dimention[2])
                # print(val)
                    for i in range(self.dimention[0] * self.dimention[1] * self.dimention[2]):
                        z = int(val[i]) % self.dimention[2]
                        y = int(val[i]) / self.dimention[2] % self.dimention[1]
                        x = int(val[i]) / self.dimention[2] / self.dimention[1] % self.dimention[0]
                        self.space[int(x), int(y), int(z)] = self.monte_test(self.neighborhood_3d_M(int(x), int(y), int(z), self.space))



    def neighborhood_3d_M(self, x, y, z, space3d):
        val, neighborhood = self.neighborhood_2d_M(x, y, space3d[:, :, z])
        if z == 0:
            if self.boundary == "a":
                behind_val, behind_nieghborhood = 0, []
            if self.boundary == "p":
                behind_val, behind_nieghborhood = self.neighborhood_2d_M(x, y, space3d[:, :, self.dimention[2]-1])

        else:
            behind_val, behind_nieghborhood = self.neighborhood_2d_M(x, y, space3d[:, :, z - 1])
        if z == self.dimention[2]-1:
            if self.boundary == "a":
                under_val, under_nieghborhood = 0, []
            if self.boundary == "p":
                under_val, under_nieghborhood = self.neighborhood_2d_M(x, y, space3d[:, :, 0])
        else:
                under_val, under_nieghborhood = self.neighborhood_2d_M(x, y, space3d[:, :, z + 1])

        neighborhood += neighborhood + [under_val] + [behind_val] + behind_nieghborhood + under_nieghborhood
        return val, neighborhood


    def neighborhood_3d_N(self, x, y, z, space3d):
        val, neighborhood = self.neighborhood_2d_N(x, y, space3d[:, :, z])
        if z == 0:
            if self.boundary == "a":
                behind_val, behind_nieghborhood = 0, []
            if self.boundary == "p":
                behind_val, behind_nieghborhood = space3d[x, y, self.dimention[2]-1]

        else:
            behind_val, behind_nieghborhood = space3d[x, y, z - 1]
        if z == self.dimention[2]-1:
            if self.boundary == "a":
                under_val, under_nieghborhood = 0, []
            if self.boundary == "p":
                under_val, under_nieghborhood = space3d[x, y, 0]
        else:
                under_val, under_nieghborhood = space3d[x, y, z + 1]

        neighborhood += neighborhood + [under_val] + [behind_val] + behind_nieghborhood + under_nieghborhood
        return val, neighborhood


    def neighborhood_2d_M(self, x, y, matrix):
        val, neighborhood = self.neighborhood_1d(x, matrix[:, y])

        if y == 0:
            if self.boundary == "a":
                behind_val, behind_nieghborhood = 0, []
            if self.boundary == "p":
                behind_val, behind_nieghborhood = self.neighborhood_1d(x, matrix[:, self.dimention[1]-1])
        else:
            behind_val, behind_nieghborhood = self.neighborhood_1d(x, matrix[:, y-1])
        if y == self.dimention[1] - 1:
            if self.boundary == "a":
                under_val, under_nieghborhood = 0, []
            if self.boundary == "p":
                under_val, under_nieghborhood = self.neighborhood_1d(x, matrix[:, 0])

        else:
            under_val, under_nieghborhood = self.neighborhood_1d(x, matrix[:, y+1])
        neighborhood += neighborhood + [under_val] + [behind_val] + behind_nieghborhood + under_nieghborhood
        return val, neighborhood


    def neighborhood_2d_N(self, x, y, matrix):
        val, neighborhood = self.neighborhood_1d(x, matrix[:, y])

        if y == 0:
            if self.boundary == "a":
                behind_val, behind_nieghborhood = 0, []
            if self.boundary == "p":
                behind_val, behind_nieghborhood = matrix[x, self.dimention[1]-1]
        else:
            behind_val, behind_nieghborhood = matrix[x, y-1]
        if y == self.dimention[1] - 1:
            if self.boundary == "a":
                under_val, under_nieghborhood = 0, []
            if self.boundary == "p":
                under_val, under_nieghborhood = matrix[x, 0]

        else:
            under_val, under_nieghborhood = matrix[x, y+1]
        neighborhood += neighborhood + [under_val] + [behind_val] + behind_nieghborhood + under_nieghborhood
        return val, neighborhood


    def neighborhood_1d(self, x, line):
        neighborhood = []
        val = line[x]
        if x == 0:
            if self.boundary == "a":
                neighborhood += []
            if self.boundary == "p":
                neighborhood += [line[self.dimention[0]-1]]
        else:
            neighborhood += [line[x - 1]]

        if x == self.dimention[0] - 1:
            if self.boundary == "a":
                neighborhood += []
            if self.boundary == "p":
                neighborhood += [line[0]]
        else:
            neighborhood += [line[x + 1]]
        return val, neighborhood


    def add_to_grain(self, val_neighborhood):
        val, neighborhood = val_neighborhood
        if val != 0:
            return val
        neighborhood = [i for i in neighborhood if i != 0]
        if neighborhood == []:
            return 0
        self.n_empty -= 1
        return most_frequent(neighborhood)

    def monte_test(self, val_neighborhood):
        val, neighborhood = val_neighborhood
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







