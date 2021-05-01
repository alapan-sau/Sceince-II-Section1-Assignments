from molecule import Molecule
import autograd.numpy as np
import os
import re
from multiprocessing import Pool, Manager
import functools

epsilon = 0.238
sigma = 3.4


def pbc_sep(p1, p2):
    direction = p1-p2
    # rem = np.mod(direction, 18)
    rem = np.array([direction[i]%18 for i in range(len(direction))])
    # mic_separation_vector = np.mod(rem+9, 18)-9
    mic_separation_vector = [(rem[i]+ 9)%18 -9 for i in range(len(rem)) ]
    return np.array(mic_separation_vector)


def lj_potential(geom):
    te = 0
    tgeom = np.array(geom)
    pairs = []
    for i in range(len(tgeom)):
        for j in range(i+1, len(tgeom)):
            pairs.append((tgeom[i], tgeom[j]))
    for pair in pairs:
        r = np.linalg.norm(pbc_sep(pair[0], pair[1]))
        if r == 0:
            continue
        te += ((sigma/r)**12-(sigma/r)**6)
    return 4 * epsilon * te


class Hessian(object):
    def __init__(self, mol, disp_size=0.005):
        self.N = len(mol)
        self.mol = mol
        self.energy = dict()
        self.h = disp_size


    def find_E(self, i, j, hi, hj):
        return self.energy["X%dX%d_%d%d" % (i, j, hi, hj)]


    def set_energy(self, key, geom, d=None):
        e = lj_potential(np.array(geom))
        if d is None:
            self.energy[key] = e
        else:
            d[key] = e


    def process(self, i, d):
        h, N, geom = self.h, self.N, self.mol.geom
        print(i)
        for j in range(i):
            forward = "X"+str(i)+"X"+str(j)+"_11"
            reverse = "X"+str(i)+"X"+str(j)+"_-1-1"

            geom_copy2 = self.mol.copygeom()

            alpha = i//3
            beta = i%3
            gamma = j//3
            delta = j%3

            geom_copy2[alpha, beta] = geom_copy2[alpha, beta] + h
            geom_copy2[gamma, delta] = geom_copy2[gamma, delta] + h

            self.set_energy(forward, geom_copy2, d)

            geom_copy2[alpha, beta] = geom_copy2[alpha, beta] - 2*h
            geom_copy2[gamma, delta] = geom_copy2[gamma, delta] - 2*h

            self.set_energy(reverse, geom_copy2, d)

    def run_disps(self):
        h = self.h
        N = self.N
        xoxo = "X0X0_00"
        geom = self.mol.geom
        self.set_energy(xoxo, geom)

        ####   Run single displacements   ####
        for i in range(3*N):
            print(i)
            forward = "X%dX0_10" % i
            reverse = "X%dX0_-10" % i
            geom_copy = self.mol.copygeom()

            alpha = i//3
            beta = i%3

            geom_copy[alpha, beta] = geom_copy[alpha, beta]+h
            self.set_energy(forward, geom_copy)

            geom_copy[alpha, beta] = geom_copy[alpha, beta]-2*h
            self.set_energy(reverse, geom_copy)
        ####   Run double displacements    ######
        mylist = [*range(3*N)]
        pool = Pool()
        D = Manager().dict()                     # Create a multiprocessing Pool
        pool.map(functools.partial(self.process, d=D), mylist)
        pool.close()
        pool.join()
        self.energy.update(D)

    def make_Hessian(self):

        self.run_disps()

        h= self.h
        E0 = self.find_E(0, 0, 0, 0)
        N = self.N

        self.H = np.ones((3*self.N, 3*self.N)) * 0

        for i in range(3*N):
            print(i)
            for i in range(3*N):
                self.H[i, i] = (self.find_E(i, 0, 1, 0) +
                                self.find_E(i, 0, -1, 0)-2*E0)/(h**2)
                for j in range(0, i):
                    self.H[i, j] = (self.find_E(i, j, 1, 1)+self.find_E(i, j, -1, -1)-self.find_E(
                        i, 0, 1, 0)-self.find_E(j, 0, 1, 0)-self.find_E(j, 0, -1, 0)-self.find_E(i, 0, -1, 0)+2*E0)
                    self.H[i, j] /= 2*h**2
                    self.H[j, i] = self.H[i, j]

    def make_eigh(self):
        w, v = np.linalg.eigh(self.H)
        np.savetxt("eigen_vectors.dat", v, "%15.7f", " ", "\n")
        np.savetxt("eigen_values.dat", w, "%15.7f", " ", "\n")

    def write_Hessian(self):
        self.make_Hessian()
        self.make_eigh()
        np.savetxt("hessian.dat", self.H, "%15.7f", " ", "\n")
