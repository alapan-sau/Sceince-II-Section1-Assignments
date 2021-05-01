from molecule import Molecule
import sys
import autograd.numpy as np
from matplotlib import pyplot as plt
hartree2J = 4.359744e-18
amu2kg = 1.660538782e-27
bohr2m = 0.52917720859e-10
c = 2.99792458E8


class Frequencies:

    def __init__(self, mol, hessString):

        self.hess = hessString
        self.mol = mol
        self.N = len(mol)

        m = []
        for i in range(self.N):
            m += [1/(mol.masses[i])**0.5]*3

        self.MM = np.diag(m)
        self.m = m

    def get_MWhessian(self):
        temp = []
        for i in self.hess.splitlines():
            temp.append(i.split())

        H0 = np.matrix(temp, float)
        A = np.dot(H0, self.MM)
        mwH = np.dot(A, self.MM)

        return mwH

    def get_frequencies(self):

        self.e, self.l = np.linalg.eigh(self.get_MWhessian())

        self.Q = np.matrix(self.MM) * np.matrix(self.l)
        freq = []

        A =  np.sqrt(hartree2J)
        B = A/(amu2kg*bohr2m**2)
        C = (c*2*np.pi)

        conv = B / C

        for i in self.e:
            if i >= 0:
                freq.append(i**0.5*conv)
            elif i < -1:
                freq.append((-i)**0.5*conv)
            else:
                freq.append((-i)**0.5*conv)
        return freq


    def frequency_output(self, output):

        mol = self.mol
        freq = self.get_frequencies()

        t = open(output, "w")
        for i in range(3*self.N):
            t.write("%d\n%s cm^{-1}\n" % (self.N, str(freq[i])))
            for j in range(self.N):
                threejs = 3 * j
                y = mol.geom[j, 1]
                x = mol.geom[j, 0]
                z = mol.geom[j, 2]

                dy = self.Q[threejs+1, i]
                dx = self.Q[threejs, i]
                dz = self.Q[threejs+2, i]

                t.write("{:12.7f}{:12.7f}{:12.7f}\n".format(x, y, z))
            t.write("\n")
        t.close()
        a = np.array(freq)
        plt.hist(a, bins=100)
        plt.title("Frequency Histogram")
        plt.savefig("hist.png")

        return None
