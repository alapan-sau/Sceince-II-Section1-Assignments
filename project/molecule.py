import autograd.numpy as np
import sys

class Molecule:
    def __init__(self, f, units="Angstrom"):
        self.units = units
        self.read(f)
        self.masses = float(39.96238312251) * np.ones(len(self.geom))

    def read(self, f):
        geom_str = f.readlines()
        geom = []
        for line in geom_str:
            x, y, z = line.split()
            geom.append([float(x), float(y), float(z)])
        self.geom = np.array(geom)

    def __len__(self):
        count = 0
        for i in self.geom:
            count+=1
        return count

    def conv(self, type_text):
        if(type_text == 'Bohr' and self.units == "Angstrom" ):
            self.geom *= 1.889725989
            self.units = "Bohr"
            return self.geom

        elif(type_text == 'Angstrom' and self.units == "Bohr"):
            self.geom /= 1.889725989
            self.units = "Angstrom"
            return self.geom
        else:
            return self.geom


    def copygeom(self):
        return np.array(self.geom)
