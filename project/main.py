from numpy.core.fromnumeric import cumprod
from numpy.core.numeric import _count_nonzero_dispatcher
from molecule import Molecule
from hessian import Hessian
from hessian import lj_potential
from hessian import pbc_sep
from frequencies import Frequencies
import autograd.numpy as np
from autograd import grad
import time

def get_config():
    current_set = []
    fail = 0
    f = open("init_config.txt", "w")
    while len(current_set) < 108:
        print(len(current_set))
        new_mol = np.random.rand(3,) * 18

        flag = 1
        for current_mol in current_set:
            if np.linalg.norm(pbc_sep(new_mol, current_mol)) <= 3.4:
                flag = 0

        if flag:
            current_set.append(new_mol)
            fail = 0
        else:
            fail+=1
            if fail >= 10000:
                fail = 0
                temp = []
                temp.append(new_mol)
                for mol in current_set:
                    if np.linalg.norm(pbc_sep(mol, new_mol)) >= 3.4:
                        temp.append(mol)
                current_set = temp
                np.random.seed(int(time.time()))

    for point in current_set:
        print(point[0], point[1], point[2], file=f)
    f.close()


def gradient_descent(alpha, max_its, w):
    gradient = grad(lj_potential)
    k = 0
    while k < 100:
        grad_eval = gradient(w)
        w = w - alpha*grad_eval
        c = lj_potential(w)
        print("It:", k)
        print(c)
        k+=1

    return w, lj_potential(w)



if __name__ == '__main__':
    ### CAUTION DONOT RUN
    get_config()
    ##
    init_mol = None
    f =  open("init_config.txt", "r")
    init_mol = Molecule(f, "Angstrom")
    f.close()

    e = lj_potential(np.array(init_mol.geom))
    print("LJ Potential : " , e)
    w, c = gradient_descent(0.1, 100, init_mol.geom)
    print("Original Energy: ", e)
    print("New Energy:", c)

    f2 = open("optimum_config.txt", "w")
    for p in w:
        print(
            f"{p[0]} {p[1]} {p[2]}", file=f2)
    f2.close()

    f2 = open("optimum_config.txt", "r")
    mol = Molecule(f2, "Angstrom")
    mol.conv('Bohr')
    hessian = Hessian(mol, 0.00001)
    hessian.write_Hessian()
    f2.close()
    with open("optimum_config.txt", "r") as f2:
        mol = Molecule(f2)
        hessian = open("hessian.dat", "r").read()
        freq = Frequencies(mol, hessian)
        freq.frequency_output("modes.txt")
