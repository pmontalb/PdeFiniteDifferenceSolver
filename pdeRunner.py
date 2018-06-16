import numpy as np
from subprocess import Popen
import os
from plotter import *

CWD = os.getcwd()
debugDll = "{}\\x64\\Debug\\PdeFiniteDifferenceSolver.exe".format(CWD)
releaseDll = "{}\\x64\\Release\\PdeFiniteDifferenceSolver.exe".format(CWD)

GRID_FILE = "{}\\grid.npy".format(CWD)
INITIAL_CONDITION_FILE = "{}\\ic.npy".format(CWD)


def run_transport1D(run=True, save=False):
    output_file = "sol.cl"

    try:
        os.remove(GRID_FILE)
        os.remove(INITIAL_CONDITION_FILE)
    except FileNotFoundError:
        pass

    if run:
        grid = np.linspace(-4.0, 4.0, 128)
        ic = np.exp(-.25 * grid * grid)

        np.savetxt(GRID_FILE, grid)
        np.savetxt(INITIAL_CONDITION_FILE, ic)

        p = Popen([releaseDll] +
                  ["-ic", INITIAL_CONDITION_FILE] +
                  ["-g", GRID_FILE] +
                  ["-of", output_file] +
                  ["-lbct", "Neumann"] +
                  ["-lbc", "0.0"] +
                  ["-rbct", "Neumann"] +
                  ["-st", "CrankNicolson"] +
                  ["-d", "0"] +
                  ["-v", "1"] +
                  ["-dt", "0.1"] +
                  ["-n", "1"] +
                  ["-N", "100"])
        p.communicate()

    solution = np.loadtxt(output_file)
    animate(solution, grid, show=False, save=save)


def run_diffusion1D(run=True, save=False):
    output_file = "sol.cl"

    try:
        os.remove(GRID_FILE)
        os.remove(INITIAL_CONDITION_FILE)
    except FileNotFoundError:
        pass

    if run:
        grid = np.linspace(-4.0, 4.0, 128)
        ic = np.exp(-.25 * grid * grid)

        np.savetxt(GRID_FILE, grid)
        np.savetxt(INITIAL_CONDITION_FILE, ic)

        p = Popen([releaseDll] +
                  ["-ic", INITIAL_CONDITION_FILE] +
                  ["-g", GRID_FILE] +
                  ["-of", output_file] +
                  ["-lbct", "Neumann"] +
                  ["-lbc", "0.0"] +
                  ["-rbct", "Neumann"] +
                  ["-st", "CrankNicolson"] +
                  ["-d", "1"] +
                  ["-v", "0"] +
                  ["-dt", "0.1"] +
                  ["-n", "1"] +
                  ["-N", "100"])
        p.communicate()

    solution = np.loadtxt(output_file)
    animate(solution, grid, show=False, save=save)


if __name__ == "__main__":
    run_transport1D(run=True, save=True)
