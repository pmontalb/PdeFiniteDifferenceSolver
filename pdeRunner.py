import numpy as np
from subprocess import Popen
import os
from plotter import *

CWD = os.getcwd()
debugDll = "{}\\x64\\Debug\\PdeFiniteDifferenceSolver.exe".format(CWD)
releaseDll = "{}\\x64\\Release\\PdeFiniteDifferenceSolver.exe".format(CWD)

GRID_FILE = "{}\\grid.npy".format(CWD)
INITIAL_CONDITION_FILE = "{}\\ic.npy".format(CWD)


def run_transport1D(run=True, show=True, save=False):
    output_file = "transport1.cl"

    try:
        os.remove(GRID_FILE)
        os.remove(INITIAL_CONDITION_FILE)
    except FileNotFoundError:
        pass

    # grid = np.linspace(-np.pi, np.pi, 128)
    # ic = np.sin(grid)
    grid = np.linspace(-np.pi, np.pi, 128)
    ic = np.exp(-grid ** 2)
    if run:
        np.savetxt(GRID_FILE, grid)
        np.savetxt(INITIAL_CONDITION_FILE, ic)

        p = Popen([debugDll] +
                  ["-ic", INITIAL_CONDITION_FILE] +
                  ["-g", GRID_FILE] +
                  ["-of", output_file] +
                  ["-md", "Double"] +
                  ["-lbct", "Periodic"] +
                  ["-lbc", "0.0"] +
                  ["-rbct", "Periodic"] +
                  ["-st", "RungeKutta4"] +
                  ["-sdt", "Upwind"] +
                  ["-d", "0"] +
                  ["-v", ".5"] +
                  ["-dt", "0.001"] +
                  ["-n", "1"] +
                  ["-N", "100"])
        p.communicate()

    solution = np.loadtxt(output_file)
    animate(solution, grid, show=show, save=save)


def run_diffusion1D(run=True, show=True, save=False):
    output_file = "diffusion.cl"

    try:
        os.remove(GRID_FILE)
        os.remove(INITIAL_CONDITION_FILE)
    except FileNotFoundError:
        pass

    grid = np.linspace(-np.pi, np.pi, 128)
    ic = np.exp(-.5 * grid * grid)
    if run:
        np.savetxt(GRID_FILE, grid)
        np.savetxt(INITIAL_CONDITION_FILE, ic)

        p = Popen([releaseDll] +
                  ["-ic", INITIAL_CONDITION_FILE] +
                  ["-g", GRID_FILE] +
                  ["-of", output_file] +
                  ["-md", "Double"] +
                  ["-lbct", "Neumann"] +
                  ["-lbc", "0.0"] +
                  ["-rbct", "Neumann"] +
                  ["-st", "CrankNicolson"] +
                  ["-d", "1"] +
                  ["-v", "0"] +
                  ["-dt", "0.005"] +
                  ["-n", "10"] +
                  ["-N", "100"])
        p.communicate()

    solution = np.loadtxt(output_file)
    animate(solution, grid, show=show, save=save)


if __name__ == "__main__":
    run_transport1D(run=False, show=True, save=False)
