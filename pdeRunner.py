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
    output_file = "transport.cl"

    try:
        os.remove(GRID_FILE)
        os.remove(INITIAL_CONDITION_FILE)
    except FileNotFoundError:
        pass

    grid = np.linspace(-np.pi, np.pi, 128)
    ic = np.sin(grid)
    if run:
        np.savetxt(GRID_FILE, grid)
        np.savetxt(INITIAL_CONDITION_FILE, ic)

        p = Popen([releaseDll] +
                  ["-ic", INITIAL_CONDITION_FILE] +
                  ["-g", GRID_FILE] +
                  ["-of", output_file] +
                  ["-md", "Single"] +
                  ["-lbct", "Periodic"] +
                  ["-lbc", "0.0"] +
                  ["-rbct", "Periodic"] +
                  ["-st", "ExplicitEuler"] +
                  ["-sdt", "LaxWendroff"] +
                  ["-d", "0"] +
                  ["-v", ".5"] +
                  ["-dt", "0.05"] +
                  ["-n", "5"] +
                  ["-N", "1000"])
        p.communicate()

    solution = np.loadtxt(output_file)
    animate(solution, grid, show=show, save=save, name="lw.gif")


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
    animate(solution, grid, show=show, save=save, name="diffusion")


def run_wave1D(run=True, show=True, save=False):
    output_file = "wave1.cl"

    try:
        os.remove(GRID_FILE)
        os.remove(INITIAL_CONDITION_FILE)
    except FileNotFoundError:
        pass

    grid = np.linspace(-np.pi, np.pi, 128)
    ic = np.sin(grid)
    if run:
        np.savetxt(GRID_FILE, grid)
        np.savetxt(INITIAL_CONDITION_FILE, ic)

        p = Popen([debugDll] +
                  ["-ic", INITIAL_CONDITION_FILE] +
                  ["-g", GRID_FILE] +
                  ["-of", output_file] +
                  ["-md", "Single"] +
                  ["-lbct", "Neumann"] +
                  ["-lbc", "0.0"] +
                  ["-rbct", "Neumann"] +
                  ["-st", "ExplicitEuler"] +
                  ["-sdt", "Central"] +
                  ["-pde", "WaveEquation"] +
                  ["-d", "0"] +
                  ["-v", ".5"] +
                  ["-dt", "0.05"] +
                  ["-n", "5"] +
                  ["-N", "1000"])
        p.communicate()

    solution = np.loadtxt(output_file)
    animate(solution, grid, show=show, save=save, name="lw.gif")


if __name__ == "__main__":
    run_wave1D(run=False, show=True, save=False)
