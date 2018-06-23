import numpy as np
from subprocess import Popen
import os
from plotter import *

CWD = os.getcwd()
debugDll = "{}\\x64\\Debug\\PdeFiniteDifferenceSolver.exe".format(CWD)
releaseDll = "{}\\x64\\Release\\PdeFiniteDifferenceSolver.exe".format(CWD)

GRID_FILE = "{}\\grid.npy".format(CWD)
INITIAL_CONDITION_FILE = "{}\\ic.npy".format(CWD)


def run_transport_1D(space_discretizer="LaxWendroff",
                     output_file="transport.cl",
                     name="transport.gif",
                     run=True, show=True, show_grid=False, save=False,
                     run_animation=True):

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
                  ["-md", "Float"] +
                  ["-lbct", "Periodic"] +
                  ["-lbc", "0.0"] +
                  ["-rbct", "Periodic"] +
                  ["-st", "ExplicitEuler"] +
                  ["-sdt", space_discretizer] +
                  ["-d", "0"] +
                  ["-v", ".5"] +
                  ["-dt", "0.005"] +
                  ["-n", "25"] +
                  ["-N", "500"])
        p.communicate()

    solution = np.loadtxt(output_file)
    if run_animation:
        animate(solution, grid, show=show, save=save, grid=show_grid, name=name)

    return grid, solution


def run_diffusion_1D(solver_type="CrankNicolson", output_file="diffusion.cl", name="diffusion.gif",
                     run=True, show=True, show_grid=False, save=False,
                     run_animation=True):

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
                  ["-st", solver_type] +
                  ["-d", "1"] +
                  ["-v", "0"] +
                  ["-dt", "0.0003"] +
                  ["-n", "20"] +
                  ["-N", "200"])
        p.communicate()

    solution = np.loadtxt(output_file)
    if run_animation:
        animate(solution, grid, show=show, save=save, grid=show_grid, name=name)

    return grid, solution


def run_wave_1D(solver_type="CrankNicolson",
                output_file="wave.cl", name="wave.gif", run=True, show=True, show_grid=False, save=False,
                run_animation=True):

    try:
        os.remove(GRID_FILE)
        os.remove(INITIAL_CONDITION_FILE)
    except FileNotFoundError:
        pass

    grid = np.linspace(-np.pi, np.pi, 128)
    ic = np.exp(-grid * grid)
    if run:
        np.savetxt(GRID_FILE, grid)
        np.savetxt(INITIAL_CONDITION_FILE, ic)

        p = Popen([releaseDll] +
                  ["-ic", INITIAL_CONDITION_FILE] +
                  ["-g", GRID_FILE] +
                  ["-of", output_file] +
                  ["-md", "Double"] +
                  ["-lbct", "Dirichlet"] +
                  ["-lbc", "0.0"] +
                  ["-rbct", "Dirichlet"] +
                  ["-st", solver_type] +
                  ["-pde", "WaveEquation"] +
                  ["-d", "0"] +
                  ["-v", ".05"] +
                  ["-dt", "0.0015"] +
                  ["-n", "100"] +
                  ["-N", "50"])
        p.communicate()

    solution = np.loadtxt(output_file)

    if run_animation:
        animate(solution, grid, show=show, grid=show_grid, save=save, name=name)

    return grid, solution


def __compare_solver_worker(worker, solver_list,
                            run=True, show=True, show_grid=False, save=False, name="comparison.gif",
                            y_lim=None):
    out = []
    for solver in solver_list:
        out.append(worker(solver, "{}.cl".format(solver), run=run, show=False,
                          show_grid=show_grid, save=False, run_animation=False))

    animate_multicurve([x[1] for x in out], [x[0] for x in out], grid=show_grid, show=show,
                       labels=solver_list, save=save, name=name, y_lim=y_lim)


def compare_solvers_transport_1D(solver_list, run=True, show=True, show_grid=False, save=False, name="comparison.gif"):
    __compare_solver_worker(run_transport_1D, solver_list,
                            run=run, show=show, show_grid=show_grid, save=save, name=name)


def compare_solvers_diffusion_1D(solver_list, run=True, show=True, show_grid=False, save=False, name="comparison.gif"):
    __compare_solver_worker(run_diffusion_1D, solver_list,
                            run=run, show=show, show_grid=show_grid, save=save, name=name)


def compare_solvers_wave_1D(solver_list, run=True, show=True, show_grid=False, save=False, name="comparison.gif"):
    __compare_solver_worker(run_wave_1D, solver_list,
                            run=run, show=show, show_grid=show_grid, save=save, name=name, y_lim=(-1.1, 1.1))


if __name__ == "__main__":

    # compare_solvers_transport_1D(["LaxWendroff", "Upwind"], run=False, show=True, show_grid=True,
    #                              save=True, name="numericalDiffusion.gif")

    # compare_solvers_diffusion_1D(["AdamsBashforth2", "AdamsMouldon2", "CrankNicolson"],
    #                              run=False, show=True, show_grid=True,
    #                              save=True, name="multiStep.gif")

    compare_solvers_wave_1D(["ExplicitEuler", "ImplicitEuler"],
                            run=True, show=True, show_grid=True,
                            save=True, name="waveInstability1D.gif")