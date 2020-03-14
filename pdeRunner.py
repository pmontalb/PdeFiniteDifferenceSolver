import numpy as np
from subprocess import Popen
import os
from plotter import *

CWD = os.getcwd()
if os.name == 'nt':
    debugBin = "{}\\x64\\Debug\\PdeFiniteDifferenceSolver.exe".format(CWD)
    releaseBin = "{}\\x64\\Release\\PdeFiniteDifferenceSolver.exe".format(CWD)

    GRID_FILE = "{}\\grid.npy".format(CWD)
    INITIAL_CONDITION_FILE = "{}\\ic.npy".format(CWD)

    X_GRID_FILE = "{}\\x_grid.npy".format(CWD)
    Y_GRID_FILE = "{}\\y_grid.npy".format(CWD)
else:
    debugBin = "{}/cmake-build-gcc-debug/PdeFiniteDifferenceSolver".format(CWD)
    releaseBin = "{}/cmake-build-gcc-release/PdeFiniteDifferenceSolver".format(CWD)

    GRID_FILE = "{}/grid.npy".format(CWD)
    INITIAL_CONDITION_FILE = "{}/ic.npy".format(CWD)

    X_GRID_FILE = "{}/x_grid.npy".format(CWD)
    Y_GRID_FILE = "{}/y_grid.npy".format(CWD)

chosenBin = debugBin


def run_transport_1D(space_discretizer="Upwind",
                     solver_type='ExplicitEuler',
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
        np.save(GRID_FILE, grid)
        np.save(INITIAL_CONDITION_FILE, ic)

        p = Popen([chosenBin] +
                  ["-ic", INITIAL_CONDITION_FILE] +
                  ["-g", GRID_FILE] +
                  ["-of", output_file] +
                  ["-md", "Double"] +
                  ["-lbct", "Periodic"] +
                  ["-lbc", "0.0"] +
                  ["-rbct", "Periodic"] +
                  ["-rbc", "0.0"] +
                  ["-st", solver_type] +
                  ["-sdt", space_discretizer] +
                  ["-d", "0"] +
                  ["-v", ".05"] +
                  ["-dt", "0.1"] +
                  ["-n", "50"] +
                  ["-N", "500"])
        p.communicate()

    solution = np.load(output_file).transpose()
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
        np.save(GRID_FILE, grid)
        np.save(INITIAL_CONDITION_FILE, ic)

        p = Popen([chosenBin] +
                  ["-ic", INITIAL_CONDITION_FILE] +
                  ["-g", GRID_FILE] +
                  ["-of", output_file] +
                  ["-md", "Double"] +
                  ["-lbct", "Neumann"] +
                  ["-lbc", "0.0"] +
                  ["-rbct", "Neumann"] +
                  ["-st", solver_type] +
                  ["-d", "0.5"] +
                  ["-v", "0"] +
                  ["-dt", "0.00246"] +
                  ["-n", "50"] +
                  ["-N", "500"])
        p.communicate()

    solution = np.load(output_file).transpose()
    if run_animation:
        animate(solution, grid, show=show, save=save, grid=show_grid, name=name)

    return grid, solution


def run_wave_1D(solver_type="ExplicitEuler",
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
        np.save(GRID_FILE, grid)
        np.save(INITIAL_CONDITION_FILE, ic)

        p = Popen([chosenBin] +
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
                  ["-dt", "0.035"] +
                  ["-n", "100"] +
                  ["-N", "50"] +
                  ["-dbg"])
        p.communicate()

    solution = np.load(output_file).transpose()

    if run_animation:
        animate(solution, grid, show=show, grid=show_grid, save=save, name=name)

    return grid, solution


def __compare_solver_worker(worker, solver_list, is_space_solver=False,
                            run=True, show=True, show_grid=False, save=False, name="comparison.gif",
                            y_lim=None):
    out = []
    for solver in solver_list:
        print("Running {}".format(solver))
        if not is_space_solver:
            out.append(worker(solver_type=solver, output_file="{}.cl".format(solver), run=run, show=False,
                              show_grid=show_grid, save=False, run_animation=False))
        else:
            out.append(worker(space_discretizer=solver, output_file="{}.cl".format(solver), run=run, show=False,
                              show_grid=show_grid, save=False, run_animation=False))

    animate_multicurve([x[1] for x in out], [x[0] for x in out], grid=show_grid, show=show,
                       labels=solver_list, save=save, name=name, y_lim=y_lim)


def compare_solvers_transport_1D(solver_list, run=True, show=True, show_grid=False, save=False, name="comparison.gif"):
    __compare_solver_worker(run_transport_1D, solver_list, is_space_solver=True,
                            run=run, show=show, show_grid=show_grid, save=save, name=name)


def compare_solvers_diffusion_1D(solver_list, run=True, show=True, show_grid=False, save=False, name="comparison.gif"):
    __compare_solver_worker(run_diffusion_1D, solver_list,
                            run=run, show=show, show_grid=show_grid, save=save, name=name)


def compare_solvers_wave_1D(solver_list, run=True, show=True, show_grid=False, save=False, name="comparison.gif"):
    __compare_solver_worker(run_wave_1D, solver_list,
                            run=run, show=show, show_grid=show_grid, save=save, name=name, y_lim=(-1.1, 1.1))


def run_transport_2D(space_discretizer="Upwind",
                     output_file="transport2d.cl",
                     name="transport2d.gif",
                     run=True, show=True, save=False,
                     run_animation=True):

    try:
        os.remove(X_GRID_FILE)
        os.remove(Y_GRID_FILE)
        os.remove(INITIAL_CONDITION_FILE)
    except FileNotFoundError:
        pass

    x_grid = np.linspace(-np.pi, np.pi, 128)
    y_grid = np.linspace(-np.pi, np.pi, 128)
    X, Y = np.meshgrid(x_grid, y_grid)
    ic = np.exp(-X * X - Y * Y)
    N = 50
    if run:
        np.save(X_GRID_FILE, x_grid)
        np.save(Y_GRID_FILE, y_grid)
        np.save(INITIAL_CONDITION_FILE, ic)

        p = Popen([chosenBin] +
                  ["-dbg"] +
                  ["-dim", "2"] +
                  ["-ic", INITIAL_CONDITION_FILE] +
                  ["-gx", X_GRID_FILE] +
                  ["-gy", Y_GRID_FILE] +
                  ["-of", output_file] +
                  ["-md", "Double"] +
                  ["-ubct", "Periodic"] +
                  ["-ubc", "0.0"] +
                  ["-dbct", "Periodic"] +
                  ["-dbc", "0.0"] +
                  ["-lbct", "Periodic"] +
                  ["-lbc", "0.0"] +
                  ["-rbct", "Periodic"] +
                  ["-st", "ExplicitEuler"] +
                  ["-sdt", space_discretizer] +
                  ["-d", "0"] +
                  ["-vx", "0.5"] +
                  ["-vy", "0.5"] +
                  ["-dt", "0.05"] +
                  ["-n", "10"] +
                  ["-N", str(N)])
        p.communicate()

    _solution = np.load(output_file).flatten()
    solution = []
    for n in range(N):
        m = np.zeros((len(x_grid), len(y_grid)))
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                m[i, j] = _solution[i + j * len(x_grid) + n * len(x_grid) * len(y_grid)]
        solution.append(m)
    solution = np.array(solution)
    np.savetxt("a1.txt", solution[0])
    np.savetxt("a2.txt", solution[1])
    if run_animation:
        animate_3D(solution, x_grid, y_grid, show=show, save=save, name=name, rstride=4, cstride=4)

    return x_grid, y_grid, solution


def run_diffusion_2D(solver_type="ImplicitEuler", output_file="diffusion2d.cl",
                     name="diffusion2d.gif",
                     run=True, show=True, save=False,
                     run_animation=True):

    try:
        os.remove(X_GRID_FILE)
        os.remove(Y_GRID_FILE)
        os.remove(INITIAL_CONDITION_FILE)
    except FileNotFoundError:
        pass

    x_grid = np.linspace(-np.pi, np.pi, 64)
    y_grid = np.linspace(-np.pi, np.pi, 64)
    X, Y = np.meshgrid(x_grid, y_grid)
    ic = np.exp(-X ** 2 - Y ** 2)
    N = 50
    if run:
        np.save(X_GRID_FILE, x_grid)
        np.save(Y_GRID_FILE, y_grid)
        np.save(INITIAL_CONDITION_FILE, ic)

        p = Popen([chosenBin] +
                  ["-dbg"] +
                  ["-dim", "2"] +
                  ["-ic", INITIAL_CONDITION_FILE] +
                  ["-gx", X_GRID_FILE] +
                  ["-gy", Y_GRID_FILE] +
                  ["-of", output_file] +
                  ["-md", "Double"] +
                  ["-lbct", "Neumann"] +
                  ["-lbc", "0.0"] +
                  ["-rbct", "Neumann"] +
                  ["-st", solver_type] +
                  ["-sdt", "Centered"] +
                  ["-d", "1"] +
                  ["-vx", "0"] +
                  ["-vy", "0"] +
                  ["-dt", "0.005"] +
                  ["-n", "10"] +
                  ["-N", str(N)])
        p.communicate()

    _solution = np.load(output_file).flatten()
    solution = []
    for n in range(N):
        m = np.zeros((len(x_grid), len(y_grid)))
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                m[i, j] = _solution[i + j * len(x_grid) + n * len(x_grid) * len(y_grid)]
        solution.append(m)
    solution = np.array(solution)
    if run_animation:
        animate_3D(solution, x_grid, y_grid, show=show, save=save, name=name, rstride=2, cstride=2, fixed_view=True)

    return x_grid, y_grid, solution


def run_wave_2D(solver_type="ImplicitEuler", output_file="wave2d.cl",
                     name="wave2d.gif",
                     run=True, show=True, save=False,
                     run_animation=True):

    try:
        os.remove(X_GRID_FILE)
        os.remove(Y_GRID_FILE)
        os.remove(INITIAL_CONDITION_FILE)
    except FileNotFoundError:
        pass

    x_grid = np.linspace(-np.pi, np.pi, 64)
    y_grid = np.linspace(-np.pi, np.pi, 64)
    X, Y = np.meshgrid(x_grid, y_grid)
    ic = np.exp(-X ** 2 - Y ** 2)

    np.save(X_GRID_FILE, x_grid)
    np.save(Y_GRID_FILE, x_grid)
    np.save(INITIAL_CONDITION_FILE, ic)
    N = 50
    if run:
        p = Popen([debugBin] +
                  ["-dbg"] +
                  ["-pde", "WaveEquation"] +
                  ["-dim", "2"] +
                  ["-ic", INITIAL_CONDITION_FILE] +
                  ["-gx", X_GRID_FILE] +
                  ["-gy", Y_GRID_FILE] +
                  ["-of", output_file] +
                  ["-md", "Double"] +
                  ["-lbct", "Neumann"] +
                  ["-lbc", "0.0"] +
                  ["-rbct", "Neumann"] +
                  ["-st", solver_type] +
                  ["-sdt", "Centered"] +
                  ["-d", "0"] +
                  ["-vx", "0.05"] +
                  ["-vy", "0.0"] +
                  ["-dt", "0.1"] +
                  ["-n", "20"] +
                  ["-N", str(N)])
        p.communicate()

    _solution = np.load(output_file).flatten()
    solution = []
    for n in range(N):
        m = np.zeros((len(x_grid), len(y_grid)))
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                m[i, j] = _solution[i + j * len(x_grid) + n * len(x_grid) * len(y_grid)]
        solution.append(m)
    solution = np.array(solution)
    if run_animation:
        animate_3D(solution, x_grid, y_grid, show=show, save=save, name=name, rstride=2, cstride=2, fixed_view=True)

    return x_grid, y_grid, solution


if __name__ == "__main__":

    #run_transport_1D()
    # compare_solvers_transport_1D(["LaxWendroff", "Upwind"], run=True, show=True, show_grid=True,
    #                              save=False, name="numericalDiffusion.gif")

    # run_diffusion_1D()
    # compare_solvers_diffusion_1D(["ExplicitEuler", "ImplicitEuler", "CrankNicolson"],
    #                              run=True, show=True, show_grid=True,
    #                              save=False, name="multiStep.gif")
    # compare_solvers_diffusion_1D(["AdamsBashforth2", "AdamsMouldon2", "CrankNicolson"],
    #                              run=True, show=True, show_grid=True,
    #                              save=True, name="multiStep.gif")

    # run_wave_1D("ImplicitEuler")
    # compare_solvers_wave_1D(["ExplicitEuler", "ImplicitEuler"],
    #                         run=True, show=True, show_grid=True,
    #                         save=False, name="waveInstability1D.gif")

    # run_transport_2D(run=True, save=False, show=True)
    # run_diffusion_2D(run=True, save=False, show=True)
    run_wave_2D(run=True, save=False, show=True)
