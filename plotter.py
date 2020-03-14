import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def __ax_formatter(ax, grid=False, title="", x_label="", y_label="", show_legend=False):
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if grid:
        ax.grid()

    if show_legend:
        ax.legend(loc='best')


def surf(z, x, y, title="", x_label="", y_label="", show_legend=False, show=False, rstride=1, cstride=1, cmap=cm.coolwarm):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Create X and Y data
    x_grid, y_grid = np.meshgrid(x, y)

    ax.plot_surface(x_grid, y_grid, z, cmap=cmap, rstride=rstride, cstride=cstride, antialiased=True)

    __ax_formatter(ax, False, title, x_label, y_label, show_legend)

    if show:
        plt.show()


def plot(z, x, grid=False, title="", x_label="", y_label="", show_legend=False, show=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(x)):
        ax.plot(z[:, i])

    __ax_formatter(ax, grid, title, x_label, y_label, show_legend)

    if show:
        plt.show()


def animate(z, x, grid=False, show=False, save=False, label="", name="", y_lim=None):
    fig, ax = plt.subplots()

    line, = ax.plot(x, z[:, 0], label=label)
    if grid:
        ax.grid()

    def update_line(i):
        if i >= z.shape[1]:
            return line,
        line.set_ydata(z[:, i])  # update the data
        return line,

    # Init only required for blitting to give a clean slate.
    def init():
        line.set_ydata(z[:, 0])
        return line,

    ani = animation.FuncAnimation(fig, update_line, np.arange(1, 200), init_func=init, interval=25, blit=True)

    if show:
        plt.show()

    if y_lim is not None:
        ax.set_ylim(y_lim)

    if save:
        ani.save(name, fps=60, writer="imagemagick")


def animate_multicurve(z_vector, x_vector, grid=False, show=False, save=False, labels=None, name="",
                       y_lim=None):
    assert(len(z_vector) == len(x_vector))

    fig, ax = plt.subplots()

    lines = []

    # Init only required for blitting to give a clean slate.
    def init():
        for line in lines:
            line.set_ydata(z[:, 0])
        return tuple(lines)

    def update_line(i):
        if i >= z.shape[1]:
            return tuple(lines)
        for j, line in enumerate(lines):
            line.set_ydata(z_vector[j][:, i])  # update the data
        return tuple(lines)

    show_legend = True
    if labels is None:
        show_legend = False
        labels = [""] * len(x_vector)
    for x, z, l in zip(x_vector, z_vector, labels):
        line, = ax.plot(x, z[:, 0], label=l)
        lines.append(line)

    if grid:
        ax.grid()

    if show_legend:
        ax.legend(loc='best')

    if y_lim is not None:
        ax.set_ylim(y_lim)

    ani = animation.FuncAnimation(fig, update_line, np.arange(1, 200), init_func=init, interval=25, blit=True)

    if save:
        ani.save(name, fps=60, writer="imagemagick")

    if show:
        plt.show()


def animate_3D(z, x, y, rstride=1, cstride=1, cmap=cm.coolwarm, show=False, save=False, name="", fixed_view=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Create X and Y data
    x_grid, y_grid = np.meshgrid(x, y)

    line = ax.plot_surface(x_grid, y_grid, z[0], cmap=cmap, rstride=rstride, cstride=cstride, antialiased=True)
    z_lim = ax.get_zlim()

    def update_line(i):
        if i >= z.shape[0]:
            return line,
        ax.clear()
        l = ax.plot_surface(x_grid, y_grid, z[i], cmap=cm.coolwarm, rstride=rstride, cstride=cstride, antialiased=True)
        if fixed_view:
            ax.set_zlim(z_lim)
        return l,

    ani = animation.FuncAnimation(fig, update_line, np.arange(1, 200), interval=25, blit=False)

    if show:
        plt.show()

    if save:
        ani.save(name, writer='imagemagick', fps=60)


def animate_colormap(z, x, y, cmap='PuBu_r', shading='gouraud', show=False, save=False, name=""):
    fig, ax = plt.subplots()
    x, y = np.meshgrid(x, y)
    ax.pcolormesh(x, y, z[0], shading=shading, cmap=cmap)

    def update_line(i):
        if i >= z.shape[0]:
            return None,
        ax.clear()
        ax.pcolormesh(x, y, z[i], shading=shading, cmap=cmap)
        return None,

    ani = animation.FuncAnimation(fig, update_line, np.arange(1, 200), interval=25, blit=False)

    if show:
        plt.show()

    if save:
        ani.save(name, writer='imagemagick', fps=60)