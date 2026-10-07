import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from Wave2D import Wave2D_Neumann


def movie():
    solver = Wave2D_Neumann()
    data = solver(40, 40, cfl=1 / np.sqrt(2), c=1, mx=2, my=2, store_data=1)
    xij, yij = solver.create_mesh(40)
    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={"projection": "3d"})
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(-1, 1), xlabel="x", ylabel="y", zlabel="u", title="Neumann wave")
    ax.view_init(elev=25, azim=-60)
    frames = []
    for n, val in data.items():
        frame = ax.plot_surface(xij, yij, val, cmap=cm.coolwarm)
        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True, repeat_delay=1000)
    ani.save("report/neumannwave.gif", writer="pillow", fps=5, dpi=75)
    plt.close(fig)


if __name__ == "__main__":
    movie()
