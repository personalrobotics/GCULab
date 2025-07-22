import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import to_rgba
from packing3d import Display


def plot_cross_section(container_path):
    # Load container
    with open(container_path, "rb") as f:
        container = pickle.load(f)

    cube = np.array(container.geometry.cube)  # (z, x, y)

    # Compute fill stats
    filled_voxels = np.count_nonzero(cube)
    total_voxels = np.prod(cube.shape)
    fill_ratio = filled_voxels / total_voxels

    print(f"Total filled volume: {filled_voxels} unit³")
    print(f"Total container volume: {total_voxels} unit³")
    print(f"Fill ratio: {fill_ratio:.3%}")

    # Color mapping
    colors = ['lightcoral', 'lightsalmon', 'gold', 'olive',
              'mediumaquamarine', 'deepskyblue', 'blueviolet', 'pink',
              'brown', 'darkorange', 'yellow', 'lawngreen', 'turquoise',
              'dodgerblue', 'darkorchid', 'hotpink', 'deeppink', 'peru',
              'orange', 'darkolivegreen', 'cyan', 'purple', 'crimson']
    color_rgba = np.asarray([to_rgba(c) for c in colors])

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    z_max = cube.shape[0] - 1
    slice_ax = ax.imshow(np.zeros(cube.shape[1:]), origin='lower')
    title = ax.set_title("Cross-section Z=0 | Filled Area = 0 unit²")

    ax_z = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider_z = Slider(ax_z, "Z Height", 0, z_max, valinit=0, valstep=1)

    def update(z):
        z = int(z)
        slice_data = cube[z]
        filled_area = np.count_nonzero(slice_data)

        # Map ints to RGBA
        flat = slice_data.astype(int).flatten()
        rgba = np.zeros((flat.size, 4))
        mask = flat > 0
        rgba[mask] = color_rgba[flat[mask] % len(color_rgba)]
        rgba = rgba.reshape((*slice_data.shape, 4))

        slice_ax.set_data(rgba)
        title.set_text(f"Cross-section Z={z} | Filled Area = {filled_area} unit²")
        fig.canvas.draw_idle()

    slider_z.on_changed(update)
    update(0)

    plt.show(block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D container cross-sections with a slider.")
    parser.add_argument("--data_path", type=str, required=True, help="Root path to the saved container data.")
    parser.add_argument("--env_id", type=int, required=True, help="Environment index (e.g., 0, 1, ...).")
    parser.add_argument("--step_id", type=int, required=True, help="Step index for the container (e.g., 0, 23, ...).")
    args = parser.parse_args()

    container_path = os.path.join(args.data_path, "containers", f"env_{args.env_id}", f"{args.step_id}.pkl")
    plot_cross_section(container_path)
