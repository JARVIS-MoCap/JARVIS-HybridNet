import os
import numpy as np
import matplotlib.pyplot as plt

from jarvis.visualization.visualization_utils import get_colors_and_lines

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.4*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_slices(csv_file, filename, start_frame, num_frames, skip_number, skeleton_preset, plot_azim, plot_elev):
    if not os.path.isfile(csv_file):
        print ('3D Coordinate CSV file does not exist!')
        return
    data = np.genfromtxt(csv_file, delimiter=',')[1:] #TODO make this work depending on header is 1 or 2 lines long
    data = data.reshape([data.shape[0], -1, 3])
    if skeleton_preset != None:
        colors, line_idxs =  get_colors_and_lines(skeleton_preset)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # List to save your projections to
    projections = []
    projections.append((ax.azim, ax.elev))

    # This is called everytime you release the mouse button
    def on_click(event):
        azim, elev = ax.azim, ax.elev
        projections.append((azim, elev))

    cid = fig.canvas.mpl_connect('button_release_event', on_click)
    for i, point in enumerate(data[start_frame]):
        print (i, point)
        print (colors[i], len(colors))
        ax.scatter(point[0], point[1], point[2], color = colors[i], s = 10)
    for line in line_idxs:
        ax.plot([data[start_frame][line[0]][0], data[start_frame][line[1]][0]],
                  [data[start_frame][line[0]][1], data[start_frame][line[1]][1]],
                  [data[start_frame][line[0]][2], data[start_frame][line[1]][2]],
                  c = colors[line[1]])
    set_axes_equal(ax)
    plt.show()

    projection = projections[-1]
    if plot_azim != None and plot_elev != None:
        projection = (float(plot_azim), float(plot_elev))

    print (projection)


    fig, axs = plt.subplots(1, num_frames, subplot_kw={'projection': '3d'})
    for frame in range(num_frames):
        data_ind = frame*skip_number+start_frame
        axs[frame].set_axis_off()
        axs[frame].margins(0)
        axs[frame].azim = projection[0]
        axs[frame].elev = projection[1]
        for i, point in enumerate(data[data_ind]):
            axs[frame].scatter(point[0], point[1], point[2], color = colors[i])
        for line in line_idxs:
            axs[frame].plot([data[data_ind][line[0]][0], data[data_ind][line[1]][0]],
                      [data[data_ind][line[0]][1], data[data_ind][line[1]][1]],
                      [data[data_ind][line[0]][2], data[data_ind][line[1]][2]],
                      c = colors[line[1]])
        set_axes_equal(axs[frame])
        axs[frame].autoscale_view('tight')
    plt.subplots_adjust(wspace=0, hspace=0, right=1, left = 0, top = 1, bottom = 0)
    plt.savefig(filename, dpi = 800)
    plt.show()
