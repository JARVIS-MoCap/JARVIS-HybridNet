import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from lib.vortex.utils import Camera
import os
import mpl_toolkits.mplot3d as mplot3d
import itertools

class SetupVisualizer():
    def __init__(self, primary_camera, root_dir, intrinsics, extrinsics):
        self.cameras = {}
        for camera in intrinsics:
            if camera == primary_camera:
                self.cameras[camera] = Camera(camera, True, os.path.join(root_dir, intrinsics[camera]), None)
            else:
                self.cameras[camera] = Camera(camera, False, os.path.join(root_dir, intrinsics[camera]), os.path.join(root_dir, extrinsics[camera]))
        self.camera_list = [self.cameras[cam] for cam in self.cameras]
        self.num_cameras = len(self.camera_list)


    def plot_cameras(self, axes = None):
        show_plot = False
        if axes == None:
            figure = plt.figure()
            axes = figure.gca(projection='3d')
            show_plot = True

        cam_positions = []
        for cam in self.cameras:
            camera_mesh = mesh.Mesh.from_file('/home/timo/Desktop/VoRTEx/lib/dataset/Camera.stl')
            camera_mesh.points = camera_mesh.points*0.7
            rotMat = np.eye(3)
            rotMat[0,0] = -1
            rotMat[1,1] = 1
            rotMat[2,2] = -1
            cam_positions.append(np.dot(rotMat,np.dot(self.cameras[cam].rotationMatrix, self.cameras[cam].position)))
            camera_mesh.rotate_using_matrix(rotMat)
            camera_mesh.translate(-self.cameras[cam].position)
            camera_mesh.rotate_using_matrix(np.transpose(self.cameras[cam].rotationMatrix))
            if self.cameras[cam].primary:
                axes.add_collection3d(mplot3d.art3d.Poly3DCollection(camera_mesh.vectors, color='r'))
            else:
                axes.add_collection3d(mplot3d.art3d.Poly3DCollection(camera_mesh.vectors, color='b'))

        cam_positions = np.array(cam_positions).squeeze()
        x_range = np.array([np.min(cam_positions[:,0]), np.max(cam_positions[:,0])])
        y_range = np.array([np.min(cam_positions[:,1]), np.max(cam_positions[:,1])])
        z_range = np.array([np.min(cam_positions[:,2]), np.max(cam_positions[:,2])])
        ranges = np.array([x_range, y_range, z_range])
        max_range = np.max(ranges[:,1]-ranges[:,0])
        centers = (ranges[:,1]+ranges[:,0])/2.

        axes.set_xlim3d(centers[0]-max_range/2, centers[0]+max_range/2)
        axes.set_ylim3d(centers[1]-max_range/2, centers[1]+max_range/2)
        axes.set_zlim3d(centers[2]-max_range/2, centers[2]+max_range/2)
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)

        if show_plot:
            plt.show()

    def plot_tracking_area(self, tracking_area, axes = None):
        show_plot = False
        if axes == None:
            figure = plt.figure()
            axes = figure.gca(projection='3d')
            show_plot = True
        tracking_area = tracking_area.transpose()
        cube = np.array(list(itertools.product(*zip(tracking_area[0],tracking_area[1]))))
        cube = cube
        for corner in cube:
            axes.scatter(corner[0],corner[1],corner[2], c = 'green', s = 20)
        vertices = [[0,1], [0,2], [2,3], [1,3], [4,5], [4,6], [6,7], [5,7], [0,4], [1,5], [2,6], [3,7]]
        for vertex in vertices:
            axes.plot([cube[vertex[0]][0],cube[vertex[1]][0]],[cube[vertex[0]][1],cube[vertex[1]][1]],[cube[vertex[0]][2],cube[vertex[1]][2]],c = 'green')
        if show_plot:
            plt.show()

    def plot_datapoints(self, points, axes):
        show_plot = False
        if axes == None:
            figure = plt.figure()
            axes = figure.gca(projection='3d')
            show_plot = True
        for i, point in enumerate(points):
            axes.scatter(point[0], point[1], point[2])

        if show_plot:
            plt.show()
