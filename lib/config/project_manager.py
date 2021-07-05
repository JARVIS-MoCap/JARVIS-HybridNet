"""
project_manager.py
===============
Project Manager Class to load and setup projects and find suitable values for Network
parameters.
"""


import os
import ruamel.yaml
import shutil
from yacs.config import CfgNode as CN

from lib.config import cfg
from lib.dataset.dataset2D import VortexDataset2D
from lib.dataset.dataset3D import VortexDataset3D


class ProjectManager:
    def __init__(self):
        self.cfg = None

    def load(self, project_name):
        self.cfg = cfg
        self.cfg.PROJECT_NAME = project_name
        if not (os.path.isfile(os.path.join(self.cfg.PROJECTS_ROOT_PATH, project_name, 'config.yaml'))):
            print ('Project does not exist, change name or create new Project by calling create_new(...).')
            self.cfg = None
            return
        cfg.merge_from_file(os.path.join(self.cfg.PROJECTS_ROOT_PATH, project_name, 'config.yaml'))
        print (self.cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE)
        self.cfg.logPaths = CN()
        self.cfg.savePaths = CN()
        for module in ['efficientdet', 'efficienttrack', 'vortex']:
            model_savepath = os.path.join(self.cfg.PROJECTS_ROOT_PATH, project_name, module, 'models')
            log_path = os.path.join(self.cfg.PROJECTS_ROOT_PATH, project_name, module, 'logs')
            self.cfg.savePaths[module] = model_savepath
            self.cfg.logPaths[module] = log_path
        print ('Successfully loaded project ' + project_name + '!')


    def create_new(self, name, dataset2D_path, dataset3D_path = None):
        self.cfg = cfg
        #if (os.path.isfile(os.path.join(self.cfg.PROJECTS_ROOT_PATH, name, 'config.yaml'))):
        #    print ('Project already exist, change name or delete old project.')
        #    self.cfg = None
        #    return
        self.cfg.PROJECT_NAME = name
        self.cfg.DATASET.DATASET_2D = dataset2D_path
        self.cfg.DATASET.DATASET_3D = dataset3D_path
        os.makedirs(os.path.join(cfg.PROJECTS_ROOT_PATH, name), exist_ok=True)

        self.cfg.logPaths = CN()
        self.cfg.savePaths = CN()
        for module in ['efficientdet', 'efficienttrack', 'vortex']:
            model_savepath = os.path.join(self.cfg.PROJECTS_ROOT_PATH, name, module, 'models')
            log_path = os.path.join(self.cfg.PROJECTS_ROOT_PATH, name, module, 'logs')
            self.cfg.savePaths[module] = model_savepath
            self.cfg.logPaths[module] = log_path
            os.makedirs(log_path, exist_ok=True)
            os.makedirs(model_savepath, exist_ok=True)
        self._init_dataset2D()
        if dataset3D_path != None:
            self._init_dataset3D()
        self._init_config(name)


    def get_cfg(self):
        if self.cfg == None:
            print ('No Project loaded yet! Call either load(...) or create_new(...).')
        return self.cfg

    def _init_dataset2D(self):
        dataset2D = VortexDataset2D(self.cfg, set='train', mode = 'keypoints')
        suggested_bbox_size = dataset2D.get_dataset_config()
        print ('\nEfficientTrack Configuration:')
        print (f'Use suggested Bounding Box size of {suggested_bbox_size} pixels? (yes/no)')
        ans = input()
        if ans == 'no' or ans == 'n':
            print('Enter custom Bounding Box size, make sure it is divisible by 64:')
            suggested_bbox_size = int(input())
        self.cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE = suggested_bbox_size

    def _init_dataset3D(self):
        dataset3D = VortexDataset3D(self.cfg, set='train')
        suggestions  = dataset3D.get_dataset_config()
        print ('\nVortex Configuration:')
        print (f'Use suggested 3D Bounding Box size of {suggestions["bbox"]} mm? (yes/no)')
        ans = input()
        if ans == 'no' or ans == 'n':
            print('Enter custom 3D Bounding Box size, make sure it is divisible by 8:')
            suggestions['bbox'] = int(input())
        print (f'Use suggested grid spacing of {suggestions["resolution"]} mm? (yes/no)')
        ans = input()
        if ans == 'no' or ans == 'n':
            print('Enter custom grid spacing:')
            suggestions['resolution'] = int(input())
        print (f'Use suggested x-Axis Grid Dimension of {suggestions["grid_x"]} mm? (yes/no)')
        ans = input()
        if ans == 'no' or ans == 'n':
            print('Enter custom x-Axis Grid Dimensions, make sure they are divisible by your resolution:')
            suggestions['grid_x'] = int(input())
        print (f'Use suggested y-Axis Grid Dimension of {suggestions["grid_y"]} mm? (yes/no)')
        ans = input()
        if ans == 'no' or ans == 'n':
            print('Enter custom y-Axis Grid Dimension, make sure they are divisible by your resolution:')
            suggestions['grid_y'] = int(input())
        print (f'Use suggested z-Axis Grid Dimension of {suggestions["grid_z"]} mm? (yes/no)')
        ans = input()
        if ans == 'no' or ans == 'n':
            print('Enter custom z-Axis Grid Dimension, make sure they are divisible by your resolution:')
            suggestions['grid_z'] = int(input())

        self.cfg.VORTEX.ROI_CUBE_SIZE = suggestions['bbox']
        self.cfg.VORTEX.GRID_SPACING = suggestions['resolution']
        self.cfg.VORTEX.GRID_DIM_X = suggestions['grid_x']
        self.cfg.VORTEX.GRID_DIM_Y = suggestions['grid_y']
        self.cfg.VORTEX.GRID_DIM_Z = suggestions['grid_z']

    def _init_config(self, name):
        config_path = os.path.join(cfg.PROJECTS_ROOT_PATH, name, 'config.yaml')
        shutil.copyfile('lib/config/config_template.yaml', config_path)
        with open(config_path, 'r') as stream:
            data = ruamel.yaml.load(stream, Loader=ruamel.yaml.RoundTripLoader)
            self._update_values(data, self.cfg)

        with open(config_path, 'w') as outfile:
            ruamel.yaml.dump(data, outfile, Dumper=ruamel.yaml.RoundTripDumper)


    def _update_values(self, config_dict, cfg):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                self._update_values(v, cfg[k])
            else:
                try:
                    config_dict[k] = cfg[k]
                except:
                    print ('No val in cfg')
