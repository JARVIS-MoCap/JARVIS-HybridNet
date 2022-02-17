"""
project_manager.py
==================
"""

import os,sys,inspect
import ruamel.yaml
import shutil
from yacs.config import CfgNode as CN
import numpy as np
import streamlit as st


from jarvis.config import cfg
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.utils.utils import CLIColors


class ProjectManager:
    """
    Project Manager Class to load and setup projects and find suitable values
    for network parameters.
    """
    def __init__(self):
        self.cfg = None
        current_dir = os.path.dirname(
                    os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.parent_dir = os.path.dirname(os.path.dirname(current_dir))


    def load(self, project_name):
        """
        Load an existing project.

        :param project_name: Name of the project
        :type project_name: string
        """
        self.cfg = cfg
        self.cfg.PROJECT_NAME = project_name
        if not (os.path.isfile(os.path.join(self.parent_dir,
                    self.cfg.PROJECTS_ROOT_PATH, project_name, 'config.yaml'))):
            print (CLIColors.FAIL + 'Project does not exist, change name or create new Project '
                   'by calling create_new(...).' + CLIColors.ENDC)
            self.cfg = None
            return False

        cfg.merge_from_file(os.path.join(self.parent_dir,
                    self.cfg.PROJECTS_ROOT_PATH, project_name, 'config.yaml'))
        self.cfg.logPaths = CN()
        self.cfg.savePaths = CN()
        for module in ['CenterDetect', 'KeypointDetect', 'HybridNet']:
            model_savepath = os.path.join(self.parent_dir,
                        self.cfg.PROJECTS_ROOT_PATH, project_name, 'models',
                        module)
            log_path = os.path.join(self.parent_dir,self.cfg.PROJECTS_ROOT_PATH, project_name,
                        'logs', module)
            self.cfg.savePaths[module] = model_savepath
            self.cfg.logPaths[module] = log_path
        self.cfg.PARENT_DIR = self.parent_dir
        print (CLIColors.OKGREEN  + 'Successfully loaded project ' + project_name + '.' + CLIColors.ENDC)
        return True


    def create_new(self, name, dataset2D_path, dataset3D_path = None):
        """
        Create a new project. This sets up the project directory structure and
        initializes the config file values obtained by analyzing the training
        sets provided.

        :param name: Name of the new project
        :type name: string
        :param dataset2D_path: Path to the dataset that is used to train the
                               EffDet cropping and the EffTrack 2D tracking
                               network. This does not have to be the same
                               Dataset that is used for final 3D training.
        :type dataset2D_path: string
        :param dataset3D_path: path to the dataset that is used to train the
                               full 3D network. If None, the config will not try
                               to setup 3D Network parameters.
        :type dataset3D_path: string

        """
        self.cfg = cfg
        if (os.path.isfile(os.path.join(self.parent_dir, self.cfg.PROJECTS_ROOT_PATH, name,
                   'config.yaml'))):
            print (f'{CLIColors.FAIL}Project already exist, change name or delete old project.{CLIColors.ENDC}')
            self.cfg = None
            return False

        if not os.path.isdir(dataset2D_path):
            print (f'{CLIColors.FAIL}Dataset2D directory does not exist.{CLIColors.ENDC}')
            return False
        if dataset3D_path != None:
            if not os.path.isdir(dataset3D_path):
                print (f'{CLIColors.FAIL}Dataset3D directory does not exist.{CLIColors.ENDC}')
                return False

        self.cfg.PROJECT_NAME = name
        self.cfg.DATASET.DATASET_2D = dataset2D_path
        self.cfg.DATASET.DATASET_3D = dataset3D_path
        os.makedirs(os.path.join(self.parent_dir, cfg.PROJECTS_ROOT_PATH, name), exist_ok=True)

        self.cfg.logPaths = CN()
        self.cfg.savePaths = CN()

        for module in ['CenterDetect', 'KeypointDetect', 'HybridNet']:
            model_savepath = os.path.join(self.parent_dir, self.cfg.PROJECTS_ROOT_PATH, name,
                        'models', module)
            log_path = os.path.join(self.parent_dir, self.cfg.PROJECTS_ROOT_PATH, name,
                        'logs', module)
            self.cfg.savePaths[module] = model_savepath
            self.cfg.logPaths[module] = log_path
            os.makedirs(log_path, exist_ok=True)
            os.makedirs(model_savepath, exist_ok=True)
        self._init_dataset2D()
        if dataset3D_path != None:
            self._init_dataset3D()
        self._init_config(name)
        print (f'{CLIColors.OKGREEN}Project created succesfully.{CLIColors.ENDC}')
        return True


    def get_create_config_interactive(self, name, dataset2D_path, dataset3D_path = None):
        st.session_state['creating_project'] = True
        self.cfg = cfg
        self.cfg.PROJECT_NAME = name
        self.cfg.DATASET.DATASET_2D = dataset2D_path
        self.cfg.DATASET.DATASET_3D = dataset3D_path

        if (os.path.isfile(os.path.join(self.parent_dir, self.cfg.PROJECTS_ROOT_PATH, name,
                   'config.yaml'))):
            st.error("Project already exists, please choose a different name.")
            return
        if dataset3D_path != None:
            if not os.path.isdir(dataset3D_path):
                st.error("Dataset3D directory does not exist..")
                return
        if not os.path.isdir(dataset2D_path):
            st.error("Dataset2D directory does not exist..")
            return

        dataset2D = Dataset2D(self.cfg, set='train', mode = 'keypoints')
        suggested_bbox_size = dataset2D.get_dataset_config()
        if dataset3D_path != None:
            dataset3D = Dataset3D(self.cfg, set='train')
            suggestions = dataset3D.get_dataset_config()
        st.title("Project Configuration")
        with st.form("config_form"):
            bbox_size = st.number_input("2D bounding box size (has to be divisible by 64):", value = suggested_bbox_size, min_value=64, max_value=6400, step=64)
            if dataset3D_path != None:
                bbox_size_3D = st.number_input("3D tracking Volume size (has to be divisible by 8):", value = suggestions["bbox"], min_value=8, step=8)
                grid_spacing = st.number_input("Grid spacing:", value = int(np.floor((bbox_size_3D/100.))), min_value=1, max_value=1024, step=1)
            submitted2 = st.form_submit_button("Confirm")
        if submitted2:
            if bbox_size % 64 != 0:
                st.error("2D bounding box size has to be divisible by 64.")
                return
            if bbox_size_3D % 8 != 0:
                st.error("3D bounding box size has to be divisible by 8.")
                return
            if grid_spacing > bbox_size_3D:
                st.error("Grid spacing can not be bigger than bounding box.")
                return
            os.makedirs(os.path.join(self.parent_dir, cfg.PROJECTS_ROOT_PATH, name), exist_ok=True)
            self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE = bbox_size
            self.cfg.KEYPOINTDETECT.NUM_JOINTS = dataset2D.num_keypoints[0]
            if dataset3D_path != None:
                self.cfg.HYBRIDNET.ROI_CUBE_SIZE = bbox_size_3D
                self.cfg.HYBRIDNET.GRID_SPACING = grid_spacing
                self.cfg.HYBRIDNET.NUM_CAMERAS = dataset3D.num_cameras
            self.cfg.logPaths = CN()
            self.cfg.savePaths = CN()
            for module in ['CenterDetect', 'KeypointDetect', 'HybridNet']:
                model_savepath = os.path.join(self.parent_dir, self.cfg.PROJECTS_ROOT_PATH, name,
                            'models', module)
                log_path = os.path.join(self.parent_dir, self.cfg.PROJECTS_ROOT_PATH, name,
                            'logs', module)
                self.cfg.savePaths[module] = model_savepath
                self.cfg.logPaths[module] = log_path
                os.makedirs(log_path, exist_ok=True)
                os.makedirs(model_savepath, exist_ok=True)
            self._init_config(name)
            st.session_state['creating_project'] = False
            st.session_state['created_project'] = name
            st.experimental_rerun()


    def get_cfg(self):
        """
        Get configuration handle for the configuration of the current project.
        """
        if self.cfg == None:
            print ('No Project loaded yet! Call either load(...) or '
                   'create_new(...).')
        return self.cfg

    def get_projects(self):
        return os.listdir(os.path.join(self.parent_dir, 'projects'))


    def _get_number_from_user(self, question, default, div = None,
        bounds = None):
        """
        Get a valid number divisible by div and in range bounds from the user.
        :param question: Question to ask if no selected
        :type question: string
        :param default: Default Value, returned if user selects yes
        :type default: int
        :param div: Number input should be divisble by. Not used if None
        :type div: int
        :param div: Range for input number
        :type bounds: list of two ints. Not used if None
        """
        number = default
        if div == None:
            div = 1
        valid_accepts = ['yes', 'Yes', 'y', 'Y']
        valid_declines = ['no', 'No', 'n', 'N']
        got_valid_answer = False
        while not got_valid_answer:
            ans = input()
            if ans in valid_declines:
                got_valid_answer = True
                print(question)
                ans_is_valid = False
                while not ans_is_valid:
                    ans = input()
                    if ans.isdigit() and int(ans)%div == 0:
                        if (bounds == None or int(ans) >= bounds[0]
                                    and int(ans) <= bounds[1]):
                            number = int(ans)
                            ans_is_valid = True
                        else:
                            print (f"Please enter a Number between {bounds[0]} "
                                   f"and {bounds[1]}!")
                    else:
                        print (f"Please enter a Number divisible by {div}!")
            elif ans in valid_accepts:
                got_valid_answer = True
            else:
                print ("Please enter either yes or no!")
        return number


    def _init_dataset2D(self):
        dataset2D = Dataset2D(self.cfg, set='train', mode = 'keypoints')
        suggested_bbox_size = dataset2D.get_dataset_config()
        print ('\KeypointDetect 2D Configuration:')
        print (f'Use suggested Bounding Box size of {suggested_bbox_size} '
               'pixels? (yes/no)')
        q = 'Enter custom Bounding Box size, make sure it is divisible by 64:'
        bbox_size = suggested_bbox_size
        bbox_size = self._get_number_from_user(q, suggested_bbox_size, 64)

        self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE = bbox_size
        self.cfg.KEYPOINTDETECT.NUM_JOINTS = dataset2D.num_keypoints[0]


    def _init_dataset3D(self):
        dataset3D = Dataset3D(self.cfg, set='train')
        suggestions  = dataset3D.get_dataset_config()
        print ('\HybridNet 3D Configuration:')
        print (f'Use suggested 3D Bounding Box size of {suggestions["bbox"]} '
               'mm? (yes/no)')
        q = 'Enter custom 3D Bounding Box size, make sure it is divisible by 8:'
        bbox_size = suggestions["bbox"]
        bbox_size = self._get_number_from_user(q, suggestions["bbox"], 8)
        resolution_suggestion = int(np.floor((bbox_size/100.)))

        print (f'Use suggested grid spacing of {resolution_suggestion} '
               'mm? (yes/no)')
        q = 'Enter custom grid spacing:'
        resolution = resolution_suggestion
        resolution = self._get_number_from_user(q, resolution_suggestion,
                    bounds = [0,4])
        self.cfg.HYBRIDNET.ROI_CUBE_SIZE = bbox_size
        self.cfg.HYBRIDNET.GRID_SPACING = resolution
        self.cfg.HYBRIDNET.NUM_CAMERAS = dataset3D.num_cameras


    def _init_config(self, name):
        config_path = os.path.join(self.parent_dir, cfg.PROJECTS_ROOT_PATH, name, 'config.yaml')
        shutil.copyfile(os.path.join(self.parent_dir, 'jarvis/config/config_template.yaml'), config_path)
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
                    print (k,v)
