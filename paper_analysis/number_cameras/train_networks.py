import numpy as np
import os

import jarvis.train_interface as train


camera_names = ['Camera_B', 'Camera_LBB', 'Camera_LBT', 'Camera_LC', 'Camera_LFB', 'Camera_LFT', 'Camera_RBB', 'Camera_RBT', 'Camera_RC', 'Camera_RFB', 'Camera_RFT', 'Camera_T']

num_cameras = 4
assert num_cameras > 1 and num_cameras < 12, "Please select a number of cameras between 2 and 11!"

sets = np.genfromtxt(os.path.join('camera_sets', f'Set_{num_cameras}.csv'), delimiter=',').astype(int)

for i,set in enumerate(sets):
    camera_list = [camera_names[i] for i in set]
    print ("Training for:", camera_list)

    train.train_efficienttrack('CenterDetect', 'Hand_Num_Cameras',
                1, 'ecoset', run_name = f'Camera_Numbers_{num_cameras}_{i}', cameras_to_use = camera_list)

    train.train_efficienttrack('KeypointDetect', 'Hand_Num_Cameras',
                1, 'ecoset', run_name = f'Camera_Numbers_{num_cameras}_{i}', cameras_to_use = camera_list)

    train.train_hybridnet('Hand_Num_Cameras', 1,
                'latest', None, '3D_only')

    train.train_hybridnet('Hand_Num_Cameras', 1,
                None, 'latest', 'all', finetune = True,
                run_name = f'Camera_Numbers_{num_cameras}_{i}', cameras_to_use = camera_list)
