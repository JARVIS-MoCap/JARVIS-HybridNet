import os,sys,inspect
from yacs.config import CfgNode as CN

#General Configurations
_C = CN()
_C.PROJECTS_ROOT_PATH = 'projects'
_C.PROJECT_NAME = None
_C.NUM_GPUS = 1
_C.GPU_IDS = None
_C.DATALOADER_NUM_WORKERS = 2
_C.USE_MIXED_PRECISION = True


#Dataset Configurations
_C.DATASET = CN()
_C.DATASET.DATASET_ROOT_DIR = 'datasets'
_C.DATASET.DATASET_2D = None
_C.DATASET.DATASET_3D = None
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.VAL_SET = 'val'
_C.DATASET.MEAN = [0.485, 0.456, 0.406]
_C.DATASET.STD = [0.229, 0.224, 0.225]
_C.DATASET.OBJ_LIST = None


#EfficientDet Cropping Network Configuration
_C.EFFICIENTDET = CN()
_C.EFFICIENTDET.COMPOUND_COEF = 0
_C.EFFICIENTDET.IMG_SIZE = 256
_C.EFFICIENTDET.BATCH_SIZE = 16
_C.EFFICIENTDET.OPTIMIZER = 'adamw'
_C.EFFICIENTDET.LEARNING_RATE = 0.001
_C.EFFICIENTDET.CHECKPOINT_SAVE_INTERVAL = 10
_C.EFFICIENTDET.ONLY_SAVE_ON_IMPROVEMENT = False    #TODO: actually implement
_C.EFFICIENTDET.VAL_INTERVAL = 1
_C.EFFICIENTDET.USE_EARLY_STOPPING = False
_C.EFFICIENTDET.EARLY_STOPPING_MIN_DELTA = 0.002
_C.EFFICIENTDET.EARLY_STOPPING_PATIENCE = 5
_C.EFFICIENTDET.ANCHOR_SCALES = '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
_C.EFFICIENTDET.ANCHOR_RATIOS = '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'

_C.EFFICIENTDET.AUGMENTATION = CN()
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION = CN()
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.ENABLED = True
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.GAUSSIAN_BLUR = CN()
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.GAUSSIAN_BLUR.PROBABILITY = 0.5
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.GAUSSIAN_BLUR.SIGMA = [0,0.5]
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.GAUSSIAN_NOISE = CN()
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.GAUSSIAN_NOISE.PER_CHANNEL_PROBABILITY = 0.6
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.GAUSSIAN_NOISE.SCALE = [0.0, 0.02]
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.LINEAR_CONTRAST = CN()
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.LINEAR_CONTRAST.PROBABILITY = 1.0
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.LINEAR_CONTRAST.SCALE = [0.7,1.0]
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.MULTIPLY = CN()
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.MULTIPLY.PROBABILITY = 0.3
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.MULTIPLY.SCALE = [0.7,1.5]
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.PER_CHANNEL_MULTIPLY = CN()
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.PER_CHANNEL_MULTIPLY.PROBABILITY = 0.3
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.PER_CHANNEL_MULTIPLY.PER_CHANNEL_PROBABILITY = 0.3
_C.EFFICIENTDET.AUGMENTATION.COLOR_MANIPULATION.PER_CHANNEL_MULTIPLY.SCALE = [0.8,1.2]
_C.EFFICIENTDET.AUGMENTATION.MIRROR = CN()
_C.EFFICIENTDET.AUGMENTATION.MIRROR.PROBABILITY = 0.5
_C.EFFICIENTDET.AUGMENTATION.AFFINE_TRANSFORM = CN()
_C.EFFICIENTDET.AUGMENTATION.AFFINE_TRANSFORM.PROBABILITY = 1.0
_C.EFFICIENTDET.AUGMENTATION.AFFINE_TRANSFORM.ROTATION_RANGE = [-60,60]
_C.EFFICIENTDET.AUGMENTATION.AFFINE_TRANSFORM.SCALE_RANGE = [0.8, 1.25]



#EFFICIENTTRACK 2D Tracking Network Configuration
_C.EFFICIENTTRACK = CN()
_C.EFFICIENTTRACK.COMPOUND_COEF = 3
_C.EFFICIENTTRACK.NUM_JOINTS = 23
_C.EFFICIENTTRACK.BOUNDING_BOX_SIZE = 320
_C.EFFICIENTTRACK.BATCH_SIZE = 4
_C.EFFICIENTTRACK.OPTIMIZER = 'adamw'
_C.EFFICIENTTRACK.LEARNING_RATE = 0.001
_C.EFFICIENTTRACK.CHECKPOINT_SAVE_INTERVAL = 10
_C.EFFICIENTTRACK.ONLY_SAVE_ON_IMPROVEMENT = False    #TODO: actually implement
_C.EFFICIENTTRACK.VAL_INTERVAL = 1
_C.EFFICIENTTRACK.USE_EARLY_STOPPING = False
_C.EFFICIENTTRACK.EARLY_STOPPING_MIN_DELTA = 0.002
_C.EFFICIENTTRACK.EARLY_STOPPING_PATIENCE = 5

_C.EFFICIENTTRACK.AUGMENTATION = CN()
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION = CN()
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.ENABLED = True
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.GAUSSIAN_BLUR = CN()
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.GAUSSIAN_BLUR.PROBABILITY = 0.5
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.GAUSSIAN_BLUR.SIGMA = [0,0.5]
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.GAUSSIAN_NOISE = CN()
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.GAUSSIAN_NOISE.PER_CHANNEL_PROBABILITY = 0.6
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.GAUSSIAN_NOISE.SCALE = [0.0, 0.02]
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.LINEAR_CONTRAST = CN()
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.LINEAR_CONTRAST.PROBABILITY = 1.0
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.LINEAR_CONTRAST.SCALE = [0.7,1.0]
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.MULTIPLY = CN()
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.MULTIPLY.PROBABILITY = 0.3
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.MULTIPLY.SCALE = [0.7,1.5]
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.PER_CHANNEL_MULTIPLY = CN()
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.PER_CHANNEL_MULTIPLY.PROBABILITY = 0.3
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.PER_CHANNEL_MULTIPLY.PER_CHANNEL_PROBABILITY = 0.3
_C.EFFICIENTTRACK.AUGMENTATION.COLOR_MANIPULATION.PER_CHANNEL_MULTIPLY.SCALE = [0.8,1.2]
_C.EFFICIENTTRACK.AUGMENTATION.MIRROR = CN()
_C.EFFICIENTTRACK.AUGMENTATION.MIRROR.PROBABILITY = 0.5
_C.EFFICIENTTRACK.AUGMENTATION.AFFINE_TRANSFORM = CN()
_C.EFFICIENTTRACK.AUGMENTATION.AFFINE_TRANSFORM.PROBABILITY = 1.0
_C.EFFICIENTTRACK.AUGMENTATION.AFFINE_TRANSFORM.ROTATION_RANGE = [-60,60]
_C.EFFICIENTTRACK.AUGMENTATION.AFFINE_TRANSFORM.SCALE_RANGE = [0.8, 1.25]


#VORTEX3D 3D Tracking Network Configuration
_C.VORTEX = CN()
_C.VORTEX.ROI_CUBE_SIZE = None
_C.VORTEX.GRID_SPACING = None
_C.VORTEX.GRID_DIM_X = None
_C.VORTEX.GRID_DIM_Y = None
_C.VORTEX.GRID_DIM_Z = None

_C.VORTEX.BATCH_SIZE = 1
_C.VORTEX.OPTIMIZER = 'adamw'
_C.VORTEX.LEARNING_RATE = 0.0005
_C.VORTEX.CHECKPOINT_SAVE_INTERVAL = 5
_C.VORTEX.VAL_INTERVAL = 1
_C.VORTEX.USE_EARLY_STOPPING = False
_C.VORTEX.EARLY_STOPPING_MIN_DELTA = 0.002
_C.VORTEX.EARLY_STOPPING_PATIENCE = 5
