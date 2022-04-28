from dataclasses import dataclass, field
from typing import List

@dataclass
class Predict3DParams:
    project_name: str
    recording_path: str
    weights_center_detect: str = 'latest'
    weights_hybridnet: str = 'latest'
    frame_start: int = 0
    number_frames: int = -1
    dataset_name = None
    progress_bar = None
    trt_mode: str = 'off'
    output_dir: str =  ''


@dataclass
class Predict2DParams:
    project_name: str
    recording_path: str
    weights_center_detect: str = 'latest'
    weights_keypoint_detect: str = 'latest'
    frame_start: int = 0
    number_frames: int = -1
    progress_bar = None
    trt_mode: str = 'off'


@dataclass
class CreateVideos3DParams:
    project_name: str
    recording_path: str
    data_csv: str
    frame_start: int = 0
    number_frames: int = -1
    video_cam_list: List[str] = field(default_factory=list)
    dataset_name = None
    progress_bar = None
    output_dir: str =  ''


@dataclass
class CreateVideos2DParams:
    project_name: str
    recording_path: str
    data_csv: str
    frame_start: int = 0
    number_frames: int = -1
    progress_bar = None
