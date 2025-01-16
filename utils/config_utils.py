import yaml

def load_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from per scene/dataset cfg.
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # merge per dataset cfg. and main cfg.
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively. dict1 get masked by dict2, and we retuen dict1.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

from dataclasses import dataclass
from typing import Optional


@dataclass
class CalibrationConfig:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    distorted: bool
    width: int
    height: int
    depth_scale: float


@dataclass
class DatasetConfig:
    type: str = "tum"
    sensor_type: str = "depth"
    dataset_path: Optional[str] = None
    pcd_downsample: int = 128
    pcd_downsample_init: int = 32
    adaptive_pointsize: bool = True
    point_size: float = 0.01
    calibration: Optional[CalibrationConfig] = None


@dataclass
class LearningRate:
    cam_rot_delta: float = 0.003
    cam_trans_delta: float = 0.001


@dataclass
class Training:
    # Initialization
    init_itr_num: int = 1050
    init_gaussian_update: int = 100
    init_gaussian_reset: int = 500
    init_gaussian_th: float = 0.005
    init_gaussian_extent: int = 30

    # Tracking and Mapping
    tracking_itr_num: int = 100
    mapping_itr_num: int = 150
    gaussian_update_every: int = 150
    gaussian_update_offset: int = 50
    gaussian_th: float = 0.7
    gaussian_extent: float = 1.0
    gaussian_reset: int = 2001
    size_threshold: int = 20
    kf_interval: int = 5
    window_size: int = 8
    pose_window: int = 3
    edge_threshold: float = 1.1
    rgb_boundary_threshold: float = 0.01
    alpha: float = 0.9
    kf_translation: float = 0.08
    kf_min_translation: float = 0.05
    kf_overlap: float = 0.9
    kf_cutoff: float = 0.3
    prune_mode: str = "slam"
    single_thread: bool = False
    spherical_harmonics: bool = False

    # Nested learning rate
    lr: LearningRate = LearningRate()


@dataclass
class ModelParams:
    sh_degree: int = 3
    source_path: str = ""
    model_path: str = ""
    resolution: int = -1
    white_background: bool = False
    device = "cuda"


@dataclass
class GSParams:
    iterations: int = 30000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.001
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002


@dataclass
class PipelineParams:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False


class Config:
    def __init__(self):
        self.dataset = DatasetConfig()
        self.model_params = ModelParams()
        self.training = Training()
