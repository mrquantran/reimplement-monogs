import torch
import numpy as np

from .graphics_utils import focal2fov
from .config_utils import Config

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass

class MonocularDataset(BaseDataset):
    def __init__(self, config: Config):
        super().__init__(config)
        calibration = config.dataset.calibration

        # Camera Parameters
        self.fx = calibration.fx
        self.fy = calibration.fy
        self.cx = calibration.cx
        self.cy = calibration.cy
        self.width = calibration.width
        self.height = calibration.height
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)

        # Distortion Parameters
        self.disorted = calibration.distorted
        self.dist_coeffs = np.array(
            [
                calibration.k1,
                calibration.k2,
                calibration.p1,
                calibration.p2,
                calibration.k3,
            ]
        )

class EurocDataset(MonocularDataset):
    def __init__(self, config: Config):
        super().__init__(config)

class RealsenseDataset(MonocularDataset):
    def __init__(self, config: Config):
        super().__init__(config)

class ReplicaDataset(MonocularDataset):
    def __init__(self, config: Config):
        super().__init__(config)

class TUMDataset(MonocularDataset):
    def __init__(self, config: Config):
        super().__init__(config)

def load_dataset(config: Config):
    if config.dataset.type == "tum":
        return TUMDataset(config)
    else:
        raise ValueError("Unknown dataset type")
