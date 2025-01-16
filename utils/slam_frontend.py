import torch
from .logging_utils import logger
from typing import Union
from .config_utils import Config
from .graphics_utils import getProjectionMatrixFromIntrinsics
from .dataset import TUMDataset, ReplicaDataset, EurocDataset, RealsenseDataset
from .camera_utils import Camera


class FrontEnd:
    def __init__(self, config: Config):
        self.dataset: Union[
            TUMDataset, ReplicaDataset, EurocDataset, RealsenseDataset
        ] = None
        self.frontend_queue = None

    def run(self):
        logger.info("Frontend started")
        curr_frame_idx = 0  # Current frame index

        projection_matrix = getProjectionMatrixFromIntrinsics(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)