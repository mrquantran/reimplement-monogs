import torch
from .logging_utils import logger
from typing import Union
from .config_utils import Config
from .graphics_utils import getProjectionMatrixFromIntrinsics
from .dataset import TUMDataset, ReplicaDataset, EurocDataset, RealsenseDataset
from .camera_utils import Camera
from typing import Dict


CameraDict = Dict[int, Camera]

class FrontEnd:
    def __init__(self, config: Config):
        self.dataset: Union[
            TUMDataset, ReplicaDataset, EurocDataset, RealsenseDataset
        ] = None
        self.frontend_queue = None
        self.cameras: CameraDict = dict()

        self.config = config

        # ---- Initialize Config ----
        self.initialized = False
        self.kf_indices = []
        self.monocular = False
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        # ---- Initialize Local Map ----
        self.reset = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        # ---- Add RGBD Frame ----
        rgb_boundary_threshold = self.config.training.rgb_boundary_threshold
        self.kf_indices.append(cur_frame_idx)
        viewpoint: CameraDict = self.cameras[cur_frame_idx]

        gt_img = viewpoint.original_image.to(self.device)
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]

        # ---- Add Depth Frame ----
        initial_depth = viewpoint.depth.unsqueeze(0) if isinstance(viewpoint.depth, torch.Tensor) else torch.from_numpy(viewpoint.depth).unsqueeze(0)

        # Move valid_rgb to CPU if needed for masking
        valid_rgb_cpu = valid_rgb.cpu()
        initial_depth[~valid_rgb_cpu] = 0

        return initial_depth[0].cpu().numpy()

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def initialize(self, cur_frame_idx, viewpoint: Camera):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        # # remove everything from the queues
        # while not self.backend_queue.empty():
        #     self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

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

        while True:
            if self.frontend_queue.empty():
                # ---- Finish Processing ----
                if curr_frame_idx >= self.dataset.num_imgs:
                    break

                viewpoint = Camera.init_from_dataset(self.dataset, curr_frame_idx, projection_matrix)

                self.cameras[curr_frame_idx] = viewpoint

                # ---- Initialize Local Map ----
                if self.reset:
                    self.initialize(curr_frame_idx, viewpoint)
                    self.current_window.append(curr_frame_idx)
                    curr_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )
