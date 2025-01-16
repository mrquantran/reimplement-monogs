import os
import torch
import trimesh
import cv2
from PIL import Image

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


class TUMParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }

            self.frames.append(frame)

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

        # Depth Parameters
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path)) / self.depth_scale

        # Convert image to tensor and move to appropriate device
        image = torch.from_numpy(image / 255.0).clamp(0.0, 1.0).permute(2, 0, 1)
        pose = torch.from_numpy(pose)

        # Check if CUDA is available and device is set to cuda
        if torch.cuda.is_available() and 'cuda' in self.device:
            image = image.cuda()
            pose = pose.cuda()
            if depth is not None:
                depth = torch.from_numpy(depth).cuda()
        else:
            image = image.cpu()
            pose = pose.cpu()
            if depth is not None:
                depth = torch.from_numpy(depth).cpu()

        image = image.to(dtype=self.dtype)
        pose = pose.to(dtype=self.dtype)

        return image, depth, pose

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
        dataset_path = config.dataset.dataset_path
        parser = TUMParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses

def load_dataset(config: Config):
    if config.dataset.type == "tum":
        return TUMDataset(config)
    else:
        raise ValueError("Unknown dataset type")
