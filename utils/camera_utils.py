import torch
from torch import nn

class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device if torch.cuda.is_available() else "cpu"

        T = torch.eye(4, device=self.device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=self.device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=self.device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=self.device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=self.device)
        )

        self.projection_matrix = projection_matrix.to(device=self.device)

    def update_RT(self, R, T):
        self.R = R.to(self.device)
        self.T = T.to(self.device)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )