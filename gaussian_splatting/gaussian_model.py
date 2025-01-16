


class GaussianModel:
    def __init__(self, sh_degree: int, config=None):
        self.sh_degree = sh_degree
        self.config = config

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale