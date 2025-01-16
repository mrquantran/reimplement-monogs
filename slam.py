import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_model import GaussianModel

from utils.logging import logger
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd

class SLAM:
    def __init__(self, config, save_dir=None):
        # ---- Initialize Config ----
        self.config = config
        self.sh_degree = 3

        # ---- Initialize Gaussian Model ----
        self.gaussians = GaussianModel(self.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)

        # ---- Initialize SLAM Frontend and Backend ----
        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        # ---- Initialize Process ----
        backend_process = mp.Process(target=self.backend.run)
        frontend_process = mp.Process(target=self.frontend.run)
        self.backend_process = backend_process
        self.frontend_process = frontend_process

    def run(self):
        if torch.cuda.is_available():
            # ---- Initialize CUDA Event ----
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            self.backend_process.start()

            self.frontend_process.start()

            end.record()
            torch.cuda.synchronize()

        self.frontend_process.start()
        self.backend_process.start()

        pass


if __name__ == "__main__":
    slam = SLAM(config={}, save_dir=None)

    slam.run()

    logger.info("SLAM Finished")
