import sys
import yaml
from munch import munchify
from argparse import ArgumentParser
from box import Box


import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_model import GaussianModel

from utils.config_utils import load_config
from utils.logging_utils import logger
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd
from utils.dataset import load_dataset
from utils.config_utils import Config, ModelParams, GSParams, PipelineParams

class SLAM:
    def __init__(self, config: Config, save_dir=None):
        # ---- Initialize Config ----
        self.config = munchify(config)

        model_params: ModelParams = munchify(config["model_params"])
        gs_params: GSParams = munchify(config["gs_params"])
        pipeline_params: PipelineParams = munchify(config["pipeline_params"])

        # ---- Initialize Gaussian Model ----
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)

        # ---- Initialize SLAM Frontend and Backend ----
        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        # ---- Initialize Queue & Process ----
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()
        backend_process = mp.Process(target=self.backend.run)
        frontend_process = mp.Process(target=self.frontend.run)

        self.backend_process = backend_process
        self.frontend_process = frontend_process
        self.frontend.frontend_queue = frontend_queue
        self.backend.frontend_queue = frontend_queue

        # ---- Initialize Dataset ----
        self.dataset = load_dataset(config)
        self.frontend.dataset = self.dataset

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

        self.frontend_process.join()
        self.backend_process.join()

        pass


if __name__ == "__main__":
    # ---- Arguments ----
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file",
    )

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    # ---- Load Configuration ----
    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    # Load inherited configuration
    config: Config = load_config(args.config)
    config = Box(config)

    slam = SLAM(config=config, save_dir=None)
    slam.run()

    logger.info("SLAM Finished")
