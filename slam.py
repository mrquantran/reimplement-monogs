import sys
import time
import yaml
from munch import munchify
from argparse import ArgumentParser
from box import Box


import torch
import torch.multiprocessing as mp

from MonoGS.gui import gui_utils, slam_gui
from MonoGS.utils.multiprocessing_utils import FakeQueue
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
        opt_params: GSParams = munchify(config["gs_params"])
        pipeline_params: PipelineParams = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        # ---- Initialize Gaussian Model ----
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")

        # ---- Initialize SLAM Frontend and Backend ----
        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        # ---- Initialize Queue & Process ----
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        frontend_process = mp.Process(target=self.frontend.run)
        self.frontend_process = frontend_process
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue

        backend_process = mp.Process(target=self.backend.run)
        self.backend_process = backend_process
        self.backend.frontend_queue = frontend_queue

        # ---- Initialize Dataset ----
        self.dataset = load_dataset(config)
        self.frontend.dataset = self.dataset

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

    def run(self):
        # if torch.cuda.is_available():
        #     # ---- Initialize CUDA Event ----
        #     start = torch.cuda.Event(enable_timing=True)
        #     end = torch.cuda.Event(enable_timing=True)

        #     start.record()
        #     self.backend_process.start()
        #     self.frontend_process.start()

        #     end.record()
        #     torch.cuda.synchronize()

        self.frontend_process.start()
        self.backend_process.start()

        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)


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
