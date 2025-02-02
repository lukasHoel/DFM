# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.distributed.elastic.multiprocessing.errors import record


from ..denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    GaussianDiffusion,
    Trainer,
)
from ..data_io import get_dataset
from ..data_io.io_utils import pmgr, FBENV

from ..PixelNeRF import PixelNeRFModelCond
import numpy as np
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator


@hydra.main(
    version_base=None, config_path="../configurations/", config_name="config",
)
def train(cfg: DictConfig):
    # download all necessary models
    if cfg.checkpoint_path is not None:
        cfg.checkpoint_path = pmgr.get_local_path(cfg.checkpoint_path)
    if cfg.lpips_model_path is not None:
        cfg.lpips_model_path = pmgr.get_local_path(cfg.lpips_model_path)

    train_settings = get_train_settings(cfg.setting_name, cfg.ngpus)
    cfg.num_context = train_settings["num_context"]
    cfg.num_target = train_settings["num_target"]

    # initialize the accelerator at the beginning
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        split_batches=True, mixed_precision="no", kwargs_handlers=[ddp_kwargs],
    )

    dataset = get_dataset(cfg)
    accelerator.print(f"length dataset {len(dataset)}")

    dl = DataLoader(
        dataset,
        batch_size=train_settings["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        worker_init_fn=lambda id: np.random.seed(id * 4),
    )
    model_type = cfg.model_type
    model = PixelNeRFModelCond(
        near=dataset.z_near,
        far=dataset.z_far,
        model=model_type,
        background_color=dataset.background_color,
        viz_type=cfg.dataset.viz_type,
        use_first_pool=cfg.use_first_pool,
        mode=cfg.mode,
        feats_cond=cfg.feats_cond,
        use_high_res_feats=True,
        render_settings=train_settings,
        use_viewdir=train_settings["use_viewdir"],
        image_size=dataset.image_size,
        use_abs_pose=cfg.use_abs_pose
        # use_viewdir=False,
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=dataset.image_size,
        timesteps=1000,  # number of steps
        sampling_timesteps=10,
        loss_type="l2",  # L1 or L2
        objective="pred_x0",
        beta_schedule="cosine",
        use_guidance=cfg.use_guidance,
        guidance_scale=1.0,
        lpips_model_path=cfg.lpips_model_path,
    ).cuda()

    print(f"using settings {train_settings}")
    warmup_period = 1_000

    print(f"using lr {cfg.lr}")
    trainer = Trainer(
        diffusion,
        accelerator=accelerator,
        dataloader=dl,
        train_batch_size=train_settings["batch_size"],
        train_lr=cfg.lr,
        train_num_steps=cfg.train_num_steps,  # total training steps
        gradient_accumulate_every=cfg.grad_acc_steps,  # gradient accumulation steps
        ema_decay=cfg.ema_decay,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        sample_every=2000,
        wandb_every=cfg.val_every,
        save_every=cfg.save_every,
        num_samples=1,
        warmup_period=warmup_period,
        checkpoint_path=cfg.checkpoint_path,
        wandb_config=None,
        run_name=cfg.name,
        dist_loss_weight=cfg.dataset.dist_loss_weight,
        depth_smooth_loss_weight=cfg.dataset.depth_smooth_loss_weight,
        lpips_loss_weight=cfg.dataset.lpips_loss_weight,
        # rgb_loss_weight=cfg.dataset.rgb_loss_weight,
        cfg=cfg,
        num_context=train_settings["num_context"],
        load_pn=cfg.load_pn,
    )
    trainer.train()


def get_train_settings(name, ngpus):
    if name == "co3d_3ctxt":
        return {
            "n_coarse": 64,
            "n_fine": 64,
            "n_coarse_coarse": 64,
            "n_coarse_fine": 0,
            "num_pixels": int(24 ** 2),
            "batch_size": 1 * ngpus,
            "num_context": 3,
            "num_target": 2,
            "n_feats_out": 64,
            "use_viewdir": False,
            "sampling": "patch",
            "self_condition": False,
            "cnn_refine": False,
            "lindisp": False
        }
    elif name == "co3d_3ctxt_paper":
        return {
            "n_coarse": 64,
            "n_fine": 64,
            "n_coarse_coarse": 64,
            "n_coarse_fine": 0,
            "num_pixels": int(24 ** 2),
            "batch_size": 3 * ngpus,
            "num_context": 3,
            "num_target": 2,
            "n_feats_out": 64,
            "use_viewdir": False,
            "sampling": "patch",
            "self_condition": False,
            "cnn_refine": False,
            "lindisp": False
        }
    elif name == "re":
        return {
            "n_coarse": 64,
            "n_fine": 64,
            "n_coarse_coarse": 32,
            "n_coarse_fine": 0,
            "num_pixels": int(24 ** 2),
            "batch_size": 3 * ngpus,
            "num_context": 1,
            "num_target": 2,
            "n_feats_out": 64,
            "use_viewdir": False,
            "sampling": "patch",
            "cnn_refine": False,
            "self_condition": False,
            "lindisp": False,
        }
    elif name == "re_128res":
        return {
            "n_coarse": 64,
            "n_fine": 64,
            "n_coarse_coarse": 32,
            "n_coarse_fine": 0,
            "num_pixels": int(24 ** 2),
            "batch_size": 2 * ngpus,
            "num_context": 1,
            "num_target": 2,
            "n_feats_out": 64,
            "use_viewdir": False,
            "sampling": "patch",
            "cnn_refine": False,
            "self_condition": False,
            "lindisp": False,
        }
    elif name == "re_2ctxt":
        return {
            "n_coarse": 64,
            "n_fine": 64,
            "n_coarse_coarse": 32,
            "n_coarse_fine": 0,
            "num_pixels": int(24 ** 2),
            "batch_size": 2 * ngpus,
            "num_context": 2,
            "num_target": 2,
            "n_feats_out": 64,
            "use_viewdir": False,
            "sampling": "patch",
            "cnn_refine": False,
            "self_condition": False,
            "lindisp": False,
        }
    elif name == "debug":
        return {
            "n_coarse": 64,
            "n_fine": 0,
            "n_coarse_coarse": 32,
            "n_coarse_fine": 0,
            "num_pixels": 8 ** 2 * 2,
            "batch_size": 1 * ngpus,
            "num_context": 1,
            "num_target": 2,
            "n_feats_out": 64,
        }
    else:
        raise ValueError(f"unknown setting {name}")


@record
def launch_fbcode():
    from uuid import uuid4
    import torch.distributed.launcher as pet

    num_gpus = 8
    lc = pet.LaunchConfig(
        # min, max nodes == 1 since we assume devgpu testing
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=num_gpus,
        rdzv_backend="zeus",
        # run_id just has to be globally unique
        run_id=f"nerf_completion_baseline_dfm_run_{uuid4()}",
        # for fault tolerance; for testing set it to 0 (no fault tolerance)
        max_restarts=0,
        start_method="spawn",
    )

    # The "train" function is called inside the elastic_launch along.
    print("launcher_config:", lc)
    pet.elastic_launch(lc, train)()


if __name__ == "__main__":
    print(f"running {__file__}")
    if FBENV:
        launch_fbcode()
    else:
        train()
