
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.distributed.elastic.multiprocessing.errors import record

from ..denoising_diffusion_pytorch.pixelnerf_trainer import (
    PixelNeRFModelWrapper,
    Trainer,
)
from ..data_io import get_dataset
from ..PixelNeRF import PixelNeRFModelVanilla
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator

from ..data_io.io_utils import pmgr, FBENV

@hydra.main(
    version_base=None, config_path="../configurations/", config_name="config",
)
def train(cfg: DictConfig):
    # download all necessary models
    if cfg.checkpoint_path is not None:
        cfg.checkpoint_path = pmgr.get_local_path(cfg.checkpoint_path)
    if cfg.lpips_model_path is not None:
        cfg.lpips_model_path = pmgr.get_local_path(cfg.lpips_model_path)

    # dataset
    # initialize the accelerator at the beginning
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        split_batches=True, mixed_precision="no", kwargs_handlers=[ddp_kwargs],
    )

    train_batch_size = cfg.batch_size
    dataset = get_dataset(cfg)
    dl = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )

    # model = PixelNeRFModel(near=1.2, far=4.0, dim=64, dim_mults=(1, 1, 2, 4)).cuda()

    # model_type = "vit"
    # backbone = "vitb_rn50_384"
    # backbone = "vitl16_384"
    # vit_path = get_vit_path(backbone)

    # model_type = "dit"
    model_type = cfg.model_type
    backbone = None
    vit_path = None
    model = PixelNeRFModelVanilla(
        near=dataset.z_near,
        far=dataset.z_far,
        model=model_type,
        backbone=backbone,
        background_color=dataset.background_color,
        viz_type=cfg.dataset.viz_type,
        use_first_pool=cfg.use_first_pool,
        lindisp=cfg.dataset.lindisp,
        path=vit_path,
        use_abs_pose=cfg.use_abs_pose,
        use_high_res_feats=True
    ).cuda()

    modelwrapper = PixelNeRFModelWrapper(
        model, image_size=dataset.image_size, loss_type="l2",  # L1 or L2
        lpips_model_path=cfg.lpips_model_path
    ).cuda()
    print(f"using lr {cfg.lr}")
    trainer = Trainer(
        reconstruction_model=modelwrapper,
        accelerator=accelerator,
        dataloader=dl,
        train_batch_size=train_batch_size,
        train_lr=cfg.lr,
        train_num_steps=cfg.train_num_steps,  # total training steps
        gradient_accumulate_every=cfg.grad_acc_steps,  # gradient accumulation steps
        ema_decay=cfg.ema_decay,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        sample_every=5000,
        wandb_every=cfg.val_every,
        save_every=cfg.save_every,
        num_samples=2,
        warmup_period=1_000,
        checkpoint_path=cfg.checkpoint_path,
        wandb_config=None,
        run_name=cfg.name,
        output_dir=cfg.output_dir
    )

    trainer.train()


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
