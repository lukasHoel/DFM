from omegaconf import DictConfig
from torch.utils.data import Dataset
from .co3d_new import CO3DDataset


def get_path(dataset_name: str) -> str:
    if dataset_name == "realestate10k":
        raise NotImplementedError()
    elif dataset_name in ["CO3D", "CO3DPN"]:
        return "path"


def get_dataset(config: DictConfig) -> Dataset:
    name = config.dataset.name
    del config.dataset.name

    if name == "realestate10k":
        raise NotImplementedError()
    elif name == "CO3D":
        if config.all_class:
            categories = ["apple", "ball",  "bench",  "cake",  "donut",  "hydrant", "plant", "suitcase", "teddybear", "vase"]
        else:
            categories = ["hydrant"]
        return CO3DDataset(
            root=None,
            num_context=config.num_context,
            num_target=config.num_target,
            stage=config.stage,
            scale_aug_ratio=config.scale_aug_ratio,
            image_size=config.image_size,
            categories=categories,
            noise=config.noise,
            eval_10_frame=config.eval_10_frame,
        )
    raise NotImplementedError(f'Dataset "{name}" not supported.')
