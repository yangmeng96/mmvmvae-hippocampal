from dataclasses import dataclass, field
from typing import List, Dict, Optional

from omegaconf import MISSING

@dataclass
class LogConfig:
    # wandb
    wandb_entity: str = "your_wandb_entity"
    wandb_group: str = "your_wandb_group"
    wandb_run_name: str = "your_wandb_run_name"
    wandb_project_name: str = "your_wandb_project_name"
    wandb_log_freq: int = 50
    wandb_offline: bool = False

    # logs
    dir_logs: str = "/your_dir_logs"


@dataclass
class DataConfig:
    name: str = MISSING
    num_workers: int = 8
    # num views
    num_views: int = 5 
    dir_clfs_base: str = "/your_dir_clfs_base"

@dataclass
class SPIKEDataConfig(DataConfig):
    name: str = "SPIKE"
    dir_data: str = "/your_dir_data"
    suffix_clfs: str = "your_suffix_clfs"


@dataclass
class ModelConfig:
    device: str = "cuda"
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 51 


@dataclass
class MyClfRatsConfig:
    seed: int = 0
    checkpoint_metric: str = "val/loss/mean_acc"
    model: ModelConfig = MISSING
    log: LogConfig = MISSING
    dataset: DataConfig = MISSING
