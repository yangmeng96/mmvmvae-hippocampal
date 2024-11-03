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
class ModelConfig:
    device: str = "cuda"
    batch_size: int = 128
    lr: float = 0.001
    epochs: int = 999

    latent_dim: int = 2

    resample_eval: bool = False

    # loss hyperparameters
    beta: float = 1.0

    # weight on N(0,1) in mixed prior
    stdnormweight: float = 0.0

    # network architectures
    # use_resnets: bool = True


@dataclass
class EvalConfig:
    # latent representation
    num_samples_train: int = 10000
    max_iteration: int = 10000
    eval_downstream_task: bool = True

    # coherence
    coherence: bool = True


@dataclass
class DRPMModelConfig(ModelConfig):
    name: str = "drpm"
    # drpm
    n_groups: int = 2
    hard_pi: bool = True
    add_gumbel_noise: bool = False

    # temperature annealing
    init_temp: float = 1.0
    final_temp: float = 0.5
    num_steps_annealing: int = 100000

    # loss hyperparameters
    gamma: float = 3.0
    delta: float = 0.03

    # learning drpm parameters
    learn_const_dist_params: bool = False
    encoders_rpm: bool = True


@dataclass
class JointModelConfig(ModelConfig):
    name: str = "joint"
    aggregation: str = "mopoe"


@dataclass
class MixedPriorModelConfig(ModelConfig):
    name: str = "mixedprior"

@dataclass
class DataConfig:
    name: str = MISSING
    num_workers: int = 8
    # num views
    num_views: int = MISSING 
    dir_clfs_base: str = "/your_dir_clfs_base"

@dataclass
class SPIKEDataConfig(DataConfig):
    name: str = "SPIKE"
    num_views: int = 5
    dir_data: str = "/your_dir_data"
    suffix_clfs: str = "/your_suffix_clfs"

@dataclass
class MyRATSWSLConfig:
    seed: int = 0
    checkpoint_metric: str = "val/loss/loss"
    # logger
    log: LogConfig = MISSING
    # dataset
    dataset: DataConfig = MISSING
    # model
    model: ModelConfig = MISSING
    # eval
    eval: EvalConfig = MISSING
