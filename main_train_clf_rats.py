import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import wandb

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

from MyClfRatsConfig import MyClfRatsConfig
from MyClfRatsConfig import ModelConfig
from MyClfRatsConfig import LogConfig
from MyClfRatsConfig import DataConfig
from MyClfRatsConfig import SPIKEDataConfig

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(group="log", name="log", node=LogConfig)
cs.store(group="model", name="model", node=ModelConfig)
cs.store(group="dataset", name="SPIKE", node=SPIKEDataConfig)
cs.store(name="base_config", node=MyClfRatsConfig)

from utils import dataset_rats

from clfs.rats_clf import ClfRats

@hydra.main(version_base=None, config_path="conf", config_name="config_clf")
def run_experiment(cfg: MyClfRatsConfig):
    print(cfg)
    pl.seed_everything(cfg.seed, workers=True)

    # get data loaders
    train_loader, train_dst, val_loader, val_dst = dataset_rats.get_dataset(cfg)

    # load model
    model = ClfRats(cfg)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.dataset.dir_clfs_base, cfg.dataset.suffix_clfs),
        monitor=cfg.checkpoint_metric,
        mode="max",
        save_last=True,
    )
    wandb_logger = WandbLogger(
        name=cfg.log.wandb_run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        project=cfg.log.wandb_project_name,
        group=cfg.log.wandb_group,
        offline=cfg.log.wandb_offline,
        entity=cfg.log.wandb_entity,
        save_dir=cfg.log.dir_logs,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.model.epochs,
        devices=1,
        accelerator="gpu" if cfg.model.device == "cuda" else cfg.model.device,
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        deterministic=True,
        callbacks=[checkpoint_callback],
    )

    trainer.logger.watch(model, log="all")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    run_experiment()
