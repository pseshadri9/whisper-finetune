import logging

import hydra
import torch
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from manifest_sns import SNS
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config/", config_name="train.yaml")
def main(config: DictConfig):
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")

    log.info(
        f"Pre-processing files from <{config.data.preprocess.source_path}> to \
            <{config.data.preprocess.destination_path}>"
    )
    metadata = hydra.utils.instantiate(config.data.preprocess)
    config.data.datamodule.metadata = metadata

    log.info(f"Instantiating datamodule <{config.data.datamodule._target_}> from {config.data.datamodule.metadata}")
    data = hydra.utils.instantiate(config.data.datamodule)

    log.info(f"Instantiating trainer with logger: <{config.trainer.logger._target_}>")
    trainer = hydra.utils.instantiate(config.trainer)

    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)
    train_and_evaluate(model, trainer, data, ckpt_path=config.ckpt)


@SNS(subject="ML TRAINING RESULTS for project []")
def train_and_evaluate(
    model: LightningModule, trainer: Trainer, data: LightningDataModule, ckpt_path: str | None = None
):
    log.info(f"STARTING TRAINING OF {model.model.__class__.__name__}")

    if ckpt_path is not None:
        log.info(f"RESUMING FROM CHECKPOINT: {ckpt_path}")

    trainer.fit(model, data, ckpt_path=ckpt_path)

    log.info(f"STARTING EVALUATION OF {model.model.__class__.__name__}")
    best_ckpt = trainer.checkpoint_callback.best_model_path
    return trainer.test(model=model, datamodule=data, ckpt_path=best_ckpt)[0]


if __name__ == "__main__":
    main()
