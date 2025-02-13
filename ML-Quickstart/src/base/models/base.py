from typing import Literal

import torch

# from constants import *
from lightning import LightningModule
from models.loss import Loss
from models.metrics import Metrics
from models.model_blocks import ModelBlock

LOSS = "loss"
OUTPUT = "output"
TRAIN = "train"
VAL = "val"
TEST = "test"
"""
BASE LIGHTNING MODULE TO TRAIN

This class is effectively a model runner, as intricate model details are expected to be defined in a separate "model_blocks.py" file

Operations in this class are intentionally left basic for easy adaptation to diverse tasks
"""


class LightningModel(LightningModule):
    def __init__(
        self,
        model_type: str,
        loss: str,
        learning_rate: float = 1e-3,
        feature_extractor: torch.nn.Module = None,
        augmentations: torch.nn.Module = torch.nn.Identity(),  # Training augmentations
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor", "augmentations"])

        # Create the model based on the model_type
        if model_type == "":
            self.model = ModelBlock()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create the loss function based on the loss type
        if loss == "":
            self.loss = Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss}")

        # Set the learning rate
        self.lr = learning_rate

        self.feature_extractor = feature_extractor
        self.augmentation = augmentations
        self.metrics = Metrics()

    def to_device(
        self, x: torch.Tensor | dict[str, torch.Tensor], cpu: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Move either tensors or a dict of tensors to specified device
        """
        if cpu:
            if isinstance(x, torch.Tensor):
                return x.cpu()
            else:
                return {k: v.cpu() for k, v in x.items()}
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)
        else:
            return {k: v.to(self.device) for k, v in x.items()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.feature_extractor(x)
        return self.model(output)

    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self(batch)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self(x)

    def common_step(
        self, batch, stage: Literal["train", "val", "test"] = TRAIN
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        x, y = batch
        output = self(x)
        loss = self.loss(output, y)

        """
        temporary fix for nan in training data (or very large values?)
        TODO: compute mean, stdev of training data and normalize
        """
        if torch.isnan(loss):
            return {LOSS: None, OUTPUT: output}

        self.log(f"{stage}_loss", loss, prog_bar=True)
        return {LOSS: loss, OUTPUT: output}

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        audio, labels = batch
        audio = self.augmentations(audio)
        output_dict = self.common_step((audio, labels), stage=TRAIN)
        return output_dict[LOSS]

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        output_dict = self.common_step(batch, stage=VAL)
        self.metric.update(self.to_device(output_dict[OUTPUT], cpu=True), self.to_device(batch[1], cpu=True))
        return output_dict[LOSS]

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        output_dict = self.common_step(batch, stage=TEST)
        self.metric.update(self.to_device(output_dict[OUTPUT], cpu=True), self.to_device(batch[1], cpu=True))
        return output_dict[LOSS]

    def common_epoch_end(self, stage: Literal["train", "val", "test"] = VAL):
        self.log_dict({f"{stage}_{k}": v for k, v in self.metric.compute().items()}, prog_bar=True)

    def on_validation_epoch_end(self):
        self.common_epoch_end(stage=VAL)

    def on_test_epoch_end(self):
        self.common_epoch_end(stage=TEST)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        """
        reduce LR by a factor of 0.9 every 10000 steps
        TODO: move args to hydra config
        """
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 10000},
        }
