from typing import Literal

import torch
from constants import MSPEC_PARAMS, NUM_CLASSES
from lightning import LightningModule
from models.loss import RegressOnsetOffsetFrameVelocityLoss, RegressPedalLoss
from models.metrics import FramewiseMetrics, NoteLevelMetrics
from models.model_blocks import (
    RegressOnsetOffsetFrameVelocityCRNN,
    RegressPedalCRNN,
)

LOSS = "loss"
OUTPUT = "output"
TRAIN = "train"
VAL = "val"
TEST = "test"


class TranscriptionModel(LightningModule):
    def __init__(
        self,
        model_type: str,
        loss: str,
        learning_rate: float = 1e-3,
        feature_extractor: torch.nn.Module = None,
        augmentations: torch.nn.Module = torch.nn.Identity(),
        num_classes: int = NUM_CLASSES,
        mel_bins: int = MSPEC_PARAMS["n_mels"],
        midfeat: int = 1792,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor", "augmentations"])
        self.num_classes = num_classes

        # Create the model based on the model_type
        if model_type == "RegressOnsetOffsetFrameVelocityCRNN":
            self.model = RegressOnsetOffsetFrameVelocityCRNN(num_classes, mel_bins, midfeat, momentum)
        elif model_type == "RegressPedalCRNN":
            self.model = RegressPedalCRNN(1, mel_bins, midfeat, momentum)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create the loss function based on the loss type
        if loss == "RegressOnsetOffsetFrameVelocityLoss":
            self.loss = RegressOnsetOffsetFrameVelocityLoss()
        elif loss == "RegressPedalLoss":
            self.loss = RegressPedalLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss}")

        # Set the learning rate
        self.lr = learning_rate

        self.feature_extractor = feature_extractor
        self.augmentations = augmentations
        self.frame_metric = FramewiseMetrics(num_classes)
        self.note_metric = NoteLevelMetrics(num_classes)

    def to_device(
        self, x: torch.Tensor | dict[str, torch.Tensor], cpu: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
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
        output = self.feature_extractor(x).transpose(1, 2).unsqueeze(dim=1)  # (batch_size, 1, time_steps, mel_bins)
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
        self.frame_metric.update(self.to_device(output_dict[OUTPUT], cpu=True), self.to_device(batch[1], cpu=True))
        # For now, reserve note metrics for testing for speed
        self.note_metric.update(self.to_device(output_dict[OUTPUT], cpu=True), self.to_device(batch[1], cpu=True))
        return output_dict[LOSS]

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        output_dict = self.common_step(batch, stage=TEST)
        self.frame_metric.update(self.to_device(output_dict[OUTPUT], cpu=True), self.to_device(batch[1], cpu=True))
        self.note_metric.update(self.to_device(output_dict[OUTPUT], cpu=True), self.to_device(batch[1], cpu=True))
        return output_dict[LOSS]

    def common_epoch_end(self, stage: Literal["train", "val", "test"] = VAL):
        self.log_dict({f"{stage}_{k}": v for k, v in self.frame_metric.compute().items()}, prog_bar=True)
        if stage in (TEST,):
            self.log_dict({f"{stage}_{k}": v for k, v in self.note_metric.compute().items()}, prog_bar=True)

    def on_validation_epoch_end(self):
        self.common_epoch_end(stage=VAL)

    def on_test_epoch_end(self):
        self.common_epoch_end(stage=TEST)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        """
        reduce LR by a factor of 0.9 every 3333 steps*

        Original Paper uses 10000 steps per batch size of 12,
        we decrease the steps by a factor of 1/3 to account for a batch size of 36 (3x samples)

        TODO: move args to hydra config
        """
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 15000},
        }
