from typing import Callable

import torch
import torchmetrics


class Metrics(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def apply_to_tensor_dict(x: dict[str, torch.Tensor], func: Callable) -> dict[str, torch.Tensor]:
        return {k: func(v) for k, v in x.items()}

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError
