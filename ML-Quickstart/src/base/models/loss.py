import torch


class Loss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args):
        raise NotImplementedError
