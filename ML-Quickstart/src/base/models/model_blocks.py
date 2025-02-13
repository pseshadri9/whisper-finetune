import torch


class ModelBlock(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = torch.nn.Identity()

    def forward(self, input):
        return self.model(input)
