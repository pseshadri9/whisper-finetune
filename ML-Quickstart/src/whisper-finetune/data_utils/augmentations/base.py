import torch


class BatchAugmentation(torch.nn.Module):
    """
    Base class for batch augmentations

    Transforms audio per self.transform and alters labels accordingly (eg. pitch shifting)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform_audio = torch.nn.Identity()

    def forward(
        self, batch: tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        audio, labels = batch

        augmented_audio = self.transform_audio(audio)

        return (augmented_audio, labels)


class BatchAugmentor(torch.nn.Module):
    """Augment a batch of audio, labels with a series of augmentations"""

    def __init__(self, augmentor: BatchAugmentation | list[BatchAugmentation] = BatchAugmentation()):
        super().__init__()
        self.augmentor = augmentor

        if isinstance(self.augmentor, list):
            self.augmentor = torch.nn.Sequential(*self.augmentor)

    def forward(
        self, batch: tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        return self.augmentor(batch)
