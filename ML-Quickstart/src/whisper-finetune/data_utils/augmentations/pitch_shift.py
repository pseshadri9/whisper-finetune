from typing import Literal

import torch
from constants import MSPEC_PARAMS
from torchaudio.nn.functional import pitch_shift

from .base import BatchAugmentation


class PitchShift(torch.nn.Module):
    """
    Pitch shift waveforms (batch_size, num_samples) by n_steps

    Custom class (as opposed to torchaudio.transforms.PitchShift)
    to allow for different shifts per batch

    See: https://pytorch.org/audio/stable/generated/torchaudio.functional.pitch_shift.html

    Args:
        sample_rate (int): Sample rate of `waveform`.
        n_fft (int): Size of FFT, creates ``n_fft // 2 + 1`` bins.
        win_length (int): Window size. If None, then ``n_fft`` is used.
        hop_length (int ): Length of hop between STFT windows.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = MSPEC_PARAMS["n_fft"],
        hop_length: int = MSPEC_PARAMS["hop_length"],
        win_length: int = MSPEC_PARAMS["win_length"],
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, waveform: torch.Tensor, n_steps: int) -> torch.Tensor:
        return pitch_shift(
            waveform,
            self.sample_rate,
            n_steps,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )


class BatchPitchShiftAugmentation(BatchAugmentation):
    """
    Pitch shift batched audio and labels

    Args: see PitchShift

    Currently only uniform sampling of shifts is supported

    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = MSPEC_PARAMS["n_fft"],
        hop_length: int = MSPEC_PARAMS["hop_length"],
        win_length: int = MSPEC_PARAMS["win_length"],
        sampler: Literal["uniform"] = "uniform",
    ) -> None:
        super().__init__()
        self.transform = PitchShift(sample_rate, n_fft, hop_length, win_length)

        if sampler == "uniform":
            self.sample = BatchPitchShiftAugmentation._uniform_sample

    @staticmethod
    def _uniform_sample(low: int = 0, high: int = 12) -> int:
        """return one integer in [low, high]"""
        return torch.randint(low, high + 1, (1,)).item()

    def forward(
        self, batch: tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        audio, labels = batch
        shift = self.sample()

        augmented_audio = self.transform(audio, shift)
        labels["frame_roll"] = torch.roll(labels["frame_roll"], shift, dims=-1)
        return (augmented_audio, labels)
